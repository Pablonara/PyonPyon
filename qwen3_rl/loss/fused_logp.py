"""Per-token log-probability extraction via chunked matmul + gather.

This avoids materializing the full (S, V) logits tensor. For V=248K and
S=77K that would be ~38 GB — guaranteed OOM.

Pattern: hidden_states @ lm_head.T in chunks, gather the target token's
logit, subtract logsumexp. Same approach as unsloth-zoo's
chunked_hidden_states_selective_log_softmax but with explicit shift-by-1
handling and FP32 output guarantees.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn


_VALID_DTYPES = {torch.float32, torch.bfloat16, torch.float16}


def _chunked_logp(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_ids: torch.Tensor,
    n_chunks: int = 4,
) -> torch.Tensor:
    flat_h = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_targets = target_ids.reshape(-1)

    chunks_h = torch.chunk(flat_h, chunks=n_chunks, dim=0)
    chunks_t = torch.chunk(flat_targets, chunks=n_chunks, dim=0)

    parts = []
    for h_chunk, t_chunk in zip(chunks_h, chunks_t):
        logits = h_chunk.to(lm_head_weight.dtype) @ lm_head_weight.t()
        logits = logits.to(torch.float32)
        selected = torch.gather(logits, dim=-1, index=t_chunk.unsqueeze(-1)).squeeze(-1)
        lse = torch.logsumexp(logits, dim=-1)
        parts.append(selected - lse)

    return torch.cat(parts)


@torch.no_grad()
def compute_per_token_logp(
    model: nn.Module,
    input_ids: torch.LongTensor,
    n_chunks: int = 4,
) -> torch.FloatTensor:
    """Compute per-token log p(token[t+1] | tokens[:t+1]) for the full sequence.

    Returns a (S,) FP32 tensor. Position 0 is always 0.0 (no prediction for
    the first token). Position t (for t >= 1) holds log p(token[t] | token[:t]).

    The shift-by-1 is handled here: logits[t] predicts tokens[t+1], so
    logp[t+1] = log_softmax(logits[t])[tokens[t+1]].
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    outputs = model(input_ids, output_hidden_states=True, use_cache=False)
    hidden = outputs.hidden_states[-1]

    lm_head = model.get_output_embeddings()
    assert lm_head is not None, "Model has no output embeddings (lm_head)"
    weight = lm_head.weight
    assert weight.dtype in _VALID_DTYPES, (
        f"lm_head dtype {weight.dtype} not in {_VALID_DTYPES}"
    )

    # shift-by-1: hidden[:-1] predicts tokens[1:]
    hidden_shifted = hidden[:, :-1, :].contiguous()
    targets = input_ids[:, 1:].contiguous()

    logp_shifted = _chunked_logp(hidden_shifted, weight, targets, n_chunks)
    logp_shifted = logp_shifted.to(torch.float32)

    # pad position 0 with 0.0 to align with original sequence
    S = input_ids.shape[1]
    logp = torch.zeros(S, dtype=torch.float32, device=input_ids.device)
    logp[1:] = logp_shifted

    return logp
