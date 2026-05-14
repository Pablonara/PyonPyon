from __future__ import annotations

import torch
import torch.nn as nn


class HFBackend:

    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate(
        self,
        token_ids: list[int],
        max_new: int,
        stop_token_ids: list[int],
        stop: list[str],
        seed: int,
    ) -> tuple[list[int], str]:
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        torch.manual_seed(seed)
        if cuda_state is not None:
            torch.cuda.manual_seed(seed)

        out = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            eos_token_id=stop_token_ids,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )

        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)

        full_seq = out.sequences[0]
        gen_ids = full_seq[len(token_ids):].tolist()

        if len(gen_ids) == 0:
            return gen_ids, "eos"

        # determine finish reason
        if gen_ids[-1] in stop_token_ids:
            finish = "eos"
        elif len(gen_ids) >= max_new:
            finish = "length"
        else:
            finish = "stop"

        # check for stop strings — scan token-by-token to find the
        # earliest position where a stop string completes, then truncate
        # the token list there. Never decode-then-re-encode (invariant #2).
        if stop and finish != "eos":
            text_so_far = ""
            for i, tid in enumerate(gen_ids):
                text_so_far = self.tokenizer.decode(
                    gen_ids[: i + 1], skip_special_tokens=False
                )
                for s in stop:
                    if s in text_so_far:
                        gen_ids = gen_ids[: i + 1]
                        finish = "stop"
                        break
                if finish == "stop":
                    break

        return gen_ids, finish
