"""GPU test for per-token logp extraction.

Run on GPU:
    UNSLOTH_CE_LOSS_TARGET_GB=4 python -m pytest tests/test_fused_logp.py -v
"""

import os
os.environ["UNSLOTH_CE_LOSS_TARGET_GB"] = "4"

import pytest
import torch
import torch.nn.functional as F


def _has_cuda():
    return torch.cuda.is_available()


@pytest.mark.skipif(not _has_cuda(), reason="requires CUDA")
class TestFusedLogp:

    @pytest.fixture(autouse=True)
    def setup(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_name = "Qwen/Qwen3.5-0.8B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="cuda",
        )
        self.model.eval()

    def test_parity_with_naive(self):
        from qwen3_rl.loss.fused_logp import compute_per_token_logp

        text = "The capital of France is Paris."
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to("cuda")

        logp = compute_per_token_logp(self.model, input_ids)

        # naive reference
        with torch.no_grad():
            logits = self.model(input_ids).logits.float()
        log_probs = F.log_softmax(logits[0], dim=-1)
        naive_logp = torch.zeros(input_ids.shape[1], device="cuda")
        for t in range(input_ids.shape[1] - 1):
            naive_logp[t + 1] = log_probs[t, input_ids[0, t + 1]]

        assert logp.dtype == torch.float32
        assert logp.shape == naive_logp.shape
        assert logp[0].item() == pytest.approx(0.0, abs=1e-6)

        diff = (logp[1:] - naive_logp[1:]).abs()
        max_diff = diff.max().item()
        assert max_diff < 1e-3, f"max diff = {max_diff}"

    def test_shift_by_one(self):
        from qwen3_rl.loss.fused_logp import compute_per_token_logp

        text = "Hello world"
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to("cuda")

        logp = compute_per_token_logp(self.model, input_ids)

        with torch.no_grad():
            logits = self.model(input_ids).logits.float()

        # logits[0, t] predicts token[t+1]
        log_probs = F.log_softmax(logits[0], dim=-1)
        for t in range(input_ids.shape[1] - 1):
            expected = log_probs[t, input_ids[0, t + 1]].item()
            actual = logp[t + 1].item()
            assert abs(expected - actual) < 1e-3, (
                f"Shift mismatch at t={t}: expected={expected:.6f}, got={actual:.6f}"
            )

    def test_output_dtype(self):
        from qwen3_rl.loss.fused_logp import compute_per_token_logp

        input_ids = self.tokenizer.encode("test", return_tensors="pt").to("cuda")
        logp = compute_per_token_logp(self.model, input_ids)
        assert logp.dtype == torch.float32
