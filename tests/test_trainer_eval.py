import torch

from qwen3_rl.config import RolloutConfig, RunConfig
from qwen3_rl.template.qwen3_5 import QWEN3_5_SPEC
from qwen3_rl.trainer import MultiTurnGRPOTrainer


def test_sampling_defaults():
    assert RolloutConfig().top_p == 0.95
    assert RolloutConfig().top_k is None
    assert RunConfig().eval_temperature == 0.6
    assert RunConfig().eval_top_p == 0.95
    assert RunConfig().eval_top_k == 20


class TinyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 151645
        self._special = {"<|im_end|>": self.eos_token_id}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        result = []
        i = 0
        while i < len(text):
            matched = False
            for token, token_id in self._special.items():
                if text[i:].startswith(token):
                    result.append(token_id)
                    i += len(token)
                    matched = True
                    break
            if not matched:
                result.append(ord(text[i]))
                i += 1
        return result

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        inv = {v: k for k, v in self._special.items()}
        return "".join(inv.get(int(i), chr(int(i))) for i in ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._special[token]

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=False):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return self.encode(text, add_special_tokens=False) if tokenize else text


class EvalEnv:
    tools = None

    def reset(self, seed: int):
        return [{"role": "user", "content": f"problem {seed}"}]

    def reward(self, traj):
        return 1.0


class RecordingBatchBackend:
    def __init__(self, tokenizer):
        self.final_ids = tokenizer.encode("answer<|im_end|>", add_special_tokens=False)
        self.calls = []

    def generate_batch(
        self,
        token_ids_list,
        max_new,
        stop_token_ids,
        stop,
        seeds,
        return_logprobs=False,
        *,
        temperature=None,
        top_p=None,
        top_k=None,
    ):
        self.calls.append({
            "max_new": max_new,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        })
        logprobs = [-0.1] * len(self.final_ids) if return_logprobs else None
        return [(self.final_ids, "eos", logprobs) for _ in token_ids_list]


def test_eval_uses_separate_sampling_config():
    tokenizer = TinyTokenizer()
    backend = RecordingBatchBackend(tokenizer)
    config = RunConfig(
        rollout=RolloutConfig(
            max_turns=1,
            max_tokens_per_turn=128,
            max_total_tokens=2048,
            temperature=1.0,
            top_p=0.95,
            top_k=None,
        ),
        eval_temperature=0.6,
        eval_top_p=0.95,
        eval_top_k=20,
    )
    trainer = MultiTurnGRPOTrainer(
        torch.nn.Linear(1, 1),
        tokenizer,
        EvalEnv(),
        QWEN3_5_SPEC,
        config,
        backend=backend,
    )

    assert trainer._eval(n_problems=2) == 1.0

    assert backend.calls == [{
        "max_new": 128,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    }]
    assert config.rollout.temperature == 1.0
    assert config.rollout.top_p == 0.95
    assert config.rollout.top_k is None
