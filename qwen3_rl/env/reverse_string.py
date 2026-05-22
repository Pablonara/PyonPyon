"""String reversal environment.

Generates random strings of lowercase letters and asks the model to reverse
them.  Difficulty scales with string length: 4-6 chars is tractable for 0.8B,
8-12 is hard, 15+ is very hard.

Binary reward: 1 if the reversed string appears in the output, 0 otherwise.
"""

from __future__ import annotations

import random
import string

from .reward_text import extract_think_text, strip_think_blocks
from .types import Message, ToolCall, ToolResponse


class ReverseStringEnv:

    def __init__(
        self,
        tokenizer,
        min_len: int = 4,
        max_len: int = 8,
        *,
        hard_min_len: int | None = None,
        hard_max_len: int | None = None,
        hard_prob: float = 0.0,
        correct_think_bonus: float = 0.0,
        think_bonus_cap_tokens: int = 1024,
        wrong_length_penalty: float = 0.0,
        wrong_length_cap_tokens: int = 4096,
    ):
        if min_len < 1 or max_len < min_len:
            raise ValueError("reverse easy length range must satisfy 1 <= min_len <= max_len")
        if hard_prob < 0.0 or hard_prob > 1.0:
            raise ValueError("reverse hard_prob must be in [0, 1]")
        if (hard_min_len is None) != (hard_max_len is None):
            raise ValueError("reverse hard_min_len and hard_max_len must be set together")
        if hard_min_len is not None and (hard_min_len < 1 or hard_max_len < hard_min_len):
            raise ValueError(
                "reverse hard length range must satisfy 1 <= hard_min_len <= hard_max_len"
            )
        if correct_think_bonus < 0.0:
            raise ValueError("reverse correct_think_bonus must be >= 0")
        if wrong_length_penalty < 0.0:
            raise ValueError("reverse wrong_length_penalty must be >= 0")
        if think_bonus_cap_tokens < 1 or wrong_length_cap_tokens < 1:
            raise ValueError("reverse reward length caps must be >= 1")

        self.tokenizer = tokenizer
        self.min_len = min_len
        self.max_len = max_len
        self.hard_min_len = hard_min_len
        self.hard_max_len = hard_max_len
        self.hard_prob = hard_prob
        self.correct_think_bonus = correct_think_bonus
        self.think_bonus_cap_tokens = think_bonus_cap_tokens
        self.wrong_length_penalty = wrong_length_penalty
        self.wrong_length_cap_tokens = wrong_length_cap_tokens
        self._original: str = ""
        self._reversed: str = ""
        self.last_metadata: dict = {}
        self.last_reward_metadata: dict = {}

    @property
    def tools(self) -> list[dict]:
        return []

    def reset(self, seed: int) -> list[Message]:
        rng = random.Random(seed)
        use_hard = (
            self.hard_min_len is not None
            and self.hard_max_len is not None
            and rng.random() < self.hard_prob
        )
        min_len = self.hard_min_len if use_hard else self.min_len
        max_len = self.hard_max_len if use_hard else self.max_len
        assert min_len is not None and max_len is not None
        length = rng.randint(min_len, max_len)
        self._original = "".join(rng.choices(string.ascii_lowercase, k=length))
        self._reversed = self._original[::-1]
        self.last_metadata = {
            "env": "reverse_string",
            "difficulty": "hard" if use_hard else "easy",
            "length": length,
            "original": self._original,
            "target": self._reversed,
            "reward_config": {
                "correct_think_bonus": self.correct_think_bonus,
                "think_bonus_cap_tokens": self.think_bonus_cap_tokens,
                "wrong_length_penalty": self.wrong_length_penalty,
                "wrong_length_cap_tokens": self.wrong_length_cap_tokens,
            },
        }
        question = (
            f'Reverse the following string: "{self._original}"\n'
            f"Reply with ONLY the reversed string, nothing else."
        )
        return [Message(role="user", content=question)]

    def step(self, call: ToolCall) -> tuple[ToolResponse, bool]:
        raise NotImplementedError("ReverseStringEnv has no tools")

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            return len(text)
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def reward(self, trajectory) -> float:
        raw_text = trajectory.decode_last_gen_turn(self.tokenizer)
        answer_text = strip_think_blocks(raw_text)
        think_tokens = self._count_tokens(extract_think_text(raw_text))
        gen_tokens = int(getattr(trajectory, "n_trainable", 0))
        correct = self._reversed in answer_text

        if correct:
            think_fraction = min(think_tokens, self.think_bonus_cap_tokens) / (
                self.think_bonus_cap_tokens
            )
            shaping = self.correct_think_bonus * think_fraction
            reward = 1.0 + shaping
        else:
            length_fraction = min(gen_tokens, self.wrong_length_cap_tokens) / (
                self.wrong_length_cap_tokens
            )
            shaping = -self.wrong_length_penalty * length_fraction
            reward = shaping

        self.last_reward_metadata = {
            "correct": correct,
            "think_tokens": think_tokens,
            "gen_tokens": gen_tokens,
            "shaping": shaping,
            "reward": reward,
        }
        return float(reward)
