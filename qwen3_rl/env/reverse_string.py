"""String reversal environment.

Generates random strings of lowercase letters and asks the model to reverse
them.  Difficulty scales with string length: 4-6 chars is tractable for 0.8B,
8-12 is hard, 15+ is very hard.

Binary reward: 1 if the reversed string appears in the output, 0 otherwise.
"""

from __future__ import annotations

import random
import re
import string

from .types import Message, ToolCall, ToolResponse


class ReverseStringEnv:

    def __init__(self, tokenizer, min_len: int = 4, max_len: int = 8):
        self.tokenizer = tokenizer
        self.min_len = min_len
        self.max_len = max_len
        self._original: str = ""
        self._reversed: str = ""

    @property
    def tools(self) -> list[dict]:
        return []

    def reset(self, seed: int) -> list[Message]:
        rng = random.Random(seed)
        length = rng.randint(self.min_len, self.max_len)
        self._original = "".join(rng.choices(string.ascii_lowercase, k=length))
        self._reversed = self._original[::-1]
        question = (
            f'Reverse the following string: "{self._original}"\n'
            f"Reply with ONLY the reversed string, nothing else."
        )
        return [Message(role="user", content=question)]

    def step(self, call: ToolCall) -> tuple[ToolResponse, bool]:
        raise NotImplementedError("ReverseStringEnv has no tools")

    def reward(self, trajectory) -> float:
        text = trajectory.decode_last_gen_turn(self.tokenizer)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return float(self._reversed in text)
