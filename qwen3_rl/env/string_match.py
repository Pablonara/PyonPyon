"""Arithmetic environment for Phase 1 GRPO training.

Generates random arithmetic problems (2-3 digit) that base models get
~50-70% right. Binary reward: 1 if the correct numeric answer appears
in the model's output, 0 otherwise.

The difficulty is tunable: harder problems (3-digit multiplication) have
lower base-model accuracy → stronger learning signal.
"""

from __future__ import annotations

import random
import re

from .reward_text import strip_think_blocks
from .types import Message, ToolCall, ToolResponse


def _make_problem(rng: random.Random, difficulty: str = "mixed") -> tuple[str, int]:
    """Generate (question_text, correct_answer)."""
    if difficulty == "easy":
        ops = ["+", "-"]
    elif difficulty == "hard":
        ops = ["+", "-", "*"]
    else:
        ops = ["+", "-", "*"]

    op = rng.choice(ops)

    if op == "*":
        a = rng.randint(2, 99)
        b = rng.randint(2, 99)
    elif op == "-":
        a = rng.randint(10, 999)
        b = rng.randint(10, a)
    else:
        a = rng.randint(10, 999)
        b = rng.randint(10, 999)

    answer = eval(f"{a} {op} {b}")
    question = f"What is {a} {op} {b}? Reply with just the number."
    return question, answer


def _extract_number(text: str) -> int | None:
    """Extract the last integer from model output."""
    text = strip_think_blocks(text)
    # find all integers (including negative)
    numbers = re.findall(r"-?\d+", text)
    if not numbers:
        return None
    return int(numbers[-1])


class StringMatchEnv:

    def __init__(self, tokenizer, difficulty: str = "mixed"):
        self.tokenizer = tokenizer
        self.difficulty = difficulty
        self._answer: int = 0

    @property
    def tools(self) -> list[dict]:
        return []

    def reset(self, seed: int) -> list[Message]:
        rng = random.Random(seed)
        question, self._answer = _make_problem(rng, self.difficulty)
        return [Message(role="user", content=question)]

    def step(self, call: ToolCall) -> tuple[ToolResponse, bool]:
        raise NotImplementedError("StringMatchEnv has no tools")

    def reward(self, trajectory) -> float:
        text = trajectory.decode_last_gen_turn(self.tokenizer)
        extracted = _extract_number(text)
        if extracted is None:
            return 0.0
        return float(extracted == self._answer)
