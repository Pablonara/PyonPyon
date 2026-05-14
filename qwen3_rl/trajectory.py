from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal

import torch


def require(traj: Trajectory, *fields: str) -> None:
    missing = [f for f in fields if getattr(traj, f) is None]
    if missing:
        raise RuntimeError(
            f"Trajectory missing required fields {missing}; "
            f"check pipeline order (rollout -> logp recompute -> group_norm -> loss)"
        )


@dataclass(frozen=True)
class Trajectory:
    tokens: torch.LongTensor
    mask: torch.BoolTensor
    turns: list[tuple[int, int, str]]
    meta: dict = field(default_factory=dict)
    reward: float | None = None
    advantage: float | None = None
    logp_old: torch.FloatTensor | None = None
    logp_ref: torch.FloatTensor | None = None

    def decode_turn(self, idx: int, tokenizer) -> str:
        start, end, _ = self.turns[idx]
        return tokenizer.decode(self.tokens[start:end], skip_special_tokens=False)

    def decode_last_gen_turn(self, tokenizer) -> str:
        for start, end, kind in reversed(self.turns):
            if kind == "gen":
                return tokenizer.decode(self.tokens[start:end], skip_special_tokens=False)
        return ""

    @property
    def seq_len(self) -> int:
        return self.tokens.shape[0]

    @property
    def n_trainable(self) -> int:
        return int(self.mask.sum().item())

    def replace(self, **kwargs) -> Trajectory:
        return dataclasses.replace(self, **kwargs)


class TrajectoryBuilder:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokens: list[int] = []
        self.mask: list[int] = []
        self.turns: list[tuple[int, int, Literal["prompt", "gen", "env"]]] = []
        self.meta: dict = {}

    def add_prompt(self, text: str) -> None:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        start = len(self.tokens)
        self.tokens.extend(ids)
        self.mask.extend([0] * len(ids))
        self.turns.append((start, len(self.tokens), "prompt"))

    def add_generated(self, token_ids: list[int]) -> None:
        start = len(self.tokens)
        self.tokens.extend(token_ids)
        self.mask.extend([1] * len(token_ids))
        self.turns.append((start, len(self.tokens), "gen"))

    def add_env(self, text: str) -> None:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        start = len(self.tokens)
        self.tokens.extend(ids)
        self.mask.extend([0] * len(ids))
        self.turns.append((start, len(self.tokens), "env"))

    def freeze(self) -> Trajectory:
        return Trajectory(
            tokens=torch.tensor(self.tokens, dtype=torch.long),
            mask=torch.tensor(self.mask, dtype=torch.bool),
            turns=list(self.turns),
            meta=dict(self.meta),
        )
