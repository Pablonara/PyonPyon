from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .types import Message, ToolCall, ToolResponse


@runtime_checkable
class Env(Protocol):

    @property
    def tools(self) -> list[dict]:
        ...

    def reset(self, seed: int) -> list[Message]:
        ...

    def step(self, call: ToolCall) -> tuple[ToolResponse, bool]:
        ...

    def reward(self, trajectory) -> float:
        ...
