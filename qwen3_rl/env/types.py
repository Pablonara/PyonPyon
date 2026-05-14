from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    raw_text: str


@dataclass(frozen=True)
class ToolResponse:
    content: str
    is_error: bool = False
    metadata: dict = field(default_factory=dict)


class Message(TypedDict, total=False):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[dict]
    reasoning_content: str
