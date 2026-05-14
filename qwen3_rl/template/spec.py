from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ..env.types import ToolCall, ToolResponse


@dataclass
class TemplateSpec:
    name: str
    assistant_open: str
    assistant_close: str
    tool_block_open: str
    tool_resp_open: str
    tool_resp_close: str
    tool_resp_between: str
    tool_block_close: str
    eos_token: str
    stop_strings: list[str]
    parse_tool_calls: Callable[[str], list[ToolCall] | None]
    format_tool_response: Callable[[ToolResponse], str]
