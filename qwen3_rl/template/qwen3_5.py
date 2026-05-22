from __future__ import annotations

import json
import re
from typing import Any

from ..env.types import ToolCall, ToolResponse
from .spec import TemplateSpec

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAM_RE = re.compile(
    r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>",
    re.DOTALL,
)


def parse_tool_calls(text: str) -> list[ToolCall] | None:
    matches = list(_TOOL_CALL_RE.finditer(text))
    if not matches:
        return None

    calls = []
    for m in matches:
        name = m.group(1).strip()
        body = m.group(2)
        args: dict[str, Any] = {}
        for pm in _PARAM_RE.finditer(body):
            key = pm.group(1).strip()
            val = pm.group(2).strip()
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                pass
            args[key] = val
        calls.append(ToolCall(name=name, arguments=args, raw_text=m.group(0)))
    return calls


def format_tool_response(resp: ToolResponse) -> str:
    return resp.content


QWEN3_5_SPEC = TemplateSpec(
    name="qwen3_5",
    assistant_open="<|im_start|>assistant\n<think>\n",
    assistant_close="<|im_end|>\n",
    tool_block_open="<|im_start|>user",
    tool_resp_open="\n<tool_response>\n",
    tool_resp_close="\n</tool_response>",
    tool_resp_between="",
    tool_block_close="<|im_end|>\n",
    eos_token="<|im_end|>",
    stop_strings=[],
    parse_tool_calls=parse_tool_calls,
    format_tool_response=format_tool_response,
)
