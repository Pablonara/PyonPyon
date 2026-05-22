from __future__ import annotations

import re


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*", re.DOTALL)
_THINK_CONTENT_RE = re.compile(r"<think>\n?(.*?)(?:</think>|$)", re.DOTALL)


def strip_think_blocks(text: str) -> str:
    text = _THINK_BLOCK_RE.sub("", text)
    return _UNCLOSED_THINK_RE.sub("", text)


def extract_think_text(text: str) -> str:
    return "\n".join(match.group(1) for match in _THINK_CONTENT_RE.finditer(text))
