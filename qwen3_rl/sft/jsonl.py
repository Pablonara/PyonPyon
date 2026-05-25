from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Literal


TraceSchema = Literal["auto", "messages", "text", "rollout_trace"]


def load_jsonl_rows(path: str | Path, *, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object, got {type(row).__name__}")
            rows.append(row)
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def detect_schema(row: dict[str, Any]) -> Literal["messages", "text", "rollout_trace"]:
    if isinstance(row.get("text"), str):
        return "text"
    if isinstance(row.get("raw_text"), str):
        return "rollout_trace"
    if "messages" in row:
        return "messages"
    raise ValueError(
        "Could not infer JSONL schema; expected one of: text, raw_text, messages"
    )


def _json_loads_if_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return json.loads(value)


def normalize_messages(messages: Any) -> list[dict[str, Any]]:
    """Normalize OpenAI-ish trace messages for Qwen's Jinja chat template."""
    messages = _json_loads_if_string(messages)
    if not isinstance(messages, list):
        raise ValueError(f"messages must be a list or JSON string, got {type(messages).__name__}")

    normalized = copy.deepcopy(messages)
    for idx, message in enumerate(normalized):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{idx}] must be an object, got {type(message).__name__}")

        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") if isinstance(tool_call, dict) else None
            if not isinstance(function, dict):
                continue
            args = function.get("arguments")
            if isinstance(args, str):
                try:
                    function["arguments"] = json.loads(args)
                except json.JSONDecodeError:
                    function["arguments"] = {"raw": args}
    return normalized


def row_to_text(
    row: dict[str, Any],
    *,
    tokenizer=None,
    schema: TraceSchema = "auto",
) -> str:
    resolved = detect_schema(row) if schema == "auto" else schema
    if resolved == "text":
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError("text schema requires a string 'text' field")
        return text
    if resolved == "rollout_trace":
        text = row.get("raw_text")
        if not isinstance(text, str):
            raise ValueError("rollout_trace schema requires a string 'raw_text' field")
        return text
    if resolved == "messages":
        if tokenizer is None:
            raise ValueError("messages schema requires a tokenizer with apply_chat_template")
        messages = normalize_messages(row.get("messages"))
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    raise ValueError(f"Unsupported schema: {schema}")


def jsonl_to_text_records(
    path: str | Path,
    *,
    tokenizer=None,
    schema: TraceSchema = "auto",
    max_rows: int | None = None,
    min_reward: float | None = None,
) -> list[dict[str, Any]]:
    records = []
    for row in load_jsonl_rows(path, max_rows=max_rows):
        reward = row.get("reward")
        if min_reward is not None and reward is not None and float(reward) < min_reward:
            continue
        text = row_to_text(row, tokenizer=tokenizer, schema=schema)
        meta = {
            key: row[key]
            for key in (
                "instance_id",
                "run_id",
                "iter_id",
                "group_id",
                "rollout_id",
                "reward",
                "seq_len",
                "n_trainable",
            )
            if key in row
        }
        records.append({"text": text, "meta": meta})
    return records


def make_dataset(records: list[dict[str, Any]]):
    from datasets import Dataset

    return Dataset.from_list([{"text": record["text"]} for record in records])


def token_length_stats(texts: list[str], tokenizer) -> dict[str, int | float]:
    if not texts:
        return {"count": 0, "min": 0, "median": 0, "max": 0}
    lengths = sorted(len(tokenizer.encode(text, add_special_tokens=False)) for text in texts)
    mid = len(lengths) // 2
    median = lengths[mid] if len(lengths) % 2 else (lengths[mid - 1] + lengths[mid]) / 2
    return {
        "count": len(lengths),
        "min": lengths[0],
        "median": median,
        "max": lengths[-1],
    }
