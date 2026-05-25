import json

import pytest

from qwen3_rl.sft.jsonl import (
    detect_schema,
    jsonl_to_text_records,
    normalize_messages,
    row_to_text,
    token_length_stats,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        assert add_generation_prompt is False
        parts = []
        for message in messages:
            parts.append(f"<{message['role']}>:{message.get('content', '')}")
            for tool_call in message.get("tool_calls") or []:
                function = tool_call["function"]
                parts.append(f"<tool:{function['name']}:{function['arguments']!r}>")
        return "\n".join(parts)

    def encode(self, text, add_special_tokens=False):
        return text.split()


def test_detect_schema_prefers_text_then_raw_text_then_messages():
    assert detect_schema({"text": "hello", "messages": []}) == "text"
    assert detect_schema({"raw_text": "trace", "messages": []}) == "rollout_trace"
    assert detect_schema({"messages": []}) == "messages"
    with pytest.raises(ValueError):
        detect_schema({"unknown": True})


def test_normalize_messages_parses_tool_arguments():
    messages = [{
        "role": "assistant",
        "content": "thinking",
        "thought": "hidden reasoning",
        "tool_calls": [{
            "function": {
                "name": "bash",
                "arguments": "{\"command\": \"pytest\"}",
            }
        }],
    }]

    normalized = normalize_messages(json.dumps(messages))

    assert normalized[0]["tool_calls"][0]["function"]["arguments"] == {
        "command": "pytest"
    }
    assert normalized[0]["reasoning_content"] == "hidden reasoning"
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == "{\"command\": \"pytest\"}"


def test_row_to_text_supports_all_schemas():
    tokenizer = FakeTokenizer()
    assert row_to_text({"text": "plain"}, schema="auto") == "plain"
    assert row_to_text({"raw_text": "trace"}, schema="auto") == "trace"

    text = row_to_text(
        {"messages": [{"role": "user", "content": "fix bug"}]},
        tokenizer=tokenizer,
        schema="auto",
    )
    assert text == "<user>:fix bug"


def test_jsonl_to_text_records_filters_reward(tmp_path):
    path = tmp_path / "traces.jsonl"
    rows = [
        {"raw_text": "bad", "reward": 0.0, "instance_id": "a"},
        {"raw_text": "good", "reward": 1.0, "instance_id": "b"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    records = jsonl_to_text_records(path, schema="rollout_trace", min_reward=0.5)

    assert records == [{"text": "good", "meta": {"instance_id": "b", "reward": 1.0}}]


def test_token_length_stats():
    stats = token_length_stats(["a b c", "d"], FakeTokenizer())

    assert stats == {"count": 2, "min": 1, "median": 2.0, "max": 3}
