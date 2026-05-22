"""Unit tests for template spec and parse_tool_calls."""

import pytest
from qwen3_rl.template.qwen3_5 import parse_tool_calls, QWEN3_5_SPEC


def test_no_tool_calls():
    assert parse_tool_calls("Just a normal response with no tools.") is None
    assert parse_tool_calls("") is None


def test_single_tool_call():
    text = (
        '<tool_call>\n'
        '<function=read_file>\n'
        '<parameter=path>\n'
        '/tmp/test.py\n'
        '</parameter>\n'
        '</function>\n'
        '</tool_call>'
    )
    result = parse_tool_calls(text)
    assert result is not None
    assert len(result) == 1
    assert result[0].name == "read_file"
    assert result[0].arguments == {"path": "/tmp/test.py"}
    assert "<tool_call>" in result[0].raw_text


def test_multiple_parameters():
    text = (
        '<tool_call>\n'
        '<function=write_file>\n'
        '<parameter=path>\n'
        '/tmp/output.txt\n'
        '</parameter>\n'
        '<parameter=content>\n'
        'Hello, world!\nSecond line.\n'
        '</parameter>\n'
        '</function>\n'
        '</tool_call>'
    )
    result = parse_tool_calls(text)
    assert result is not None
    assert len(result) == 1
    assert result[0].name == "write_file"
    assert result[0].arguments["path"] == "/tmp/output.txt"
    assert "Hello, world!" in result[0].arguments["content"]


def test_multiple_tool_calls():
    text = (
        'Let me check both files.\n\n'
        '<tool_call>\n'
        '<function=read_file>\n'
        '<parameter=path>\n'
        '/tmp/a.py\n'
        '</parameter>\n'
        '</function>\n'
        '</tool_call>\n'
        '<tool_call>\n'
        '<function=read_file>\n'
        '<parameter=path>\n'
        '/tmp/b.py\n'
        '</parameter>\n'
        '</function>\n'
        '</tool_call>'
    )
    result = parse_tool_calls(text)
    assert result is not None
    assert len(result) == 2
    assert result[0].arguments["path"] == "/tmp/a.py"
    assert result[1].arguments["path"] == "/tmp/b.py"


def test_json_parameter_value():
    text = (
        '<tool_call>\n'
        '<function=api_call>\n'
        '<parameter=body>\n'
        '{"key": "value", "count": 42}\n'
        '</parameter>\n'
        '</function>\n'
        '</tool_call>'
    )
    result = parse_tool_calls(text)
    assert result is not None
    assert result[0].arguments["body"] == {"key": "value", "count": 42}


def test_spec_boundary_strings():
    spec = QWEN3_5_SPEC
    assert spec.name == "qwen3_5"
    assert "<|im_start|>assistant" in spec.assistant_open
    assert "<think>" in spec.assistant_open
    assert "<|im_end|>" in spec.assistant_close
    assert "<|im_start|>user" in spec.tool_block_open
    assert "<tool_response>" in spec.tool_resp_open
    assert "</tool_response>" in spec.tool_resp_close
    assert spec.stop_strings == []


def test_tool_call_with_think_block():
    text = (
        '</think>\n\n'
        "I'll read that file.\n\n"
        '<tool_call>\n'
        '<function=read_file>\n'
        '<parameter=path>\n'
        '/tmp/main.py\n'
        '</parameter>\n'
        '</function>\n'
        '</tool_call>'
    )
    result = parse_tool_calls(text)
    assert result is not None
    assert len(result) == 1
    assert result[0].name == "read_file"
