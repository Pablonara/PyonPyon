from qwen3_rl.env.reward_text import extract_think_text, strip_think_blocks
from qwen3_rl.env.string_match import _extract_number


def test_strip_think_blocks_removes_complete_and_unclosed_blocks():
    assert strip_think_blocks("<think>123</think>\nanswer 4") == "\nanswer 4"
    assert strip_think_blocks("prefix <think>123") == "prefix "


def test_extract_number_ignores_thinking_numbers():
    assert _extract_number("<think>999</think>\nThe answer is 4") == 4
    assert _extract_number("<think>999") is None


def test_extract_think_text_handles_complete_and_unclosed_blocks():
    assert extract_think_text("a<think>one</think>b<think>\ntwo") == "one\ntwo"
