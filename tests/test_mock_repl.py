"""Unit tests for MockREPLEnv."""

import pytest
import torch
from unittest.mock import MagicMock

from qwen3_rl.env.mock_repl import MockREPLEnv, _exec_sandbox
from qwen3_rl.env.types import ToolCall
from qwen3_rl.trajectory import Trajectory


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.encode = lambda text, add_special_tokens=True: list(range(len(text.split())))
    return tok


@pytest.fixture
def env(mock_tokenizer):
    return MockREPLEnv(mock_tokenizer)


# --- Tool spec ---

def test_tools_returns_python_spec(env):
    tools = env.tools
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "python"
    assert "code" in tools[0]["function"]["parameters"]["properties"]


# --- Reset ---

def test_reset_returns_user_message(env):
    msgs = env.reset(seed=42)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert len(msgs[0]["content"]) > 0


def test_reset_deterministic(env):
    msgs1 = env.reset(seed=123)
    msgs2 = env.reset(seed=123)
    assert msgs1[0]["content"] == msgs2[0]["content"]


def test_reset_different_seeds_vary(env):
    """Different seeds should produce different problems (at least sometimes)."""
    contents = set()
    for seed in range(20):
        msgs = env.reset(seed=seed)
        contents.add(msgs[0]["content"])
    # With 5 problem types and random params, 20 seeds should give > 1 unique problem
    assert len(contents) > 1


# --- Step: valid code ---

def test_step_executes_code(env):
    env.reset(seed=0)
    call = ToolCall(name="python", arguments={"code": "print(2 + 3)"}, raw_text="")
    resp, done = env.step(call)
    assert resp.content == "5"
    assert resp.is_error is False
    assert done is False


def test_step_expression_returns_repr(env):
    env.reset(seed=0)
    call = ToolCall(name="python", arguments={"code": "2 + 3"}, raw_text="")
    resp, done = env.step(call)
    assert resp.content == "5"
    assert resp.is_error is False


def test_step_variable_persists(env):
    env.reset(seed=0)
    call1 = ToolCall(name="python", arguments={"code": "x = 42"}, raw_text="")
    env.step(call1)
    call2 = ToolCall(name="python", arguments={"code": "print(x * 2)"}, raw_text="")
    resp2, _ = env.step(call2)
    assert resp2.content == "84"


# --- Step: error handling ---

def test_step_bad_code_returns_error(env):
    env.reset(seed=0)
    call = ToolCall(name="python", arguments={"code": "1/0"}, raw_text="")
    resp, done = env.step(call)
    assert resp.is_error is True
    assert "ZeroDivisionError" in resp.content
    assert done is False


def test_step_syntax_error(env):
    env.reset(seed=0)
    call = ToolCall(name="python", arguments={"code": "def f(:"}, raw_text="")
    resp, done = env.step(call)
    assert resp.is_error is True
    assert "SyntaxError" in resp.content


def test_step_unknown_tool(env):
    env.reset(seed=0)
    call = ToolCall(name="bash", arguments={"code": "ls"}, raw_text="")
    resp, done = env.step(call)
    assert resp.is_error is True
    assert "Unknown tool" in resp.content


def test_step_empty_code(env):
    env.reset(seed=0)
    call = ToolCall(name="python", arguments={"code": ""}, raw_text="")
    resp, done = env.step(call)
    assert resp.is_error is True
    assert "empty" in resp.content.lower()


def test_step_import_blocked(env):
    env.reset(seed=0)
    call = ToolCall(name="python", arguments={"code": "import os"}, raw_text="")
    resp, done = env.step(call)
    assert resp.is_error is True


# --- Sandbox ---

def test_sandbox_no_builtins_leak():
    ns = {}
    # __import__ should not be available
    with pytest.raises(Exception):
        _exec_sandbox("__import__('os')", ns)


def test_sandbox_safe_builtins():
    ns = {}
    result = _exec_sandbox("print(sum([1, 2, 3]))", ns)
    assert result == "6"


def test_sandbox_multiline():
    ns = {}
    code = "for i in range(5):\n    print(i)"
    result = _exec_sandbox(code, ns)
    assert result == "0\n1\n2\n3\n4"


def test_sandbox_timeout():
    with pytest.raises(TimeoutError):
        _exec_sandbox("while True:\n    pass", {}, timeout=0.05)


# --- Reward ---

def _make_trajectory(tokenizer, decode_text: str) -> Trajectory:
    """Create a minimal trajectory whose decode_last_gen_turn returns decode_text."""
    tok = MagicMock()
    tok.decode = MagicMock(return_value=decode_text)
    return Trajectory(
        tokens=torch.tensor([1, 2, 3]),
        mask=torch.tensor([False, True, True]),
        turns=[(0, 1, "prompt"), (1, 3, "gen")],
    ), tok


def test_reward_correct(env):
    env.reset(seed=42)
    answer = env._answer
    traj, tok = _make_trajectory(None, f"The answer is {answer}")
    env.tokenizer = tok
    assert env.reward(traj) == 1.0


def test_reward_wrong(env):
    env.reset(seed=42)
    wrong = env._answer + 999
    traj, tok = _make_trajectory(None, f"The answer is {wrong}")
    env.tokenizer = tok
    assert env.reward(traj) == 0.0


def test_reward_no_number(env):
    env.reset(seed=42)
    traj, tok = _make_trajectory(None, "I don't know")
    env.tokenizer = tok
    assert env.reward(traj) == 0.0


def test_reward_strips_think_block(env):
    env.reset(seed=42)
    answer = env._answer
    traj, tok = _make_trajectory(None, f"<think>999999</think>\nThe answer is {answer}")
    env.tokenizer = tok
    assert env.reward(traj) == 1.0


# --- Multi-step interaction ---

def test_multi_step_interaction(env):
    """Full multi-step: reset -> step(compute part) -> step(compute more) -> check."""
    env.reset(seed=7)
    # Step 1: do some computation
    call1 = ToolCall(name="python", arguments={"code": "a = 17 * 23\nprint(a)"}, raw_text="")
    resp1, done1 = env.step(call1)
    assert resp1.is_error is False
    assert done1 is False

    # Step 2: do more computation
    call2 = ToolCall(name="python", arguments={"code": "b = 45 * 12\nprint(a + b)"}, raw_text="")
    resp2, done2 = env.step(call2)
    assert resp2.is_error is False
    assert done2 is False

    # Variables persisted across steps
    assert "a" in env._namespace


# --- Env Protocol compliance ---

def test_implements_env_protocol(env):
    from qwen3_rl.env.base import Env
    assert isinstance(env, Env)


# --- Namespace isolation across resets ---

def test_reset_clears_namespace(env):
    env.reset(seed=0)
    call = ToolCall(name="python", arguments={"code": "x = 999"}, raw_text="")
    env.step(call)
    assert "x" in env._namespace

    env.reset(seed=1)
    assert "x" not in env._namespace
