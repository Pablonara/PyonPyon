"""CPU-only tests for multi-turn rollout trajectory structure.

Uses a mock backend that returns predetermined token sequences containing
tool calls. Verifies mask patterns, turn boundaries, and truncation.
"""

from __future__ import annotations

import pytest
import torch
from typing import Optional
from unittest.mock import MagicMock

from qwen3_rl.config import RolloutConfig
from qwen3_rl.env.types import Message, ToolCall, ToolResponse
from qwen3_rl.rollout.base import MultiTurnRollout
from qwen3_rl.template.qwen3_5 import QWEN3_5_SPEC
from qwen3_rl.trajectory import Trajectory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockBackend:
    """Returns predetermined (token_ids, finish_reason) pairs in order."""

    def __init__(self, responses: list[tuple[list[int], str]]):
        self._responses = list(responses)
        self._idx = 0

    def generate(self, token_ids, max_new, stop_token_ids, stop, seed):
        if self._idx >= len(self._responses):
            raise RuntimeError("MockBackend exhausted its responses")
        resp = self._responses[self._idx]
        self._idx += 1
        return resp


class SimpleTokenizer:
    """Minimal tokenizer that encodes each character as its ord value.

    This avoids any dependency on real model tokenizers while giving
    deterministic encode/decode round-trips.
    """

    def __init__(self):
        self.pad_token_id = 0
        self._special = {"<|im_end|>": 151645}
        self._special_inv = {v: k for k, v in self._special.items()}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        result = []
        i = 0
        while i < len(text):
            matched = False
            for tok_str, tok_id in self._special.items():
                if text[i:].startswith(tok_str):
                    result.append(tok_id)
                    i += len(tok_str)
                    matched = True
                    break
            if not matched:
                result.append(ord(text[i]))
                i += 1
        return result

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        chars = []
        for tid in ids:
            if tid in self._special_inv:
                if not skip_special_tokens:
                    chars.append(self._special_inv[tid])
            else:
                chars.append(chr(tid))
        return "".join(chars)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._special.get(token, ord(token[0]))

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=False):
        parts = []
        if tools:
            parts.append("<|im_start|>system\nYou have tools.\n<|im_end|>\n")
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        return "".join(parts)


class MockEnv:
    """Env that returns a user message and handles tool calls."""

    def __init__(self, responses: list[tuple[ToolResponse, bool]] | None = None):
        self._responses = list(responses) if responses else []
        self._resp_idx = 0
        self._reward_val = 0.5

    @property
    def tools(self) -> list[dict]:
        return [{"type": "function", "function": {"name": "python",
                "description": "Run code", "parameters": {"type": "object",
                "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]

    def reset(self, seed: int) -> list[Message]:
        self._resp_idx = 0
        return [Message(role="user", content="Compute 2+2")]

    def step(self, call: ToolCall) -> tuple[ToolResponse, bool]:
        if self._resp_idx < len(self._responses):
            resp = self._responses[self._resp_idx]
            self._resp_idx += 1
            return resp
        return ToolResponse(content="4"), False

    def reward(self, trajectory) -> float:
        return self._reward_val


def _make_tool_call_text() -> str:
    """A valid tool call in Qwen3.5 format (without trailing </tool_call>
    since the backend stop string cuts there, but including the tag for
    the template parser to match)."""
    return (
        '</think>\n\n'
        '<tool_call>\n'
        '<function=python>\n'
        '<parameter=code>\n'
        'print(2+2)\n'
        '</parameter>\n'
        '</function>\n'
        '</tool_call>'
    )


def _make_final_answer_text() -> str:
    return '</think>\n\nThe answer is 4.<|im_end|>'


@pytest.fixture
def tokenizer():
    return SimpleTokenizer()


@pytest.fixture
def cfg():
    return RolloutConfig(
        max_turns=8,
        max_tokens_per_turn=4096,
        max_total_tokens=100_000,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mask_pattern_prompt_gen_env(tokenizer, cfg):
    """Prompt tokens have mask=0, gen tokens have mask=1, env tokens have mask=0."""
    tool_call_text = _make_tool_call_text()
    tool_call_ids = tokenizer.encode(tool_call_text, add_special_tokens=False)
    final_text = _make_final_answer_text()
    final_ids = tokenizer.encode(final_text, add_special_tokens=False)

    backend = MockBackend([
        (tool_call_ids, "stop"),   # turn 0: model makes tool call
        (final_ids, "eos"),        # turn 1: model gives final answer
    ])

    env = MockEnv()
    rollout = MultiTurnRollout(env, QWEN3_5_SPEC, backend, tokenizer)
    traj = rollout.run(cfg, seed_env=0, seed_rollout=0, rollout_id=0)

    # Check that mask is correct per turn kind
    for start, end, kind in traj.turns:
        segment_mask = traj.mask[start:end]
        if kind == "gen":
            assert segment_mask.all(), f"gen turn ({start}:{end}) should be all mask=1"
        elif kind in ("prompt", "env"):
            assert not segment_mask.any(), f"{kind} turn ({start}:{end}) should be all mask=0"


def test_tool_block_boundaries(tokenizer, cfg):
    """Tool block open/close and tool_resp open/close are all mask=0 (prompt segments)."""
    tool_call_text = _make_tool_call_text()
    tool_call_ids = tokenizer.encode(tool_call_text, add_special_tokens=False)
    final_text = _make_final_answer_text()
    final_ids = tokenizer.encode(final_text, add_special_tokens=False)

    backend = MockBackend([
        (tool_call_ids, "stop"),
        (final_ids, "eos"),
    ])

    env = MockEnv()
    rollout = MultiTurnRollout(env, QWEN3_5_SPEC, backend, tokenizer)
    traj = rollout.run(cfg, seed_env=0, seed_rollout=0, rollout_id=0)

    # Decode each turn and check that tool boundary strings are in non-gen turns
    full_text = tokenizer.decode(traj.tokens.tolist())
    assert QWEN3_5_SPEC.tool_block_open in full_text
    assert QWEN3_5_SPEC.tool_resp_open.strip() in full_text
    assert QWEN3_5_SPEC.tool_resp_close.strip() in full_text
    assert QWEN3_5_SPEC.tool_block_close.strip() in full_text

    # All tool boundary text is in prompt or env turns (mask=0)
    for start, end, kind in traj.turns:
        if kind == "gen":
            continue
        seg = tokenizer.decode(traj.tokens[start:end].tolist())
        # These may contain boundary strings; mask should be 0
        assert not traj.mask[start:end].any()


def test_turn_types_present(tokenizer, cfg):
    """A multi-turn trajectory should have prompt, gen, and env turn kinds."""
    tool_call_text = _make_tool_call_text()
    tool_call_ids = tokenizer.encode(tool_call_text, add_special_tokens=False)
    final_text = _make_final_answer_text()
    final_ids = tokenizer.encode(final_text, add_special_tokens=False)

    backend = MockBackend([
        (tool_call_ids, "stop"),
        (final_ids, "eos"),
    ])

    env = MockEnv()
    rollout = MultiTurnRollout(env, QWEN3_5_SPEC, backend, tokenizer)
    traj = rollout.run(cfg, seed_env=0, seed_rollout=0, rollout_id=0)

    kinds = {kind for _, _, kind in traj.turns}
    assert "prompt" in kinds
    assert "gen" in kinds
    assert "env" in kinds


def test_turns_cover_full_sequence(tokenizer, cfg):
    """Turn ranges should cover every token exactly once."""
    tool_call_text = _make_tool_call_text()
    tool_call_ids = tokenizer.encode(tool_call_text, add_special_tokens=False)
    final_text = _make_final_answer_text()
    final_ids = tokenizer.encode(final_text, add_special_tokens=False)

    backend = MockBackend([
        (tool_call_ids, "stop"),
        (final_ids, "eos"),
    ])

    env = MockEnv()
    rollout = MultiTurnRollout(env, QWEN3_5_SPEC, backend, tokenizer)
    traj = rollout.run(cfg, seed_env=0, seed_rollout=0, rollout_id=0)

    # Turns should be contiguous and cover the full token range
    covered = set()
    for start, end, _ in traj.turns:
        for i in range(start, end):
            assert i not in covered, f"Token {i} covered by multiple turns"
            covered.add(i)
    assert covered == set(range(traj.seq_len))


def test_truncation_at_max_turns(tokenizer):
    """When the model keeps making tool calls until max_turns, truncation metadata is set."""
    cfg = RolloutConfig(
        max_turns=2,
        max_tokens_per_turn=4096,
        max_total_tokens=100_000,
    )

    tool_call_text = _make_tool_call_text()
    tool_call_ids = tokenizer.encode(tool_call_text, add_special_tokens=False)

    # Both turns produce tool calls -> hits max_turns
    backend = MockBackend([
        (tool_call_ids, "stop"),
        (tool_call_ids, "stop"),
    ])

    env = MockEnv()
    rollout = MultiTurnRollout(env, QWEN3_5_SPEC, backend, tokenizer)
    traj = rollout.run(cfg, seed_env=0, seed_rollout=0, rollout_id=0)

    assert "truncation" in traj.meta
    assert traj.meta["truncation"]["reason"] == "max_turns_no_answer"


def test_truncation_at_max_tokens_per_turn(tokenizer):
    """When a single turn hits max_tokens_per_turn, truncation metadata is set."""
    cfg = RolloutConfig(
        max_turns=8,
        max_tokens_per_turn=10,  # very small
        max_total_tokens=100_000,
    )

    # Return exactly max_new tokens with finish_reason "length"
    gen_ids = list(range(65, 75))  # 10 tokens
    backend = MockBackend([(gen_ids, "length")])

    env = MockEnv()
    rollout = MultiTurnRollout(env, QWEN3_5_SPEC, backend, tokenizer)
    traj = rollout.run(cfg, seed_env=0, seed_rollout=0, rollout_id=0)

    assert "truncation" in traj.meta
    assert traj.meta["truncation"]["reason"] == "max_tokens_per_turn"


def test_reward_attached_to_trajectory(tokenizer, cfg):
    """Frozen trajectory should have the env's reward."""
    final_text = _make_final_answer_text()
    final_ids = tokenizer.encode(final_text, add_special_tokens=False)

    backend = MockBackend([(final_ids, "eos")])

    env = MockEnv()
    env._reward_val = 1.0
    rollout = MultiTurnRollout(env, QWEN3_5_SPEC, backend, tokenizer)
    traj = rollout.run(cfg, seed_env=0, seed_rollout=0, rollout_id=0)

    assert traj.reward == 1.0


def test_no_tool_call_single_turn(tokenizer, cfg):
    """If the model answers directly (no tool call), rollout is a single gen turn."""
    final_text = _make_final_answer_text()
    final_ids = tokenizer.encode(final_text, add_special_tokens=False)

    backend = MockBackend([(final_ids, "eos")])
    env = MockEnv()
    rollout = MultiTurnRollout(env, QWEN3_5_SPEC, backend, tokenizer)
    traj = rollout.run(cfg, seed_env=0, seed_rollout=0, rollout_id=0)

    kinds = [kind for _, _, kind in traj.turns]
    assert "env" not in kinds
    gen_count = kinds.count("gen")
    assert gen_count == 1


def test_multiple_tool_calls_in_one_turn(tokenizer, cfg):
    """Two tool calls in a single generation share one tool_block_open/close."""
    # Two tool calls in one generation
    double_call_text = (
        '</think>\n\n'
        '<tool_call>\n'
        '<function=python>\n'
        '<parameter=code>\n'
        'print(1)\n'
        '</parameter>\n'
        '</function>\n'
        '</tool_call>\n'
        '<tool_call>\n'
        '<function=python>\n'
        '<parameter=code>\n'
        'print(2)\n'
        '</parameter>\n'
        '</function>\n'
        '</tool_call>'
    )
    double_call_ids = tokenizer.encode(double_call_text, add_special_tokens=False)
    final_text = _make_final_answer_text()
    final_ids = tokenizer.encode(final_text, add_special_tokens=False)

    env_responses = [
        (ToolResponse(content="1"), False),
        (ToolResponse(content="2"), False),
    ]
    env = MockEnv(responses=env_responses)

    backend = MockBackend([
        (double_call_ids, "stop"),
        (final_ids, "eos"),
    ])

    rollout = MultiTurnRollout(env, QWEN3_5_SPEC, backend, tokenizer)
    traj = rollout.run(cfg, seed_env=0, seed_rollout=0, rollout_id=0)

    # Should have 2 env turns (one per tool response)
    env_turns = [(s, e, k) for s, e, k in traj.turns if k == "env"]
    assert len(env_turns) == 2

    # All env turns should have mask=0
    for start, end, _ in env_turns:
        assert not traj.mask[start:end].any()


def test_two_turn_interaction(tokenizer, cfg):
    """Full 2-turn: tool call -> env response -> final answer."""
    tool_call_text = _make_tool_call_text()
    tool_call_ids = tokenizer.encode(tool_call_text, add_special_tokens=False)
    final_text = _make_final_answer_text()
    final_ids = tokenizer.encode(final_text, add_special_tokens=False)

    backend = MockBackend([
        (tool_call_ids, "stop"),
        (final_ids, "eos"),
    ])

    env = MockEnv()
    rollout = MultiTurnRollout(env, QWEN3_5_SPEC, backend, tokenizer)
    traj = rollout.run(cfg, seed_env=0, seed_rollout=0, rollout_id=0)

    # Should have exactly 2 gen turns
    gen_turns = [k for _, _, k in traj.turns if k == "gen"]
    assert len(gen_turns) == 2

    # Total trainable tokens = sum of gen turn lengths
    trainable = sum(end - start for start, end, kind in traj.turns if kind == "gen")
    assert traj.n_trainable == trainable
