"""Unit tests for TrajectoryBuilder and Trajectory."""

import torch
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.encode = lambda text, add_special_tokens=True: list(range(len(text.split())))
    return tok


def test_append_only(mock_tokenizer):
    from qwen3_rl.trajectory import TrajectoryBuilder
    b = TrajectoryBuilder(mock_tokenizer)
    assert len(b.tokens) == 0
    b.add_prompt("hello world")
    n1 = len(b.tokens)
    assert n1 > 0
    b.add_generated([100, 101, 102])
    n2 = len(b.tokens)
    assert n2 > n1
    b.add_env("tool output")
    n3 = len(b.tokens)
    assert n3 > n2


def test_mask_correctness(mock_tokenizer):
    from qwen3_rl.trajectory import TrajectoryBuilder
    b = TrajectoryBuilder(mock_tokenizer)
    b.add_prompt("hello world")
    prompt_len = len(b.tokens)
    b.add_generated([100, 101, 102])
    b.add_env("tool output")
    env_start = prompt_len + 3

    assert all(m == 0 for m in b.mask[:prompt_len])
    assert all(m == 1 for m in b.mask[prompt_len:prompt_len + 3])
    assert all(m == 0 for m in b.mask[env_start:])


def test_freeze_creates_trajectory(mock_tokenizer):
    from qwen3_rl.trajectory import TrajectoryBuilder
    b = TrajectoryBuilder(mock_tokenizer)
    b.add_prompt("hello")
    b.add_generated([1, 2])

    t = b.freeze()
    assert isinstance(t.tokens, torch.Tensor)
    assert isinstance(t.mask, torch.Tensor)
    assert t.tokens.dtype == torch.long
    assert t.mask.dtype == torch.bool
    assert len(t.turns) == 2
    assert t.reward is None
    assert t.advantage is None


def test_freeze_idempotent(mock_tokenizer):
    from qwen3_rl.trajectory import TrajectoryBuilder
    b = TrajectoryBuilder(mock_tokenizer)
    b.add_prompt("test")
    b.add_generated([5, 6, 7])

    t1 = b.freeze()
    t2 = b.freeze()
    assert torch.equal(t1.tokens, t2.tokens)
    assert torch.equal(t1.mask, t2.mask)


def test_require_raises():
    from qwen3_rl.trajectory import Trajectory, require
    t = Trajectory(
        tokens=torch.tensor([1, 2, 3]),
        mask=torch.tensor([True, True, False]),
        turns=[(0, 3, "gen")],
    )
    require(t, "tokens", "mask")

    with pytest.raises(RuntimeError, match="missing required fields"):
        require(t, "reward")

    with pytest.raises(RuntimeError, match="missing required fields"):
        require(t, "logp_old", "logp_ref")


def test_replace():
    from qwen3_rl.trajectory import Trajectory
    t = Trajectory(
        tokens=torch.tensor([1, 2, 3]),
        mask=torch.tensor([True, True, False]),
        turns=[(0, 3, "gen")],
    )
    t2 = t.replace(reward=1.0, advantage=0.5)
    assert t2.reward == 1.0
    assert t2.advantage == 0.5
    assert t.reward is None


def test_decode_last_gen_turn():
    from qwen3_rl.trajectory import Trajectory
    tok = MagicMock()
    tok.decode = lambda ids, skip_special_tokens=False: f"decoded_{len(ids)}"

    t = Trajectory(
        tokens=torch.tensor([1, 2, 3, 4, 5]),
        mask=torch.tensor([False, False, True, True, False]),
        turns=[(0, 2, "prompt"), (2, 4, "gen"), (4, 5, "env")],
    )
    result = t.decode_last_gen_turn(tok)
    assert "decoded_2" == result


def test_seq_len_and_n_trainable():
    from qwen3_rl.trajectory import Trajectory
    t = Trajectory(
        tokens=torch.tensor([1, 2, 3, 4, 5]),
        mask=torch.tensor([False, False, True, True, False]),
        turns=[],
    )
    assert t.seq_len == 5
    assert t.n_trainable == 2
