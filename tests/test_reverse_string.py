from qwen3_rl.env.reverse_string import ReverseStringEnv


class _Trajectory:
    def __init__(self, text, n_trainable=0):
        self.text = text
        self.n_trainable = n_trainable

    def decode_last_gen_turn(self, tokenizer):
        return self.text


def test_reverse_string_easy_metadata_and_prompt():
    env = ReverseStringEnv(None, min_len=4, max_len=4)

    messages = env.reset(seed=1)

    assert len(env.last_metadata["original"]) == 4
    assert env.last_metadata["target"] == env.last_metadata["original"][::-1]
    assert env.last_metadata["difficulty"] == "easy"
    assert env.last_metadata["original"] in messages[0]["content"]


def test_reverse_string_hard_metadata_when_probability_one():
    env = ReverseStringEnv(
        None,
        min_len=4,
        max_len=4,
        hard_min_len=32,
        hard_max_len=32,
        hard_prob=1.0,
    )

    env.reset(seed=1)

    assert len(env.last_metadata["original"]) == 32
    assert env.last_metadata["difficulty"] == "hard"


def test_reverse_string_rejects_invalid_hard_config():
    try:
        ReverseStringEnv(None, hard_min_len=32, hard_prob=0.5)
    except ValueError as exc:
        assert "hard_min_len" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_reverse_reward_bonuses_thinking_when_correct():
    env = ReverseStringEnv(
        None,
        min_len=4,
        max_len=4,
        correct_think_bonus=0.5,
        think_bonus_cap_tokens=10,
        wrong_length_penalty=0.5,
    )
    env.reset(seed=1)
    target = env.last_metadata["target"]

    reward = env.reward(_Trajectory(f"<think>{'x' * 5}</think>{target}", n_trainable=20))

    assert reward == 1.25
    assert env.last_reward_metadata["correct"] is True
    assert env.last_reward_metadata["think_tokens"] == 5


def test_reverse_reward_penalizes_length_when_wrong():
    env = ReverseStringEnv(
        None,
        min_len=4,
        max_len=4,
        correct_think_bonus=0.5,
        wrong_length_penalty=0.5,
        wrong_length_cap_tokens=100,
    )
    env.reset(seed=1)

    reward = env.reward(_Trajectory("<think>long</think>wrong", n_trainable=20))

    assert reward == -0.1
    assert env.last_reward_metadata["correct"] is False
    assert env.last_reward_metadata["gen_tokens"] == 20
