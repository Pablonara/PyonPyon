from __future__ import annotations

import json

from qwen3_rl.template.qwen3_5 import QWEN3_5_SPEC
from qwen3_rl.trace import RolloutTraceRecorder, trajectory_to_trace_record
from qwen3_rl.trajectory import TrajectoryBuilder


class SimpleTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(chr(int(tid)) for tid in ids)


def _make_trajectory(tokenizer: SimpleTokenizer):
    builder = TrajectoryBuilder(tokenizer)
    builder.add_prompt("<|im_start|>user\nCompute 2+2<|im_end|>\n")
    builder.add_prompt(QWEN3_5_SPEC.assistant_open)
    builder.add_generated(
        tokenizer.encode(
            "</think>\n\n"
            "<tool_call>\n"
            "<function=python>\n"
            "<parameter=code>\n"
            "print(2+2)\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
            "<|im_end|>",
            add_special_tokens=False,
        )
    )
    builder.add_prompt("\n")
    builder.add_prompt(QWEN3_5_SPEC.tool_block_open)
    builder.add_env("\n<tool_response>\n4\n</tool_response>")
    builder.add_prompt(QWEN3_5_SPEC.tool_block_close)
    return builder.freeze().replace(reward=1.0, advantage=0.5)


def test_trajectory_to_trace_record_includes_transcript_and_mask_metadata():
    tokenizer = SimpleTokenizer()
    trajectory = _make_trajectory(tokenizer)

    record = trajectory_to_trace_record(
        trajectory,
        tokenizer,
        QWEN3_5_SPEC,
        run_id="run-a",
        iter_id=3,
        group_id=1,
        rollout_id=2,
        seed_env=11,
        seed_rollout=22,
    )

    assert record["version"] == 1
    assert record["run_id"] == "run-a"
    assert record["reward"] == 1.0
    assert record["advantage"] == 0.5
    assert record["seq_len"] == trajectory.seq_len
    assert record["n_trainable"] == trajectory.n_trainable
    assert record["mask_summary"]["trainable"] == trajectory.n_trainable
    assert record["mask_summary"]["gen"] == trajectory.n_trainable
    assert "<tool_call>" in record["raw_text"]

    gen_turns = [turn for turn in record["turns"] if turn["kind"] == "gen"]
    assert len(gen_turns) == 1
    assert gen_turns[0]["mask_all"] is True
    assert gen_turns[0]["parsed"]["tool_calls"][0]["name"] == "python"
    assert gen_turns[0]["parsed"]["tool_calls"][0]["arguments"]["code"] == "print(2+2)"

    env_turns = [turn for turn in record["turns"] if turn["kind"] == "env"]
    assert env_turns[0]["mask_any"] is False
    assert env_turns[0]["parsed"]["tool_responses"] == ["4"]


def test_rollout_trace_recorder_writes_jsonl_without_blocking_training_thread(tmp_path):
    tokenizer = SimpleTokenizer()
    trajectory = _make_trajectory(tokenizer)
    recorder = RolloutTraceRecorder(
        str(tmp_path),
        tokenizer,
        QWEN3_5_SPEC,
        run_id="test-run",
        max_queue=4,
    )

    recorder.record(
        trajectory,
        iter_id=0,
        group_id=0,
        rollout_id=0,
        seed_env=42,
        seed_rollout=42,
    )
    recorder.close()

    trace_path = tmp_path / "test-run.jsonl"
    assert trace_path.exists()
    record = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert record["iter_id"] == 0
    assert record["turns"][0]["kind"] == "prompt"
    assert recorder.dropped == 0
    assert recorder.errors == 0
