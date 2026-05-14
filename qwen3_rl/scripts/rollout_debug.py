"""Phase 0 smoke test: verify all qwen3_rl modules import cleanly."""

def main():
    from qwen3_rl import __version__
    from qwen3_rl.config import RunConfig, RolloutConfig, GRPOConfig
    from qwen3_rl.env.types import ToolCall, ToolResponse, Message
    from qwen3_rl.env.base import Env
    from qwen3_rl.env.string_match import StringMatchEnv
    from qwen3_rl.template.spec import TemplateSpec
    from qwen3_rl.template.qwen3_5 import QWEN3_5_SPEC, parse_tool_calls
    from qwen3_rl.trajectory import TrajectoryBuilder, Trajectory, require
    from qwen3_rl.rollout.base import MultiTurnRollout
    from qwen3_rl.rollout.hf_backend import HFBackend
    from qwen3_rl.loss.fused_logp import compute_per_token_logp
    from qwen3_rl.loss.grpo import grpo_loss, compute_group_advantages
    from qwen3_rl.trainer import MultiTurnGRPOTrainer

    print(f"qwen3_rl v{__version__} — all imports OK")

    # quick parse_tool_calls sanity
    result = parse_tool_calls("no tool calls here")
    assert result is None, f"expected None, got {result}"

    result = parse_tool_calls(
        '<tool_call>\n<function=read_file>\n'
        '<parameter=path>\n/tmp/test.py\n</parameter>\n'
        '</function>\n</tool_call>'
    )
    assert result is not None and len(result) == 1
    assert result[0].name == "read_file"
    assert result[0].arguments["path"] == "/tmp/test.py"
    print("parse_tool_calls: OK")

    # group advantages
    adv = compute_group_advantages([1.0, 0.0, 1.0, 0.0])
    assert adv is not None and len(adv) == 4
    assert compute_group_advantages([1.0, 1.0, 1.0]) is None
    print("compute_group_advantages: OK")

    print("\nAll Phase 0 checks passed.")


if __name__ == "__main__":
    main()
