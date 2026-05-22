"""Entry point for multi-turn GRPO training."""

import os
os.environ["UNSLOTH_CE_LOSS_TARGET_GB"] = "4"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# FLA auto-selects TileLang backend on Blackwell, which has bugs (ndim
# mismatch, missing templates).  Force Triton backend — it works fine.
os.environ["FLA_TILELANG"] = "0"

CACHE_DIR = os.path.join(os.getcwd(), ".hf_cache")
os.environ["HF_HOME"] = CACHE_DIR

import argparse

import torch
from unsloth import FastModel

from qwen3_rl.config import RunConfig, RolloutConfig, GRPOConfig
from qwen3_rl.template.qwen3_5 import QWEN3_5_SPEC
from qwen3_rl.trainer import MultiTurnGRPOTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=None,
                        help="Alias for --group-size; number of rollouts per prompt")
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument("--max-turns", type=int, default=1)
    parser.add_argument("--max-tokens-per-turn", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--output-dir", default="./rl_output")
    parser.add_argument("--env", choices=["string_match", "reverse_string"], default="string_match")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-subset", type=int, default=32)
    parser.add_argument("--prompts-per-iter", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable torch.compile + CUDA graphs in vLLM (for debugging)")
    parser.add_argument("--reverse-min-len", type=int, default=4,
                        help="Minimum easy reverse-string length")
    parser.add_argument("--reverse-max-len", type=int, default=8,
                        help="Maximum easy reverse-string length")
    parser.add_argument("--reverse-hard-min-len", type=int, default=None,
                        help="Minimum hard reverse-string length")
    parser.add_argument("--reverse-hard-max-len", type=int, default=None,
                        help="Maximum hard reverse-string length")
    parser.add_argument("--reverse-hard-prob", type=float, default=0.0,
                        help="Probability of sampling a hard reverse-string problem")
    parser.add_argument("--reverse-correct-think-bonus", type=float, default=0.0,
                        help="Max reward bonus for longer thinking on correct reverse answers")
    parser.add_argument("--reverse-think-bonus-cap-tokens", type=int, default=1024,
                        help="Think-token cap used to normalize the correct-answer bonus")
    parser.add_argument("--reverse-wrong-length-penalty", type=float, default=0.0,
                        help="Max reward penalty for long wrong reverse answers")
    parser.add_argument("--reverse-wrong-length-cap-tokens", type=int, default=4096,
                        help="Generated-token cap used to normalize the wrong-answer penalty")
    parser.add_argument("--rollout-trace-dir", default=None,
                        help="Optional directory for non-blocking rollout trace JSONL logs")
    parser.add_argument("--rollout-trace-every", type=int, default=1,
                        help="Trace every N training iterations when rollout tracing is enabled")
    parser.add_argument("--rollout-trace-max-per-iter", type=int, default=8,
                        help="Max rollout traces per traced iteration; 0 means unlimited")
    parser.add_argument("--rollout-trace-queue-size", type=int, default=1024,
                        help="Best-effort trace writer queue size; full queues drop traces")
    args = parser.parse_args()

    num_generations = (
        args.num_generations if args.num_generations is not None else args.group_size
    )

    print(f"Loading {args.model}...")
    model, tokenizer = FastModel.from_pretrained(
        args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )
    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_r,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
    )

    # Qwen3.5 from FastModel returns a Qwen3VLProcessor, not a tokenizer.
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer

    template_path = os.path.join(os.path.dirname(__file__), "..", "..", "qwen35_chat_template.jinja")
    if os.path.exists(template_path):
        tokenizer.chat_template = open(template_path).read()
        print(f"Loaded chat template from {template_path}")
    else:
        print("Warning: qwen35_chat_template.jinja not found, using model default")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = RunConfig(
        rollout=RolloutConfig(
            backend=args.backend,
            max_turns=args.max_turns,
            max_tokens_per_turn=args.max_tokens_per_turn,
            max_total_tokens=args.max_seq_len,
            enforce_eager=args.enforce_eager,
        ),
        grpo=GRPOConfig(
            group_size=num_generations,
            prompts_per_iter=args.prompts_per_iter,
        ),
        model_name=args.model,
        lora_r=args.lora_r,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        eval_every=args.eval_every,
        eval_subset_size=args.eval_subset,
        seed_torch=args.seed,
        seed_rollout=args.seed,
        seed_env=args.seed,
        rollout_trace_dir=args.rollout_trace_dir,
        rollout_trace_every=args.rollout_trace_every,
        rollout_trace_max_per_iter=args.rollout_trace_max_per_iter,
        rollout_trace_queue_size=args.rollout_trace_queue_size,
    )

    # Create environment
    if args.env == "reverse_string":
        from qwen3_rl.env.reverse_string import ReverseStringEnv
        env = ReverseStringEnv(
            tokenizer,
            min_len=args.reverse_min_len,
            max_len=args.reverse_max_len,
            hard_min_len=args.reverse_hard_min_len,
            hard_max_len=args.reverse_hard_max_len,
            hard_prob=args.reverse_hard_prob,
            correct_think_bonus=args.reverse_correct_think_bonus,
            think_bonus_cap_tokens=args.reverse_think_bonus_cap_tokens,
            wrong_length_penalty=args.reverse_wrong_length_penalty,
            wrong_length_cap_tokens=args.reverse_wrong_length_cap_tokens,
        )
    else:
        from qwen3_rl.env.string_match import StringMatchEnv
        env = StringMatchEnv(tokenizer)

    # Create backend (None = default HF backend created by trainer)
    backend = None
    if args.backend == "vllm":
        from qwen3_rl.rollout.vllm_backend import VLLMBackend
        backend = VLLMBackend(args.model, config.rollout)

    trainer = MultiTurnGRPOTrainer(
        model, tokenizer, env, QWEN3_5_SPEC, config, backend=backend,
    )

    print(
        f"\nStarting GRPO training: {args.num_iters} iterations, "
        f"M={args.prompts_per_iter}, num_generations={num_generations}"
    )
    print(f"  model={args.model}, lora_r={args.lora_r}, lr={args.lr}, seed={args.seed}")
    print(f"  max_turns={args.max_turns}, max_tokens_per_turn={args.max_tokens_per_turn}")
    print(f"  env={args.env}, backend={args.backend}")
    if args.env == "reverse_string":
        print(
            "  reverse_lengths="
            f"easy[{args.reverse_min_len},{args.reverse_max_len}] "
            f"hard[{args.reverse_hard_min_len},{args.reverse_hard_max_len}] "
            f"p_hard={args.reverse_hard_prob}"
        )
        print(
            "  reverse_reward_shaping="
            f"correct_think_bonus={args.reverse_correct_think_bonus} "
            f"think_cap={args.reverse_think_bonus_cap_tokens} "
            f"wrong_length_penalty={args.reverse_wrong_length_penalty} "
            f"wrong_cap={args.reverse_wrong_length_cap_tokens}"
        )
    if args.rollout_trace_dir:
        print(
            f"  rollout_traces={args.rollout_trace_dir} "
            f"(every={args.rollout_trace_every}, max_per_iter={args.rollout_trace_max_per_iter})"
        )
    print()

    trainer.train(num_iterations=args.num_iters)


if __name__ == "__main__":
    main()
