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
    args = parser.parse_args()

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
            group_size=args.group_size,
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
    )

    # Create environment
    if args.env == "reverse_string":
        from qwen3_rl.env.reverse_string import ReverseStringEnv
        env = ReverseStringEnv(tokenizer)
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

    print(f"\nStarting GRPO training: {args.num_iters} iterations, M={args.prompts_per_iter}, G={args.group_size}")
    print(f"  model={args.model}, lora_r={args.lora_r}, lr={args.lr}, seed={args.seed}")
    print(f"  max_turns={args.max_turns}, max_tokens_per_turn={args.max_tokens_per_turn}")
    print(f"  env={args.env}, backend={args.backend}")
    print()

    trainer.train(num_iterations=args.num_iters)


if __name__ == "__main__":
    main()
