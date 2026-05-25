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
import ctypes
import site

import torch


def _preload_real_cudart() -> None:
    """Bind FlashInfer/vLLM CUDART helpers to the real libcudart before Unsloth.

    Unsloth's import path can load tilelang's ``libcudart_stub.so``.  Some
    FlashInfer/vLLM helpers scan already-loaded libraries and then accidentally
    bind to that stub.  Pre-importing FlashInfer's CUDA IPC helper after loading
    the real libcudart keeps it cached on the real runtime.
    """
    roots = site.getsitepackages() + [site.getusersitepackages()]
    candidates = []
    for root in roots:
        candidates.extend([
            os.path.join(root, "nvidia", "cu13", "lib", "libcudart.so.13"),
            os.path.join(root, "nvidia", "cu13", "lib", "libcudart.so"),
            os.path.join(root, "nvidia", "cuda_runtime", "lib", "libcudart.so.13"),
            os.path.join(root, "nvidia", "cuda_runtime", "lib", "libcudart.so.12"),
        ])
    real_cudart = next((p for p in candidates if os.path.isfile(p)), None)
    if real_cudart is None:
        return
    os.environ.setdefault("VLLM_CUDART_SO_PATH", real_cudart)
    ctypes.CDLL(real_cudart, mode=ctypes.RTLD_GLOBAL)
    try:
        import flashinfer.comm.cuda_ipc as cuda_ipc
        cuda_ipc.cudart = cuda_ipc.CudaRTLibrary(real_cudart)
    except Exception:
        pass


_preload_real_cudart()

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
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=None,
                        help="Training top-k sampling; omit to disable top-k filtering")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--output-dir", default="./rl_output")
    parser.add_argument(
        "--env",
        choices=["string_match", "mock_repl", "reverse_string"],
        default="string_match",
    )
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35,
                        help="vLLM GPU memory utilization")
    parser.add_argument("--max-lora-rank", type=int, default=None,
                        help="vLLM max LoRA rank; defaults to --lora-r")
    parser.add_argument("--async-refill", action="store_true",
                        help="Use async vLLM refill rollout scheduling")
    parser.add_argument("--async-cross-step", action="store_true",
                        help="Keep a persistent cross-step async rollout pool")
    parser.add_argument("--async-max-inflight", type=int, default=None,
                        help="Max concurrent async rollout requests")
    parser.add_argument("--async-pool-target-groups", type=int, default=None,
                        help="Target completed+active groups kept in the cross-step async pool")
    parser.add_argument("--async-max-off-policy-steps", type=int, default=16,
                        help="Drop cross-step async groups older than this many policy updates")
    parser.add_argument("--max-loaded-loras", type=int, default=None,
                        help="Maximum vLLM LoRA versions kept loaded; defaults to off-policy window + 2")
    parser.add_argument("--no-async-drop-stale", action="store_true",
                        help="Do not drop stale cross-step async groups")
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-subset", type=int, default=32)
    parser.add_argument("--eval-temperature", type=float, default=0.6)
    parser.add_argument("--eval-top-p", type=float, default=0.95)
    parser.add_argument("--eval-top-k", type=int, default=20,
                        help="Eval top-k sampling; omit only via programmatic config")
    parser.add_argument("--prompts-per-iter", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable torch.compile + CUDA graphs in vLLM (for debugging)")
    args = parser.parse_args()
    num_generations = (
        args.num_generations if args.num_generations is not None else args.group_size
    )

    from unsloth import FastModel

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
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_lora_rank=args.max_lora_rank or args.lora_r,
            max_turns=args.max_turns,
            max_tokens_per_turn=args.max_tokens_per_turn,
            max_total_tokens=args.max_seq_len,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            enforce_eager=args.enforce_eager,
            async_refill=args.async_refill or args.async_cross_step,
            async_max_inflight=args.async_max_inflight,
            async_cross_step=args.async_cross_step,
            async_pool_target_groups=args.async_pool_target_groups,
            async_max_off_policy_steps=args.async_max_off_policy_steps,
            max_loaded_loras=args.max_loaded_loras,
            async_drop_stale=not args.no_async_drop_stale,
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
        eval_temperature=args.eval_temperature,
        eval_top_p=args.eval_top_p,
        eval_top_k=args.eval_top_k,
        rollout_trace_dir=args.rollout_trace_dir,
        rollout_trace_every=args.rollout_trace_every,
        rollout_trace_max_per_iter=args.rollout_trace_max_per_iter,
        rollout_trace_queue_size=args.rollout_trace_queue_size,
        seed_torch=args.seed,
        seed_rollout=args.seed,
        seed_env=args.seed,
    )

    # Create environment
    def make_env():
        if args.env == "mock_repl":
            from qwen3_rl.env.mock_repl import MockREPLEnv
            return MockREPLEnv(tokenizer)
        if args.env == "reverse_string":
            from qwen3_rl.env.reverse_string import ReverseStringEnv
            return ReverseStringEnv(
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
        from qwen3_rl.env.string_match import StringMatchEnv
        return StringMatchEnv(tokenizer)

    env = make_env()

    # Create backend (None = default HF backend created by trainer)
    backend = None
    if args.backend == "vllm":
        from qwen3_rl.rollout.vllm_backend import VLLMBackend
        backend = VLLMBackend(args.model, config.rollout)

    trainer = MultiTurnGRPOTrainer(
        model, tokenizer, env, QWEN3_5_SPEC, config,
        backend=backend, env_factory=make_env,
    )

    print(
        f"\nStarting GRPO training: {args.num_iters} iterations, "
        f"M={args.prompts_per_iter}, num_generations={num_generations}"
    )
    print(f"  model={args.model}, lora_r={args.lora_r}, lr={args.lr}, seed={args.seed}")
    print(f"  max_turns={args.max_turns}, max_tokens_per_turn={args.max_tokens_per_turn}")
    print(
        f"  train_sampling=temp={args.temperature} top_p={args.top_p} "
        f"top_k={args.top_k if args.top_k is not None else 'disabled'}"
    )
    print(
        f"  eval_sampling=temp={args.eval_temperature} top_p={args.eval_top_p} "
        f"top_k={args.eval_top_k if args.eval_top_k is not None else 'disabled'}"
    )
    print(f"  env={args.env}, backend={args.backend}")
    if args.backend == "vllm":
        print(
            f"  vllm_gpu_memory_utilization={args.gpu_memory_utilization} "
            f"max_lora_rank={args.max_lora_rank or args.lora_r} "
            f"async_refill={args.async_refill or args.async_cross_step} "
            f"async_cross_step={args.async_cross_step} "
            f"async_max_inflight={args.async_max_inflight} "
            f"async_pool_target_groups={args.async_pool_target_groups} "
            f"max_off_policy={args.async_max_off_policy_steps} "
            f"max_loaded_loras={args.max_loaded_loras or 'auto'} "
            f"drop_stale={not args.no_async_drop_stale}"
        )
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
