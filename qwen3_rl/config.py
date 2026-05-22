from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RolloutConfig:
    backend: Literal["vllm", "hf"] = "hf"
    quantization: Literal["bf16", "fp8", "nf4"] = "bf16"
    gpu_memory_utilization: float = 0.35
    enable_lora: bool = True
    max_lora_rank: int = 64
    sleep_between_iters: bool = False
    sleep_level: Literal[1, 2] = 1
    enable_sleep_mode: bool = True
    language_model_only: bool = True
    engine_version: Literal["v1"] = "v1"
    enable_prefix_caching: bool = True
    mamba_cache_mode: Literal["align", "all"] = "align"
    enforce_eager: bool = False
    batch_across_groups: bool = True
    max_turns: int = 8
    max_tokens_per_turn: int = 4096
    max_total_tokens: int = 48_000
    temperature: float = 1.0
    top_p: float = 1.0


@dataclass
class GRPOConfig:
    group_size: int = 8
    prompts_per_iter: int = 1
    eps_low: float = 0.2
    eps_high: float = 0.28
    beta_kl: float = 0.0
    is_correction: Literal["off", "ppo_clip", "tis"] = "ppo_clip"
    tis_delta: float = 0.5
    epochs_per_batch: int = 1
    ratio_clamp: float = 20.0
    degenerate_std_threshold: float = 1e-6

    def __post_init__(self) -> None:
        if self.group_size < 2:
            raise ValueError("GRPO group_size must be >= 2")


@dataclass
class RunConfig:
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    template: str = "qwen3_5"
    env: str = "string_match"
    model_name: str = "Qwen/Qwen3.5-9B"
    output_dir: str = "./rl_output"
    lora_r: int = 64
    learning_rate: float = 5e-6
    lr_scheduler: str = "constant"
    warmup_steps: int = 20
    log_every: int = 1
    save_every: int = 50
    eval_every: int = 25
    eval_subset_size: int = 32
    resume_from_checkpoint: str | None = None
    seed_torch: int = 42
    seed_rollout: int = 42
    seed_env: int = 42
    rollout_trace_dir: str | None = None
    rollout_trace_every: int = 1
    rollout_trace_max_per_iter: int = 8
    rollout_trace_queue_size: int = 1024
