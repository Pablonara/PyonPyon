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
    async_refill: bool = False
    async_max_inflight: int | None = None
    async_rollout_groups: int | None = None
    async_cross_step: bool = False
    async_pool_target_groups: int | None = None
    async_max_off_policy_steps: int = 16
    async_drop_stale: bool = True
    max_loaded_loras: int | None = None
    max_turns: int = 8
    max_tokens_per_turn: int = 4096
    max_total_tokens: int = 48_000
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int | None = None

    def __post_init__(self) -> None:
        if self.async_cross_step:
            self.async_refill = True
        if self.async_max_inflight is not None and self.async_max_inflight < 1:
            raise ValueError("async_max_inflight must be >= 1")
        if self.async_rollout_groups is not None and self.async_rollout_groups < 1:
            raise ValueError("async_rollout_groups must be >= 1")
        if self.async_pool_target_groups is not None and self.async_pool_target_groups < 1:
            raise ValueError("async_pool_target_groups must be >= 1")
        if self.async_max_off_policy_steps < 0:
            raise ValueError("async_max_off_policy_steps must be >= 0")
        if self.max_loaded_loras is not None and self.max_loaded_loras < 1:
            raise ValueError("max_loaded_loras must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be >= 1 when set")


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
    eval_temperature: float = 0.6
    eval_top_p: float = 0.95
    eval_top_k: int | None = 20
    rollout_trace_dir: str | None = None
    rollout_trace_every: int = 1
    rollout_trace_max_per_iter: int = 8
    rollout_trace_queue_size: int = 1024
    resume_from_checkpoint: str | None = None
    seed_torch: int = 42
    seed_rollout: int = 42
    seed_env: int = 42

    def __post_init__(self) -> None:
        if self.eval_temperature < 0:
            raise ValueError("eval_temperature must be >= 0")
        if not 0 < self.eval_top_p <= 1:
            raise ValueError("eval_top_p must be in (0, 1]")
        if self.eval_top_k is not None and self.eval_top_k < 1:
            raise ValueError("eval_top_k must be >= 1 when set")
