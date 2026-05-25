from __future__ import annotations

import os
import time
import dataclasses
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable

import torch

from .loss.fused_logp import compute_per_token_logp, _chunked_logp
from .loss.grpo import compute_group_advantages, grpo_loss
from .rollout.async_pool import AsyncRolloutPool, RolloutPoolMetrics
from .rollout.base import MultiTurnRollout
from .rollout.hf_backend import HFBackend
from .trace import RolloutTraceRecorder
from .trajectory import Trajectory, require

if TYPE_CHECKING:
    from .config import RunConfig
    from .env.base import Env
    from .template.spec import TemplateSpec


@contextmanager
def _adapter_disabled(model):
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
        return

    import warnings
    warnings.warn(
        "model.disable_adapter() not available, zeroing LoRA scalings manually",
        RuntimeWarning, stacklevel=2,
    )
    scalings = {}
    for name, module in model.named_modules():
        if hasattr(module, "scaling"):
            scalings[name] = {k: v for k, v in module.scaling.items()}
            for k in module.scaling:
                module.scaling[k] = 0.0
    try:
        yield
    finally:
        for name, module in model.named_modules():
            if name in scalings:
                module.scaling.update(scalings[name])


def _forward_logp_with_grad(model, input_ids, device):
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    outputs = model(input_ids, output_hidden_states=True, use_cache=False)
    hidden = outputs.hidden_states[-1]
    weight = model.get_output_embeddings().weight

    h_shifted = hidden[:, :-1, :].contiguous()
    targets = input_ids[:, 1:].contiguous()

    logp_shifted = _chunked_logp(h_shifted, weight, targets)

    S = input_ids.shape[1]
    logp = torch.zeros(S, dtype=torch.float32, device=device)
    logp[1:] = logp_shifted
    return logp


class MultiTurnGRPOTrainer:

    def __init__(
        self,
        model,
        tokenizer,
        env: Env,
        template: TemplateSpec,
        config: RunConfig,
        backend=None,
        env_factory: Callable[[], Env] | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.env = env
        self.template = template
        self.config = config
        self.device = next(model.parameters()).device

        if backend is not None:
            self.backend = backend
        else:
            self.backend = HFBackend(model, tokenizer)
        self.rollout = MultiTurnRollout(
            env, template, self.backend, tokenizer, env_factory=env_factory,
        )

        # Eval uses the same backend as rollout (vLLM when available)
        self._eval_rollout = MultiTurnRollout(
            env, template, self.backend, tokenizer, env_factory=env_factory,
        )

        self._is_vllm = hasattr(self.backend, "generate_batch")
        self._async_pool: AsyncRolloutPool | None = None
        self.trace_recorder = RolloutTraceRecorder(
            config.rollout_trace_dir,
            tokenizer,
            template,
            run_id=f"seed{config.seed_torch}_{int(time.time())}",
            max_queue=config.rollout_trace_queue_size,
        )

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )

        if config.warmup_steps > 0:
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=config.warmup_steps,
            )
        else:
            self.scheduler = None

    def _record_rollout_trace(
        self,
        traj: Trajectory,
        *,
        iter_id: int,
        group_id: int,
        rollout_id: int,
        seed_env: int,
    ) -> None:
        cfg = self.config
        if cfg.rollout_trace_dir is None:
            return
        if cfg.rollout_trace_every <= 0:
            return
        if iter_id % cfg.rollout_trace_every != 0:
            return

        self.trace_recorder.record(
            traj,
            iter_id=iter_id,
            group_id=group_id,
            rollout_id=rollout_id,
            seed_env=seed_env,
            seed_rollout=cfg.seed_rollout + rollout_id * cfg.rollout.max_turns,
        )

    @torch.no_grad()
    def _eval(self, n_problems: int = 32) -> float:
        self.model.eval()
        cfg = dataclasses.replace(
            self.config.rollout,
            temperature=self.config.eval_temperature,
            top_p=self.config.eval_top_p,
            top_k=self.config.eval_top_k,
        )

        if self._is_vllm and cfg.max_turns == 1:
            seed_envs = [100000 + i for i in range(n_problems)]
            groups = self._eval_rollout.run_multi_group_batch(
                cfg=cfg,
                seed_envs=seed_envs,
                seed_rollout=99999,
                group_size=1,
            )
            correct = sum(g[0].reward for g in groups)
        else:
            correct = 0
            for i in range(n_problems):
                traj = self._eval_rollout.run(
                    cfg=cfg, seed_env=100000 + i,
                    seed_rollout=99999, rollout_id=0,
                )
                correct += traj.reward
        return correct / n_problems

    def train(self, num_iterations: int):
        cfg = self.config
        G = cfg.grpo.group_size
        M = cfg.grpo.prompts_per_iter
        adapter_path = os.path.join(cfg.output_dir, "adapter_latest")

        torch.manual_seed(cfg.seed_torch)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed_torch)

        degenerate_count = 0
        reward_history: list[float] = []
        adapter_saved = False
        policy_version = 0

        eval_acc = self._eval(n_problems=cfg.eval_subset_size)
        print(f"[eval  init] accuracy={eval_acc:.3f} ({cfg.eval_subset_size} problems)")

        if cfg.rollout.async_cross_step:
            if not self._is_vllm:
                raise ValueError("async_cross_step requires the vLLM backend")
            target_groups = cfg.rollout.async_pool_target_groups or max(M * 4, M + 1)
            max_inflight = cfg.rollout.async_max_inflight or (M * G)
            self._async_pool = AsyncRolloutPool(
                self.rollout,
                cfg.rollout,
                seed_env_base=cfg.seed_env,
                seed_rollout=cfg.seed_rollout,
                group_size=G,
                target_groups=target_groups,
                max_inflight_requests=max_inflight,
                max_off_policy_steps=cfg.rollout.async_max_off_policy_steps,
                drop_stale=cfg.rollout.async_drop_stale,
            )
            self._async_pool.start(policy_version=policy_version)

        for iter_id in range(num_iterations):
            t0 = time.time()

            # ── 1. Rollout (no grad) ──
            if self._is_vllm:
                if cfg.rollout.sleep_between_iters:
                    self.backend.wake()
                if (
                    iter_id > 0 and adapter_saved
                    and not cfg.rollout.async_cross_step
                ):
                    self.backend.sync_adapter(adapter_path, iter_id)

            self.model.eval()
            all_trajs: list[Trajectory] = []
            pool_metrics = RolloutPoolMetrics()

            seed_envs = [cfg.seed_env + iter_id * M + g for g in range(M)]

            if cfg.rollout.async_cross_step:
                assert self._async_pool is not None
                pool_groups, pool_metrics = self._async_pool.get_completed_groups(
                    M, current_policy_version=policy_version,
                )
                all_groups = [group.trajectories for group in pool_groups]
                seed_envs = [group.seed_env for group in pool_groups]
            elif self._is_vllm and cfg.rollout.async_refill:
                all_groups = self.rollout.run_multi_group_refill(
                    cfg=cfg.rollout,
                    seed_envs=seed_envs,
                    seed_rollout=cfg.seed_rollout,
                    group_size=G,
                )
            elif self._is_vllm and cfg.rollout.max_turns == 1 and cfg.rollout.batch_across_groups:
                all_groups = self.rollout.run_multi_group_batch(
                    cfg=cfg.rollout,
                    seed_envs=seed_envs,
                    seed_rollout=cfg.seed_rollout,
                    group_size=G,
                )
            else:
                all_groups = []
                for seed_env in seed_envs:
                    if self._is_vllm and cfg.rollout.max_turns == 1:
                        group_trajs = self.rollout.run_batch(
                            cfg=cfg.rollout,
                            seed_env=seed_env,
                            seed_rollout=cfg.seed_rollout,
                            group_size=G,
                        )
                    else:
                        group_trajs = []
                        for rollout_id in range(G):
                            traj = self.rollout.run(
                                cfg=cfg.rollout,
                                seed_env=seed_env,
                                seed_rollout=cfg.seed_rollout,
                                rollout_id=rollout_id,
                            )
                            group_trajs.append(traj)
                    all_groups.append(group_trajs)

            trace_enabled = (
                cfg.rollout_trace_dir is not None
                and cfg.rollout_trace_every > 0
                and iter_id % cfg.rollout_trace_every == 0
            )
            trace_budget = cfg.rollout_trace_max_per_iter
            trace_count = 0
            for group_id, group_trajs in enumerate(all_groups):
                # ── 2. Group normalization ──
                rewards = [t.reward for t in group_trajs]
                advantages = compute_group_advantages(
                    rewards, threshold=cfg.grpo.degenerate_std_threshold
                )
                trace_trajs = group_trajs
                if advantages is None:
                    degenerate_count += 1
                    if cfg.log_every and iter_id % cfg.log_every == 0:
                        print(
                            f"[iter {iter_id:4d}] degenerate group "
                            f"(all rewards={rewards[0]:.3f}), skipping"
                        )
                else:
                    trace_trajs = []
                    for traj, adv in zip(group_trajs, advantages):
                        traj = traj.replace(advantage=adv)
                        all_trajs.append(traj)
                        trace_trajs.append(traj)

                if trace_enabled and (trace_budget == 0 or trace_count < trace_budget):
                    for rollout_id, traj in enumerate(trace_trajs):
                        if trace_budget > 0 and trace_count >= trace_budget:
                            break
                        self._record_rollout_trace(
                            traj,
                            iter_id=iter_id,
                            group_id=group_id,
                            rollout_id=rollout_id,
                            seed_env=seed_envs[group_id],
                        )
                        trace_count += 1

            t_rollout = time.time() - t0

            # ── Sleep vLLM before training-side forwards ──
            if self._is_vllm and cfg.rollout.sleep_between_iters:
                self.backend.sleep()

            # ── 3-4: Training (only if non-degenerate trajectories exist) ──
            t_logp = 0.0
            t_train = 0.0
            enriched: list[Trajectory] = []
            all_metrics: list[dict] = []

            if all_trajs:
                # ── 3. Compute logp_old and logp_ref (no grad) ──
                t1 = time.time()

                for traj in all_trajs:
                    require(traj, "reward", "advantage")
                    ids = traj.tokens.unsqueeze(0).to(self.device)

                    if traj.logp_old is not None:
                        logp_old = traj.logp_old.to(self.device)
                    else:
                        logp_old = compute_per_token_logp(self.model, ids)

                    with _adapter_disabled(self.model):
                        logp_ref = compute_per_token_logp(self.model, ids)

                    enriched.append(
                        traj.replace(logp_old=logp_old, logp_ref=logp_ref)
                    )

                t_logp = time.time() - t1

                # ── 4. Training step (with grad) ──
                t2 = time.time()
                self.model.train()
                self.optimizer.zero_grad()

                total_loss = torch.tensor(0.0, device=self.device)

                for traj in enriched:
                    require(traj, "reward", "advantage", "logp_old", "logp_ref")
                    ids = traj.tokens.unsqueeze(0).to(self.device)
                    mask = traj.mask.to(self.device)

                    logp_new = _forward_logp_with_grad(self.model, ids, self.device)

                    loss, metrics = grpo_loss(
                        logp_new=logp_new,
                        logp_old=traj.logp_old.to(self.device),
                        logp_ref=traj.logp_ref.to(self.device),
                        mask=mask,
                        advantage=traj.advantage,
                        eps_low=cfg.grpo.eps_low,
                        eps_high=cfg.grpo.eps_high,
                        beta_kl=cfg.grpo.beta_kl,
                        ratio_clamp=cfg.grpo.ratio_clamp,
                    )
                    all_metrics.append(metrics)

                    scaled = loss / len(enriched)
                    scaled.backward()
                    total_loss = total_loss + scaled.detach()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                t_train = time.time() - t2

                # Save adapter for vLLM sync next iteration
                if self._is_vllm:
                    os.makedirs(adapter_path, exist_ok=True)
                    self.model.save_pretrained(adapter_path)
                    self.tokenizer.save_pretrained(adapter_path)
                    adapter_saved = True
                    if cfg.rollout.async_cross_step:
                        policy_version += 1
                        self.backend.sync_adapter(adapter_path, policy_version)
                        assert self._async_pool is not None
                        self._async_pool.update_policy_version(policy_version)

            elapsed = time.time() - t0

            # ── 5. Log (only if trained this iteration) ──
            if enriched and cfg.log_every and iter_id % cfg.log_every == 0:
                rewards_iter = [t.reward for t in enriched]
                iter_reward = sum(rewards_iter) / len(rewards_iter)
                reward_history.append(iter_reward)
                window = reward_history[-10:]
                avg_reward = sum(window) / len(window)

                avg_metrics = {}
                for key in all_metrics[0]:
                    avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

                print(
                    f"[iter {iter_id:4d}] "
                    f"loss={total_loss.item():.6f}  "
                    f"reward={iter_reward:.3f} "
                    f"avg10={avg_reward:.3f}  "
                    f"ratio={avg_metrics['mean_ratio']:.6f}  "
                    f"clip={avg_metrics['clip_fraction']:.3f}  "
                    f"clamp={avg_metrics['numerical_clamp_count']}  "
                    f"n={len(enriched)}  "
                    f"t={elapsed:.1f}s "
                    f"(roll={t_rollout:.1f} logp={t_logp:.1f} train={t_train:.1f})"
                    + (
                        " "
                        f"pool_wait={pool_metrics.wait_s:.1f}s "
                        f"stale={pool_metrics.mean_staleness:.2f}/"
                        f"{pool_metrics.max_staleness} "
                        f"drop={pool_metrics.dropped_stale_groups} "
                        f"drop_active={pool_metrics.dropped_active_stale_groups} "
                        f"pool_done={pool_metrics.completed_groups} "
                        f"pool_active={pool_metrics.active_groups} "
                        f"inflight={pool_metrics.inflight_requests}"
                        if cfg.rollout.async_cross_step else ""
                    )
                )
            elif cfg.log_every and iter_id % cfg.log_every == 0:
                rewards_iter = [
                    t.reward for group_trajs in all_groups for t in group_trajs
                    if t.reward is not None
                ]
                iter_reward = (
                    sum(rewards_iter) / len(rewards_iter)
                    if rewards_iter else float("nan")
                )
                print(
                    f"[iter {iter_id:4d}] "
                    f"no non-degenerate groups  "
                    f"reward={iter_reward:.3f} "
                    f"n=0/{sum(len(group_trajs) for group_trajs in all_groups)}  "
                    f"t={elapsed:.1f}s "
                    f"(roll={t_rollout:.1f} logp={t_logp:.1f} train={t_train:.1f})"
                )

            # ── 6. Periodic eval (runs even on degenerate iterations) ──
            if cfg.eval_every and (iter_id + 1) % cfg.eval_every == 0:
                eval_acc = self._eval(n_problems=cfg.eval_subset_size)
                print(f"[eval {iter_id+1:4d}] accuracy={eval_acc:.3f}")

            # ── 7. Periodic save ──
            if cfg.save_every and (iter_id + 1) % cfg.save_every == 0:
                save_path = f"{cfg.output_dir}/checkpoint-{iter_id+1}"
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
                print(f"  -> saved to {save_path}")

        self.trace_recorder.close()
        if self._async_pool is not None:
            self._async_pool.close()
        if hasattr(self.backend, "close"):
            self.backend.close()
        if self.trace_recorder.dropped:
            print(f"Rollout trace recorder dropped {self.trace_recorder.dropped} item(s).")
        if self.trace_recorder.errors:
            print(f"Rollout trace recorder failed to write {self.trace_recorder.errors} item(s).")
        print(f"\nDone. {degenerate_count} degenerate groups skipped.")
