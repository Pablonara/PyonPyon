from __future__ import annotations

import asyncio
import dataclasses
from collections import deque
from typing import TYPE_CHECKING, Callable

import torch

from ..trajectory import TrajectoryBuilder

if TYPE_CHECKING:
    from ..config import RolloutConfig
    from ..env.base import Env
    from ..template.spec import TemplateSpec
    from ..trajectory import Trajectory


def classify_truncation(out_tokens: list[int], tokenizer, template) -> dict:
    text = tokenizer.decode(out_tokens, skip_special_tokens=False)
    if "<think>" in text and "</think>" not in text:
        return {"truncated": True, "location": "think", "last_open_tag": "<think>"}
    if "<tool_call>" in text and "</tool_call>" not in text:
        return {"truncated": True, "location": "tool_call", "last_open_tag": "<tool_call>"}
    return {"truncated": True, "location": "answer", "last_open_tag": None}


def _missing_assistant_close_suffix(text: str, assistant_close: str) -> str:
    for n in range(len(assistant_close), 0, -1):
        if text.endswith(assistant_close[:n]):
            return assistant_close[n:]
    return assistant_close


def _env_metadata(env: "Env") -> dict:
    metadata = getattr(env, "last_metadata", None)
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}


def _with_reward_metadata(env: "Env", traj: "Trajectory", reward: float) -> "Trajectory":
    metadata = getattr(env, "last_reward_metadata", None)
    if not isinstance(metadata, dict):
        return dataclasses.replace(traj, reward=reward)
    meta = dict(traj.meta)
    meta["reward"] = dict(metadata)
    return dataclasses.replace(traj, reward=reward, meta=meta)


def _close_env(env: "Env") -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _full_logp_old(
    seq_len: int,
    segments: list[tuple[int, list[float]]],
) -> torch.FloatTensor | None:
    if not segments:
        return None
    logp_old = torch.zeros(seq_len, dtype=torch.float32)
    for start, values in segments:
        end = start + len(values)
        logp_old[start:end] = torch.tensor(values, dtype=torch.float32)
    return logp_old


@dataclasses.dataclass
class _ActiveState:
    group_id: int
    rollout_id: int
    seed_env: int
    env: "Env"
    builder: TrajectoryBuilder
    policy_version: int = 0
    lora_request: object | None = None
    turn: int = 0
    logprob_segments: list[tuple[int, list[float]]] = dataclasses.field(
        default_factory=list
    )


class MultiTurnRollout:

    def __init__(
        self,
        env: Env,
        template: TemplateSpec,
        backend,
        tokenizer,
        env_factory: Callable[[], Env] | None = None,
    ):
        self.env = env
        self.template = template
        self.backend = backend
        self.tokenizer = tokenizer
        self.env_factory = env_factory

    def _new_async_env(self) -> "Env":
        if self.env_factory is None:
            raise ValueError("async refill requires an env_factory for isolated env state")
        return self.env_factory()

    def _init_active_state(
        self,
        *,
        cfg: RolloutConfig,
        group_id: int,
        rollout_id: int,
        seed_env: int,
        policy_version: int = 0,
    ) -> _ActiveState:
        env = self._new_async_env()
        messages = env.reset(seed=seed_env)
        env_metadata = _env_metadata(env)
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tools=env.tools or None,
            add_generation_prompt=False, tokenize=False,
        )

        builder = TrajectoryBuilder(self.tokenizer)
        builder.meta.update(env_metadata)
        builder.add_prompt(prompt_text)
        current_lora = getattr(self.backend, "current_lora_request", None)
        return _ActiveState(
            group_id=group_id,
            rollout_id=rollout_id,
            seed_env=seed_env,
            env=env,
            builder=builder,
            policy_version=policy_version,
            lora_request=current_lora() if callable(current_lora) else None,
        )

    async def _generate_active_state(
        self,
        state: _ActiveState,
        cfg: RolloutConfig,
        seed_rollout: int,
        request_id: str,
    ) -> tuple[list[int], str, list[float] | None]:
        state.builder.add_prompt(self.template.assistant_open)
        eos_id = self.tokenizer.convert_tokens_to_ids(self.template.eos_token)
        seed = seed_rollout + state.rollout_id * cfg.max_turns + state.turn
        if hasattr(self.backend, "generate_async"):
            return await self.backend.generate_async(
                list(state.builder.tokens),
                max_new=cfg.max_tokens_per_turn,
                stop_token_ids=[eos_id],
                stop=self.template.stop_strings,
                seed=seed,
                request_id=request_id,
                return_logprobs=True,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                lora_request=state.lora_request,
            )

        out_tokens, finish_reason = await asyncio.to_thread(
            self.backend.generate,
            list(state.builder.tokens),
            cfg.max_tokens_per_turn,
            [eos_id],
            self.template.stop_strings,
            seed,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
        )
        return out_tokens, finish_reason, None

    def _finalize_active_state(self, state: _ActiveState) -> "Trajectory":
        traj = state.builder.freeze()
        meta = dict(traj.meta)
        meta["policy_version"] = state.policy_version
        meta["group_id"] = state.group_id
        meta["rollout_id"] = state.rollout_id
        meta["seed_env"] = state.seed_env
        traj = dataclasses.replace(traj, meta=meta)
        logp_old = _full_logp_old(traj.tokens.shape[0], state.logprob_segments)
        try:
            reward = state.env.reward(traj)
            traj = _with_reward_metadata(state.env, traj, reward)
        finally:
            _close_env(state.env)
        return dataclasses.replace(traj, logp_old=logp_old)

    def _advance_active_state(
        self,
        state: _ActiveState,
        cfg: RolloutConfig,
        out_tokens: list[int],
        finish_reason: str,
        gen_logprobs: list[float] | None,
    ) -> "Trajectory | None":
        gen_start = len(state.builder.tokens)
        state.builder.add_generated(out_tokens)
        if gen_logprobs is not None:
            state.logprob_segments.append((gen_start, gen_logprobs))

        if finish_reason == "length":
            state.builder.meta["truncation"] = {
                **classify_truncation(out_tokens, self.tokenizer, self.template),
                "reason": "max_tokens_per_turn",
            }
            return self._finalize_active_state(state)

        if len(state.builder.tokens) >= cfg.max_total_tokens:
            state.builder.meta["truncation"] = {
                **classify_truncation(out_tokens, self.tokenizer, self.template),
                "reason": "max_total_tokens",
            }
            return self._finalize_active_state(state)

        gen_text = self.tokenizer.decode(out_tokens, skip_special_tokens=False)
        calls = self.template.parse_tool_calls(gen_text)
        if not calls:
            return self._finalize_active_state(state)

        close_suffix = _missing_assistant_close_suffix(
            gen_text, self.template.assistant_close
        )
        if close_suffix:
            state.builder.add_prompt(close_suffix)

        in_user_block = False
        done = False
        for call in calls:
            resp, done = state.env.step(call)
            if not in_user_block:
                state.builder.add_prompt(self.template.tool_block_open)
                in_user_block = True
            else:
                state.builder.add_prompt(self.template.tool_resp_between)
            state.builder.add_env(
                self.template.tool_resp_open
                + self.template.format_tool_response(resp)
                + self.template.tool_resp_close
            )
            if done:
                break
        if in_user_block:
            state.builder.add_prompt(self.template.tool_block_close)
        if done:
            return self._finalize_active_state(state)

        if state.turn + 1 >= cfg.max_turns:
            state.builder.meta["truncation"] = {
                "truncated": True,
                "location": "answer",
                "last_open_tag": None,
                "reason": "max_turns_no_answer",
            }
            return self._finalize_active_state(state)

        state.turn += 1
        return None

    def run_multi_group_refill(
        self,
        cfg: RolloutConfig,
        seed_envs: list[int],
        seed_rollout: int,
        group_size: int,
    ) -> list[list["Trajectory"]]:
        """Run M groups × G rollouts with bounded async refill."""
        return asyncio.run(self._run_multi_group_refill_async(
            cfg=cfg,
            seed_envs=seed_envs,
            seed_rollout=seed_rollout,
            group_size=group_size,
        ))

    async def _run_multi_group_refill_async(
        self,
        *,
        cfg: RolloutConfig,
        seed_envs: list[int],
        seed_rollout: int,
        group_size: int,
    ) -> list[list["Trajectory"]]:
        states = deque(
            self._init_active_state(
                cfg=cfg,
                group_id=group_id,
                rollout_id=rollout_id,
                seed_env=seed_env,
            )
            for group_id, seed_env in enumerate(seed_envs)
            for rollout_id in range(group_size)
        )
        total = len(seed_envs) * group_size
        max_inflight = cfg.async_max_inflight or total or 1
        max_inflight = max(1, min(max_inflight, total or 1))
        groups: list[list[Trajectory | None]] = [
            [None for _ in range(group_size)] for _ in seed_envs
        ]
        tasks: dict[asyncio.Task, tuple[_ActiveState, str]] = {}
        request_seq = 0

        def submit(state: _ActiveState) -> None:
            nonlocal request_seq
            request_id = (
                f"rollout-g{state.group_id}-r{state.rollout_id}-"
                f"t{state.turn}-{request_seq}"
            )
            request_seq += 1
            task = asyncio.create_task(
                self._generate_active_state(state, cfg, seed_rollout, request_id)
            )
            tasks[task] = (state, request_id)

        try:
            while states or tasks:
                while states and len(tasks) < max_inflight:
                    submit(states.popleft())
                done, _ = await asyncio.wait(
                    tasks.keys(), return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    state, _ = tasks.pop(task)
                    out_tokens, finish_reason, gen_logprobs = task.result()
                    traj = self._advance_active_state(
                        state, cfg, out_tokens, finish_reason, gen_logprobs
                    )
                    if traj is None:
                        states.append(state)
                    else:
                        groups[state.group_id][state.rollout_id] = traj
        except BaseException:
            if hasattr(self.backend, "abort_requests_async"):
                await self.backend.abort_requests_async([
                    req_id for _, req_id in tasks.values()
                ])
            elif hasattr(self.backend, "abort_requests"):
                self.backend.abort_requests([req_id for _, req_id in tasks.values()])
            for task in tasks:
                task.cancel()
            raise

        finalized: list[list[Trajectory]] = []
        for group in groups:
            if any(t is None for t in group):
                raise RuntimeError("async refill finished with incomplete group")
            finalized.append([t for t in group if t is not None])
        return finalized

    def run_multi_group_batch(
        self,
        cfg: RolloutConfig,
        seed_envs: list[int],
        seed_rollout: int,
        group_size: int,
    ) -> list[list["Trajectory"]]:
        """Run M groups × G rollouts in a single vLLM call (single-turn)."""
        all_builders = []
        all_token_ids = []
        all_seeds = []

        for seed_env in seed_envs:
            messages = self.env.reset(seed=seed_env)
            env_metadata = _env_metadata(self.env)
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tools=self.env.tools or None,
                add_generation_prompt=False, tokenize=False,
            )

            for rollout_id in range(group_size):
                b = TrajectoryBuilder(self.tokenizer)
                b.meta.update(env_metadata)
                b.add_prompt(prompt_text)
                b.add_prompt(self.template.assistant_open)
                all_builders.append(b)
                all_token_ids.append(list(b.tokens))
                all_seeds.append(seed_rollout + rollout_id * cfg.max_turns)

        eos_id = self.tokenizer.convert_tokens_to_ids(self.template.eos_token)
        results = self.backend.generate_batch(
            all_token_ids,
            max_new=cfg.max_tokens_per_turn,
            stop_token_ids=[eos_id],
            stop=self.template.stop_strings,
            seeds=all_seeds,
            return_logprobs=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
        )

        groups: list[list[Trajectory]] = []
        idx = 0
        for gi, seed_env in enumerate(seed_envs):
            self.env.reset(seed=seed_env)
            group_trajs = []
            for _ in range(group_size):
                b = all_builders[idx]
                out_tokens, finish_reason, gen_logprobs = results[idx]
                b.add_generated(out_tokens)
                if finish_reason == "length":
                    b.meta["truncation"] = {
                        **classify_truncation(out_tokens, self.tokenizer, self.template),
                        "reason": "max_tokens_per_turn",
                    }
                traj = b.freeze()

                # Build full-sequence logp_old from vLLM logprobs
                # Prompt tokens get 0.0, generated tokens get vLLM logprobs
                logp_old = None
                if gen_logprobs is not None:
                    seq_len = traj.tokens.shape[0]
                    logp_old = torch.zeros(seq_len, dtype=torch.float32)
                    gen_start = seq_len - len(out_tokens)
                    logp_old[gen_start:] = torch.tensor(gen_logprobs, dtype=torch.float32)

                reward = self.env.reward(traj)
                traj = _with_reward_metadata(self.env, traj, reward)
                traj = dataclasses.replace(traj, logp_old=logp_old)
                group_trajs.append(traj)
                idx += 1
            groups.append(group_trajs)
        return groups

    def run_batch(
        self, cfg: RolloutConfig, seed_env: int, seed_rollout: int, group_size: int,
    ) -> list[Trajectory]:
        """Single-group convenience wrapper around run_multi_group_batch."""
        groups = self.run_multi_group_batch(cfg, [seed_env], seed_rollout, group_size)
        return groups[0]

    def run(self, cfg: RolloutConfig, seed_env: int, seed_rollout: int, rollout_id: int) -> Trajectory:
        messages = self.env.reset(seed=seed_env)

        builder = TrajectoryBuilder(self.tokenizer)
        builder.meta.update(_env_metadata(self.env))
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tools=self.env.tools or None,
            add_generation_prompt=False,
            tokenize=False,
        )
        builder.add_prompt(prompt_text)

        for turn in range(cfg.max_turns):
            done = False

            builder.add_prompt(self.template.assistant_open)

            out_tokens, finish_reason = self.backend.generate(
                builder.tokens,
                max_new=cfg.max_tokens_per_turn,
                stop_token_ids=[self.tokenizer.convert_tokens_to_ids(self.template.eos_token)],
                stop=self.template.stop_strings,
                seed=seed_rollout + rollout_id * cfg.max_turns + turn,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
            )
            builder.add_generated(out_tokens)

            if finish_reason == "length":
                builder.meta["truncation"] = {
                    **classify_truncation(out_tokens, self.tokenizer, self.template),
                    "reason": "max_tokens_per_turn",
                }
                break

            if len(builder.tokens) >= cfg.max_total_tokens:
                builder.meta["truncation"] = {
                    **classify_truncation(out_tokens, self.tokenizer, self.template),
                    "reason": "max_total_tokens",
                }
                break

            gen_text = self.tokenizer.decode(out_tokens, skip_special_tokens=False)
            calls = self.template.parse_tool_calls(gen_text)
            if not calls:
                break

            close_suffix = _missing_assistant_close_suffix(
                gen_text, self.template.assistant_close
            )
            if close_suffix:
                builder.add_prompt(close_suffix)
            in_user_block = False
            for call in calls:
                resp, done = self.env.step(call)
                if not in_user_block:
                    builder.add_prompt(self.template.tool_block_open)
                    in_user_block = True
                else:
                    builder.add_prompt(self.template.tool_resp_between)
                builder.add_env(
                    self.template.tool_resp_open
                    + self.template.format_tool_response(resp)
                    + self.template.tool_resp_close
                )
                if done:
                    break
            if in_user_block:
                builder.add_prompt(self.template.tool_block_close)
            if done:
                break
        else:
            builder.meta["truncation"] = {
                "truncated": True,
                "location": "answer",
                "last_open_tag": None,
                "reason": "max_turns_no_answer",
            }

        traj = builder.freeze()
        try:
            reward = self.env.reward(traj)
            return _with_reward_metadata(self.env, traj, reward)
        finally:
            _close_env(self.env)
