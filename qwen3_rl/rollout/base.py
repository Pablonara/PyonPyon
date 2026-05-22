from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

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


class MultiTurnRollout:

    def __init__(self, env: Env, template: TemplateSpec, backend, tokenizer):
        self.env = env
        self.template = template
        self.backend = backend
        self.tokenizer = tokenizer

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
        reward = self.env.reward(traj)
        return _with_reward_metadata(self.env, traj, reward)
