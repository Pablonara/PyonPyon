from __future__ import annotations

import asyncio
import dataclasses
import threading
import time
from collections import deque
from contextlib import suppress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import RolloutConfig
    from ..trajectory import Trajectory
    from .base import MultiTurnRollout, _ActiveState


@dataclasses.dataclass
class RolloutGroup:
    group_id: int
    seed_env: int
    policy_version: int
    trajectories: list["Trajectory"]
    created_at: float
    completed_at: float


@dataclasses.dataclass
class RolloutPoolMetrics:
    consumed_groups: int = 0
    dropped_stale_groups: int = 0
    dropped_active_stale_groups: int = 0
    completed_groups: int = 0
    active_groups: int = 0
    inflight_requests: int = 0
    ready_requests: int = 0
    wait_s: float = 0.0
    mean_staleness: float = 0.0
    max_staleness: int = 0


@dataclasses.dataclass
class _GroupState:
    group_id: int
    seed_env: int
    policy_version: int
    created_at: float
    trajectories: list["Trajectory | None"]


class AsyncRolloutPool:
    """Persistent cross-step async rollout pool.

    The pool keeps vLLM requests in flight in a background event loop and returns
    complete GRPO groups to the trainer.  Groups are tagged with the policy
    version active at creation time so stale groups can be dropped.
    """

    def __init__(
        self,
        rollout: "MultiTurnRollout",
        cfg: "RolloutConfig",
        *,
        seed_env_base: int,
        seed_rollout: int,
        group_size: int,
        target_groups: int,
        max_inflight_requests: int,
        max_off_policy_steps: int,
        drop_stale: bool,
    ) -> None:
        self.rollout = rollout
        self.cfg = cfg
        self.seed_env_base = seed_env_base
        self.seed_rollout = seed_rollout
        self.group_size = group_size
        self.target_groups = max(1, target_groups)
        self.max_inflight_requests = max(1, max_inflight_requests)
        self.max_off_policy_steps = max_off_policy_steps
        self.drop_stale = drop_stale

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="qwen3-rl-rollout-pool",
            daemon=True,
        )
        self._started = threading.Event()

        self._closed = False
        self._manager_task: asyncio.Task | None = None
        self._changed: asyncio.Event | None = None

        self._current_policy_version = 0
        self._next_group_id = 0
        self._request_seq = 0
        self._ready_states: deque["_ActiveState"] = deque()
        self._groups: dict[int, _GroupState] = {}
        self._completed: deque[RolloutGroup] = deque()
        self._tasks: dict[asyncio.Task, tuple["_ActiveState", str]] = {}
        self._dropped_stale_groups = 0
        self._dropped_active_stale_groups = 0

    def start(self, policy_version: int = 0) -> None:
        self._current_policy_version = policy_version
        self._thread.start()
        self._started.wait(timeout=10)
        self._run(self._set_policy_version(policy_version))

    def update_policy_version(self, policy_version: int) -> None:
        self._run(self._set_policy_version(policy_version))

    def get_completed_groups(
        self,
        n: int,
        *,
        current_policy_version: int,
    ) -> tuple[list[RolloutGroup], RolloutPoolMetrics]:
        return self._run(self._get_completed_groups(n, current_policy_version))

    def close(self) -> None:
        if not self._thread.is_alive():
            return
        self._run(self._close_async())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._changed = asyncio.Event()
        self._manager_task = self._loop.create_task(self._manager_loop())
        self._started.set()
        self._loop.run_forever()

    def _run(self, coro):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    async def _set_policy_version(self, policy_version: int) -> None:
        self._current_policy_version = policy_version
        await self._drop_stale_active_groups(policy_version)
        self._ensure_backlog()
        self._notify()

    def _notify(self) -> None:
        if self._changed is not None:
            self._changed.set()

    def _new_group(self) -> None:
        group_id = self._next_group_id
        self._next_group_id += 1
        seed_env = self.seed_env_base + group_id
        policy_version = self._current_policy_version
        self._groups[group_id] = _GroupState(
            group_id=group_id,
            seed_env=seed_env,
            policy_version=policy_version,
            created_at=time.time(),
            trajectories=[None for _ in range(self.group_size)],
        )
        for rollout_id in range(self.group_size):
            self._ready_states.append(
                self.rollout._init_active_state(
                    cfg=self.cfg,
                    group_id=group_id,
                    rollout_id=rollout_id,
                    seed_env=seed_env,
                    policy_version=policy_version,
                )
            )

    def _ensure_backlog(self) -> None:
        while (
            len(self._completed) + len(self._groups)
            < self.target_groups
        ):
            self._new_group()

    async def _drop_stale_active_groups(self, current_policy_version: int) -> None:
        if not self.drop_stale:
            return
        stale_ids = {
            gid for gid, group in self._groups.items()
            if current_policy_version - group.policy_version > self.max_off_policy_steps
        }
        if not stale_ids:
            return

        self._ready_states = deque(
            state for state in self._ready_states
            if state.group_id not in stale_ids
        )

        request_ids = []
        for task, (state, request_id) in list(self._tasks.items()):
            if state.group_id not in stale_ids:
                continue
            request_ids.append(request_id)
            task.cancel()
            self._tasks.pop(task, None)

        if request_ids and hasattr(self.rollout.backend, "abort_requests_async"):
            await self.rollout.backend.abort_requests_async(request_ids)
        elif request_ids and hasattr(self.rollout.backend, "abort_requests"):
            self.rollout.backend.abort_requests(request_ids)

        dropped = 0
        for gid in stale_ids:
            if self._groups.pop(gid, None) is not None:
                dropped += 1
        self._dropped_stale_groups += dropped
        self._dropped_active_stale_groups += dropped

    def _submit_ready(self) -> None:
        while self._ready_states and len(self._tasks) < self.max_inflight_requests:
            state = self._ready_states.popleft()
            request_id = (
                f"pool-g{state.group_id}-r{state.rollout_id}-"
                f"t{state.turn}-{self._request_seq}"
            )
            self._request_seq += 1
            task = asyncio.create_task(
                self.rollout._generate_active_state(
                    state, self.cfg, self.seed_rollout, request_id
                )
            )
            self._tasks[task] = (state, request_id)

    async def _manager_loop(self) -> None:
        try:
            while not self._closed:
                self._ensure_backlog()
                self._submit_ready()

                if not self._tasks:
                    await asyncio.sleep(0.01)
                    continue

                done, _ = await asyncio.wait(
                    self._tasks.keys(),
                    timeout=0.05,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    item = self._tasks.pop(task, None)
                    if item is None or task.cancelled():
                        continue
                    state, _ = item
                    try:
                        out_tokens, finish_reason, gen_logprobs = task.result()
                    except Exception:
                        self._ready_states = deque(
                            ready for ready in self._ready_states
                            if ready.group_id != state.group_id
                        )
                        self._groups.pop(state.group_id, None)
                        self._ensure_backlog()
                        self._notify()
                        continue
                    traj = await asyncio.to_thread(
                        self.rollout._advance_active_state,
                        state, self.cfg, out_tokens, finish_reason, gen_logprobs,
                    )
                    if traj is None:
                        self._ready_states.append(state)
                        continue

                    group = self._groups.get(state.group_id)
                    if group is None:
                        continue
                    group.trajectories[state.rollout_id] = traj
                    if all(t is not None for t in group.trajectories):
                        trajectories = [
                            t for t in group.trajectories if t is not None
                        ]
                        self._completed.append(RolloutGroup(
                            group_id=group.group_id,
                            seed_env=group.seed_env,
                            policy_version=group.policy_version,
                            trajectories=trajectories,
                            created_at=group.created_at,
                            completed_at=time.time(),
                        ))
                        del self._groups[group.group_id]
                        self._notify()
        except asyncio.CancelledError:
            pass

    def _staleness(self, group: RolloutGroup, current_policy_version: int) -> int:
        return current_policy_version - group.policy_version

    async def _get_completed_groups(
        self,
        n: int,
        current_policy_version: int,
    ) -> tuple[list[RolloutGroup], RolloutPoolMetrics]:
        start = time.time()
        selected: list[RolloutGroup] = []
        await self._drop_stale_active_groups(current_policy_version)
        dropped = self._dropped_stale_groups
        dropped_active = self._dropped_active_stale_groups
        self._dropped_stale_groups = 0
        self._dropped_active_stale_groups = 0
        staleness_values: list[int] = []

        while len(selected) < n:
            self._ensure_backlog()
            while self._completed and len(selected) < n:
                group = self._completed.popleft()
                stale = self._staleness(group, current_policy_version)
                if self.drop_stale and stale > self.max_off_policy_steps:
                    dropped += 1
                    continue
                for traj in group.trajectories:
                    if stale > 0 and traj.logp_old is None:
                        dropped += 1
                        break
                else:
                    selected.append(group)
                    staleness_values.append(stale)

            if len(selected) >= n:
                break
            assert self._changed is not None
            self._changed.clear()
            await self._changed.wait()

        self._ensure_backlog()
        metrics = RolloutPoolMetrics(
            consumed_groups=len(selected),
            dropped_stale_groups=dropped,
            dropped_active_stale_groups=dropped_active,
            completed_groups=len(self._completed),
            active_groups=len(self._groups),
            inflight_requests=len(self._tasks),
            ready_requests=len(self._ready_states),
            wait_s=time.time() - start,
            mean_staleness=(
                sum(staleness_values) / len(staleness_values)
                if staleness_values else 0.0
            ),
            max_staleness=max(staleness_values) if staleness_values else 0,
        )
        return selected, metrics

    async def _close_async(self) -> None:
        self._closed = True
        if self._manager_task is not None:
            self._manager_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._manager_task
        request_ids = [request_id for _, request_id in self._tasks.values()]
        if request_ids and hasattr(self.rollout.backend, "abort_requests_async"):
            await self.rollout.backend.abort_requests_async(request_ids)
        elif request_ids and hasattr(self.rollout.backend, "abort_requests"):
            self.rollout.backend.abort_requests(request_ids)
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        self._ready_states.clear()
        self._groups.clear()
        self._completed.clear()
