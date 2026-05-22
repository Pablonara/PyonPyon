from __future__ import annotations

import atexit
import json
import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .template.spec import TemplateSpec
    from .trajectory import Trajectory


TRACE_VERSION = 1
_TOOL_RESPONSE_RE = re.compile(r"<tool_response>\n?(.*?)\n?</tool_response>", re.DOTALL)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "run"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return repr(value)


def _parse_turn(kind: str, text: str, template: TemplateSpec) -> dict[str, Any]:
    if kind == "gen":
        calls = template.parse_tool_calls(text)
        if not calls:
            return {}
        return {
            "tool_calls": [
                {
                    "name": call.name,
                    "arguments": _jsonable(call.arguments),
                    "raw_text": call.raw_text,
                }
                for call in calls
            ]
        }

    if kind == "env":
        responses = [match.group(1) for match in _TOOL_RESPONSE_RE.finditer(text)]
        if responses:
            return {"tool_responses": responses}
    return {}


def trajectory_to_trace_record(
    trajectory: Trajectory,
    tokenizer,
    template: TemplateSpec,
    *,
    run_id: str,
    iter_id: int,
    group_id: int,
    rollout_id: int,
    seed_env: int,
    seed_rollout: int,
) -> dict[str, Any]:
    token_ids = trajectory.tokens.tolist()
    mask = [bool(v) for v in trajectory.mask.tolist()]

    raw_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    turns = []
    mask_summary = {"prompt": 0, "gen": 0, "env": 0, "trainable": int(sum(mask))}

    for start, end, kind in trajectory.turns:
        kind = str(kind)
        turn_tokens = token_ids[start:end]
        turn_mask = mask[start:end]
        text = tokenizer.decode(turn_tokens, skip_special_tokens=False)
        if kind in mask_summary:
            mask_summary[kind] += end - start
        turns.append(
            {
                "kind": kind,
                "start": start,
                "end": end,
                "mask_any": any(turn_mask),
                "mask_all": all(turn_mask) if turn_mask else False,
                "text": text,
                "parsed": _parse_turn(kind, text, template),
            }
        )

    return {
        "version": TRACE_VERSION,
        "run_id": run_id,
        "iter_id": iter_id,
        "group_id": group_id,
        "rollout_id": rollout_id,
        "seed_env": seed_env,
        "seed_rollout": seed_rollout,
        "reward": _jsonable(trajectory.reward),
        "advantage": _jsonable(trajectory.advantage),
        "seq_len": trajectory.seq_len,
        "n_trainable": trajectory.n_trainable,
        "mask_summary": mask_summary,
        "metadata": _jsonable(trajectory.meta),
        "truncation": _jsonable(trajectory.meta.get("truncation")),
        "turns": turns,
        "raw_text": raw_text,
    }


@dataclass(frozen=True)
class _TraceItem:
    trajectory: Trajectory
    iter_id: int
    group_id: int
    rollout_id: int
    seed_env: int
    seed_rollout: int


class RolloutTraceRecorder:
    """Best-effort JSONL recorder that never blocks rollout/training."""

    def __init__(
        self,
        trace_dir: str | None,
        tokenizer,
        template: TemplateSpec,
        *,
        run_id: str | None = None,
        max_queue: int = 1024,
    ):
        self.enabled = trace_dir is not None
        self.tokenizer = tokenizer
        self.template = template
        self.run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S")
        self.dropped = 0
        self.errors = 0
        self._closed = False

        if not self.enabled:
            self.trace_dir = None
            self.path = None
            self._queue = None
            self._thread = None
            return

        self.trace_dir = Path(trace_dir).expanduser()
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.trace_dir / f"{_safe_name(self.run_id)}.jsonl"
        self._queue: queue.Queue[_TraceItem | None] = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def record(
        self,
        trajectory: Trajectory,
        *,
        iter_id: int,
        group_id: int,
        rollout_id: int,
        seed_env: int,
        seed_rollout: int,
    ) -> None:
        if not self.enabled or self._closed or self._queue is None:
            return

        item = _TraceItem(
            trajectory=trajectory,
            iter_id=iter_id,
            group_id=group_id,
            rollout_id=rollout_id,
            seed_env=seed_env,
            seed_rollout=seed_rollout,
        )
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def close(self) -> None:
        if not self.enabled or self._closed:
            return
        self._closed = True
        if self._queue is None or self._thread is None:
            return
        try:
            self._queue.put(None, timeout=1.0)
        except queue.Full:
            self.dropped += 1
            return
        self._thread.join(timeout=5.0)

    def _writer_loop(self) -> None:
        assert self.path is not None
        assert self._queue is not None
        with self.path.open("a", encoding="utf-8", buffering=1) as handle:
            while True:
                item = self._queue.get()
                try:
                    if item is None:
                        return
                    record = trajectory_to_trace_record(
                        item.trajectory,
                        self.tokenizer,
                        self.template,
                        run_id=self.run_id,
                        iter_id=item.iter_id,
                        group_id=item.group_id,
                        rollout_id=item.rollout_id,
                        seed_env=item.seed_env,
                        seed_rollout=item.seed_rollout,
                    )
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception:
                    self.errors += 1
                finally:
                    self._queue.task_done()
