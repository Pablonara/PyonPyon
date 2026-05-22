"""Local read-only browser viewer for rollout trace JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


TRACE_VERSION = 1


class TraceLoadError(ValueError):
    """Raised when a trace JSONL file cannot be loaded."""


def discover_trace_files(trace_dir: str | Path) -> list[Path]:
    """Return sorted ``*.jsonl`` files in ``trace_dir``."""
    directory = Path(trace_dir)
    if not directory.exists():
        raise FileNotFoundError(f"trace directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"trace path is not a directory: {directory}")
    return sorted(
        (path for path in directory.glob("*.jsonl") if path.is_file()),
        key=lambda path: path.name,
    )


def load_trace_records(
    trace_dir: str | Path,
    *,
    skip_partial_final_line: bool = True,
) -> list[dict[str, Any]]:
    """Load all rollout trace records from JSONL files in a directory.

    A malformed final line without a trailing newline is skipped by default. That
    keeps the viewer usable while a recorder is actively appending to a file.
    """
    records: list[dict[str, Any]] = []

    for path in discover_trace_files(trace_dir):
        with path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()

        for line_no, line in enumerate(lines, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                is_final_line = line_no == len(lines)
                is_unfinished = not line.endswith("\n")
                if skip_partial_final_line and is_final_line and is_unfinished:
                    continue
                raise TraceLoadError(
                    f"{path}:{line_no}: invalid JSONL record: {exc.msg}"
                ) from exc

            if not isinstance(record, dict):
                raise TraceLoadError(
                    f"{path}:{line_no}: expected JSON object, got "
                    f"{type(record).__name__}"
                )

            enriched = dict(record)
            enriched["_id"] = f"{path.name}:{line_no}"
            enriched["_file"] = path.name
            enriched["_line"] = line_no
            records.append(enriched)

    return records


def _turn_kind_counts(turns: Any) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if isinstance(turns, list):
        for turn in turns:
            if isinstance(turn, dict):
                counts[str(turn.get("kind", "unknown"))] += 1
            else:
                counts["unknown"] += 1
    return dict(counts)


def build_trace_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Build the compact row shown by the trace list API."""
    turns = record.get("turns")
    raw_text = record.get("raw_text")
    return {
        "id": record.get("_id"),
        "file": record.get("_file"),
        "line": record.get("_line"),
        "version": record.get("version", TRACE_VERSION),
        "run_id": record.get("run_id"),
        "iter_id": record.get("iter_id"),
        "group_id": record.get("group_id"),
        "rollout_id": record.get("rollout_id"),
        "seed_env": record.get("seed_env"),
        "seed_rollout": record.get("seed_rollout"),
        "reward": record.get("reward"),
        "advantage": record.get("advantage"),
        "seq_len": record.get("seq_len"),
        "n_trainable": record.get("n_trainable"),
        "mask_summary": record.get("mask_summary"),
        "metadata": record.get("metadata"),
        "truncation": record.get("truncation"),
        "truncated": bool(record.get("truncation")),
        "turn_count": len(turns) if isinstance(turns, list) else 0,
        "turn_kinds": _turn_kind_counts(turns),
        "raw_chars": len(raw_text) if isinstance(raw_text, str) else 0,
    }


def _directory_signature(trace_dir: Path) -> tuple[tuple[str, int, int], ...]:
    signature: list[tuple[str, int, int]] = []
    for path in discover_trace_files(trace_dir):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        signature.append((path.name, stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


@dataclass
class TraceStore:
    """Small in-memory cache around JSONL trace files."""

    trace_dir: Path
    _signature: tuple[tuple[str, int, int], ...] | None = None
    _records: list[dict[str, Any]] = field(default_factory=list)
    _summaries: list[dict[str, Any]] = field(default_factory=list)
    _by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    _hidden_ids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.trace_dir = Path(self.trace_dir).expanduser().resolve()

    def reload_if_changed(self) -> None:
        signature = _directory_signature(self.trace_dir)
        if signature == self._signature:
            return

        records = load_trace_records(self.trace_dir)
        self._records = records
        self._summaries = [
            build_trace_summary(record)
            for record in records
            if str(record.get("_id")) not in self._hidden_ids
        ]
        self._by_id = {
            str(record["_id"]): record
            for record in records
            if record.get("_id") is not None
            and str(record.get("_id")) not in self._hidden_ids
        }
        self._signature = signature

    def list_summaries(self) -> list[dict[str, Any]]:
        self.reload_if_changed()
        return list(self._summaries)

    def get_record(self, trace_id: str) -> dict[str, Any] | None:
        self.reload_if_changed()
        record = self._by_id.get(trace_id)
        return dict(record) if record is not None else None

    def clear_visible_traces(self) -> int:
        """Hide currently loaded trace records from this viewer session."""
        self.reload_if_changed()
        visible_ids = {
            str(record["_id"])
            for record in self._records
            if record.get("_id") is not None
            and str(record.get("_id")) not in self._hidden_ids
        }
        self._hidden_ids.update(visible_ids)
        self._signature = None
        self.reload_if_changed()
        return len(visible_ids)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rollout Trace Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7fb;
      --panel: #ffffff;
      --panel-soft: #f9fafb;
      --text: #172033;
      --muted: #667085;
      --line: #d9dee8;
      --brand: #285cff;
      --prompt: #475467;
      --gen: #0f766e;
      --env: #7c3aed;
      --bad: #b42318;
      --good: #047857;
      --code: #101828;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      line-height: 1.4;
    }

    header {
      padding: 20px 28px 14px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 2;
    }

    h1 {
      font-size: 20px;
      margin: 0 0 8px;
    }

    .subtle {
      color: var(--muted);
      font-size: 13px;
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(420px, 42%) minmax(480px, 1fr);
      gap: 16px;
      padding: 16px;
      height: calc(100vh - 82px);
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      min-height: 0;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05);
    }

    .panel-head {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      background: var(--panel-soft);
    }

    .filters {
      display: grid;
      grid-template-columns: minmax(160px, 1fr) repeat(3, minmax(90px, auto)) auto auto;
      gap: 8px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      align-items: center;
    }

    input, select {
      border: 1px solid var(--line);
      border-radius: 9px;
      background: var(--panel);
      color: var(--text);
      padding: 6px 9px;
      font: inherit;
      font-size: 13px;
      min-width: 0;
    }

    .danger {
      color: var(--bad);
    }

    .panel-body {
      height: calc(100% - 49px);
      overflow: auto;
    }

    .panel-body.filtered {
      height: calc(100% - 104px);
    }

    button {
      border: 1px solid var(--line);
      border-radius: 9px;
      background: var(--panel);
      color: var(--text);
      padding: 6px 10px;
      cursor: pointer;
      font: inherit;
      font-size: 13px;
    }

    button:hover {
      border-color: var(--brand);
    }

    table {
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
    }

    th, td {
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }

    th {
      color: var(--muted);
      background: var(--panel-soft);
      position: sticky;
      top: 0;
      z-index: 1;
      font-weight: 600;
    }

    tr.trace-row {
      cursor: pointer;
    }

    tr.trace-row:hover,
    tr.trace-row.selected {
      background: #eef4ff;
    }

    .reward {
      font-variant-numeric: tabular-nums;
      font-weight: 700;
    }

    .meta-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 8px;
      padding: 14px;
    }

    .meta-card {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--panel-soft);
      padding: 9px 10px;
      min-width: 0;
    }

    .meta-label {
      color: var(--muted);
      display: block;
      font-size: 11px;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }

    .meta-value {
      display: block;
      margin-top: 3px;
      overflow-wrap: anywhere;
      font-variant-numeric: tabular-nums;
    }

    .transcript {
      padding: 0 14px 18px;
    }

    .turn {
      border: 1px solid var(--line);
      border-left-width: 5px;
      border-radius: 12px;
      margin: 12px 0;
      overflow: hidden;
      background: var(--panel);
    }

    .turn.prompt {
      border-left-color: var(--prompt);
    }

    .turn.gen {
      border-left-color: var(--gen);
    }

    .turn.env {
      border-left-color: var(--env);
    }

    .turn-head {
      padding: 9px 11px;
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-soft);
      font-size: 12px;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 2px 8px;
      font-weight: 700;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.02em;
      background: #eaecf0;
      color: #344054;
    }

    .badge.prompt {
      background: #f2f4f7;
      color: var(--prompt);
    }

    .badge.gen {
      background: #ccfbf1;
      color: var(--gen);
    }

    .badge.env {
      background: #ede9fe;
      color: var(--env);
    }

    .badge.warn {
      background: #fee4e2;
      color: var(--bad);
    }

    .badge.good {
      background: #d1fae5;
      color: var(--good);
    }

    .turn-body {
      padding: 12px;
    }

    details {
      border: 1px solid var(--line);
      border-radius: 10px;
      margin: 9px 0;
      background: var(--panel-soft);
    }

    details > summary {
      cursor: pointer;
      padding: 8px 10px;
      font-weight: 700;
      color: #344054;
    }

    pre {
      margin: 0;
      padding: 10px;
      overflow: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      color: var(--code);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
        "Liberation Mono", monospace;
      font-size: 12px;
      line-height: 1.45;
    }

    .answer {
      border: 1px solid #bae6fd;
      border-radius: 10px;
      background: #f0f9ff;
      margin: 9px 0;
    }

    .answer-label {
      padding: 8px 10px 0;
      color: #0369a1;
      font-weight: 700;
      font-size: 12px;
    }

    .empty {
      padding: 24px;
      color: var(--muted);
      text-align: center;
    }

    .error {
      color: var(--bad);
      padding: 14px;
    }

    @media (max-width: 980px) {
      .layout {
        grid-template-columns: 1fr;
        height: auto;
      }

      .filters {
        grid-template-columns: 1fr 1fr;
      }

      .panel {
        min-height: 420px;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>Rollout Trace Viewer</h1>
    <div id="status" class="subtle">Loading traces…</div>
  </header>

  <main class="layout">
    <section class="panel">
      <div class="panel-head">
        <strong>Traces</strong>
        <div>
          <button id="refresh" type="button">Refresh</button>
          <button id="clear-traces" class="danger" type="button">Clear list</button>
        </div>
      </div>
      <div class="filters">
        <input id="filter-search" placeholder="Search run/raw metadata…">
        <input id="filter-iter" placeholder="Iter">
        <select id="filter-reward">
          <option value="">Any reward</option>
          <option value="positive">Reward &gt; 0</option>
          <option value="zero">Reward = 0</option>
        </select>
        <select id="filter-truncated">
          <option value="">Any status</option>
          <option value="truncated">Truncated</option>
          <option value="complete">Complete</option>
        </select>
        <button id="reset-filters" type="button">Reset</button>
      </div>
      <div id="trace-list" class="panel-body filtered"></div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <strong>Detail</strong>
        <span id="selected" class="subtle">Select a trace</span>
      </div>
      <div id="detail" class="panel-body">
        <div class="empty">Select a trace to inspect the transcript.</div>
      </div>
    </section>
  </main>

  <script>
    const state = {
      traces: [],
      selectedId: null,
      filters: {
        search: "",
        iter: "",
        reward: "",
        truncated: "",
      },
    };

    const statusEl = document.getElementById("status");
    const listEl = document.getElementById("trace-list");
    const detailEl = document.getElementById("detail");
    const selectedEl = document.getElementById("selected");

    function escapeHtml(value) {
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    function fmt(value) {
      if (value === null || value === undefined || value === "") return "—";
      if (typeof value === "number") {
        return Number.isInteger(value) ? String(value) : value.toFixed(4);
      }
      return String(value);
    }

    function compactJson(value) {
      if (value === null || value === undefined) return "—";
      return JSON.stringify(value);
    }

    async function fetchJson(url, options = {}) {
      const response = await fetch(url, {cache: "no-store", ...options});
      const text = await response.text();
      let payload = {};
      if (text) payload = JSON.parse(text);
      if (!response.ok) {
        throw new Error(payload.error || response.statusText);
      }
      return payload;
    }

    async function loadTraces() {
      statusEl.textContent = "Loading traces…";
      try {
        const payload = await fetchJson("/api/traces");
        state.traces = payload.traces || [];
        renderTraceList();
        const filtered = getFilteredTraces();
        statusEl.textContent =
          `${filtered.length}/${state.traces.length} trace(s) from ${payload.trace_dir}`;
        if (state.selectedId) {
          await selectTrace(state.selectedId);
        }
      } catch (error) {
        listEl.innerHTML = `<div class="error">${escapeHtml(error.message)}</div>`;
        statusEl.textContent = "Failed to load traces";
      }
    }

    function getFilteredTraces() {
      return state.traces.filter((trace) => {
        const search = state.filters.search.trim().toLowerCase();
        if (search) {
          const haystack = [
            trace.id,
            trace.file,
            trace.run_id,
            trace.iter_id,
            trace.group_id,
            trace.rollout_id,
            trace.reward,
            trace.truncated ? "truncated" : "complete",
          ].map((v) => String(v ?? "").toLowerCase()).join(" ");
          if (!haystack.includes(search)) return false;
        }

        if (state.filters.iter.trim()) {
          if (String(trace.iter_id) !== state.filters.iter.trim()) return false;
        }

        if (state.filters.reward === "positive" && !(Number(trace.reward) > 0)) {
          return false;
        }
        if (state.filters.reward === "zero" && Number(trace.reward) !== 0) {
          return false;
        }

        if (state.filters.truncated === "truncated" && !trace.truncated) return false;
        if (state.filters.truncated === "complete" && trace.truncated) return false;

        return true;
      });
    }

    function renderTraceList() {
      const traces = getFilteredTraces();
      if (!state.traces.length) {
        listEl.innerHTML = '<div class="empty">No *.jsonl traces found.</div>';
        return;
      }
      if (!traces.length) {
        listEl.innerHTML = '<div class="empty">No traces match the current filters.</div>';
        return;
      }

      const rows = traces.map((trace) => {
        const selected = trace.id === state.selectedId ? " selected" : "";
        const truncated = trace.truncated
          ? '<span class="badge warn">truncated</span>'
          : '<span class="badge good">complete</span>';
        return `
          <tr class="trace-row${selected}" data-id="${escapeHtml(trace.id)}">
            <td>${escapeHtml(trace.file)}:${escapeHtml(trace.line)}</td>
            <td>${escapeHtml(fmt(trace.run_id))}</td>
            <td>${escapeHtml(fmt(trace.iter_id))}</td>
            <td>${escapeHtml(fmt(trace.group_id))}</td>
            <td>${escapeHtml(fmt(trace.rollout_id))}</td>
            <td class="reward">${escapeHtml(fmt(trace.reward))}</td>
            <td>${escapeHtml(fmt(trace.advantage))}</td>
            <td>${escapeHtml(fmt(trace.seq_len))}</td>
            <td>${escapeHtml(fmt(trace.n_trainable))}</td>
            <td>${truncated}</td>
          </tr>`;
      }).join("");

      listEl.innerHTML = `
        <table>
          <thead>
            <tr>
              <th>Source</th>
              <th>Run</th>
              <th>Iter</th>
              <th>Group</th>
              <th>Rollout</th>
              <th>Reward</th>
              <th>Adv</th>
              <th>Seq</th>
              <th>Trainable</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>`;
    }

    listEl.addEventListener("click", (event) => {
      const row = event.target.closest(".trace-row");
      if (!row) return;
      selectTrace(row.dataset.id);
    });

    async function selectTrace(id) {
      try {
        const payload = await fetchJson(`/api/traces/${encodeURIComponent(id)}`);
        state.selectedId = id;
        selectedEl.textContent = id;
        renderTraceList();
        renderDetail(payload.trace);
      } catch (error) {
        detailEl.innerHTML = `<div class="error">${escapeHtml(error.message)}</div>`;
      }
    }

    function renderMetaCard(label, value) {
      return `
        <div class="meta-card">
          <span class="meta-label">${escapeHtml(label)}</span>
          <span class="meta-value">${escapeHtml(fmt(value))}</span>
        </div>`;
    }

    function renderDetail(trace) {
      const turns = Array.isArray(trace.turns) ? trace.turns : [];
      const meta = [
        ["Run", trace.run_id],
        ["Iter", trace.iter_id],
        ["Group", trace.group_id],
        ["Rollout", trace.rollout_id],
        ["Reward", trace.reward],
        ["Advantage", trace.advantage],
        ["Seq len", trace.seq_len],
        ["Trainable tokens", trace.n_trainable],
        ["Seed env", trace.seed_env],
        ["Seed rollout", trace.seed_rollout],
        ["Metadata", compactJson(trace.metadata)],
        ["Mask summary", compactJson(trace.mask_summary)],
        ["Truncation", compactJson(trace.truncation)],
      ].map(([label, value]) => renderMetaCard(label, value)).join("");

      const transcript = turns.length
        ? turns.map((turn, index) => renderTurn(turn, index)).join("")
        : '<div class="empty">No turns recorded.</div>';

      detailEl.innerHTML = `
        <div class="meta-grid">${meta}</div>
        <div class="transcript">
          <h2 style="font-size:16px;margin:8px 0 4px;">Transcript</h2>
          ${transcript}
          <details>
            <summary>Raw text (${fmt((trace.raw_text || "").length)} chars)</summary>
            <pre>${escapeHtml(trace.raw_text || "")}</pre>
          </details>
        </div>`;
    }

    function renderTurn(turn, index) {
      const kind = String(turn.kind || "unknown");
      const maskAny = Boolean(turn.mask_any);
      const maskAll = Boolean(turn.mask_all);
      const maskBadge = maskAll
        ? '<span class="badge good">mask all</span>'
        : maskAny
          ? '<span class="badge warn">mask mixed</span>'
          : '<span class="badge">mask none</span>';
      const parsed = turn.parsed && Object.keys(turn.parsed).length
        ? `<details>
             <summary>Parsed metadata</summary>
             <pre>${escapeHtml(JSON.stringify(turn.parsed, null, 2))}</pre>
           </details>`
        : "";

      return `
        <article class="turn ${escapeHtml(kind)}">
          <div class="turn-head">
            <span class="badge ${escapeHtml(kind)}">${escapeHtml(kind)}</span>
            <span>#${index}</span>
            <span>tokens ${escapeHtml(fmt(turn.start))}–${escapeHtml(fmt(turn.end))}</span>
            ${maskBadge}
          </div>
          <div class="turn-body">
            ${renderTurnText(kind, turn.text || "")}
            ${parsed}
          </div>
        </article>`;
    }

    function renderTurnText(kind, text) {
      if (kind === "gen") return renderGeneratedText(text);
      if (kind === "env") return renderEnvText(text);
      if (!text) return '<div class="subtle">Empty turn text.</div>';
      return `<pre>${escapeHtml(text)}</pre>`;
    }

    function renderGeneratedText(text) {
      let rest = text || "";
      let output = "";

      const fullThink = rest.match(/^<think>\n?([\s\S]*?)<\/think>/);
      if (fullThink) {
        output += renderDetails("Thinking", fullThink[1], false);
        rest = rest.slice(fullThink[0].length);
      } else {
        const closeIndex = rest.indexOf("</think>");
        if (closeIndex >= 0) {
          output += renderDetails("Thinking", rest.slice(0, closeIndex), false);
          rest = rest.slice(closeIndex + "</think>".length);
        }
      }

      output += renderAnswerAndToolCalls(rest);
      return output || '<div class="subtle">Empty generated text.</div>';
    }

    function renderAnswerAndToolCalls(text) {
      const pattern = /<tool_call>[\s\S]*?<\/tool_call>/g;
      let cursor = 0;
      let output = "";
      let match;

      while ((match = pattern.exec(text)) !== null) {
        output += renderAnswerFragment(text.slice(cursor, match.index));
        output += renderDetails("Tool call", match[0], true);
        cursor = pattern.lastIndex;
      }

      output += renderAnswerFragment(text.slice(cursor));
      return output;
    }

    function renderAnswerFragment(text) {
      if (!text || !text.trim()) return "";
      return `
        <div class="answer">
          <div class="answer-label">Final answer / generated text</div>
          <pre>${escapeHtml(text.trim())}</pre>
        </div>`;
    }

    function renderEnvText(text) {
      const pattern = /<tool_response>\n?([\s\S]*?)\n?<\/tool_response>/g;
      let cursor = 0;
      let output = "";
      let match;

      while ((match = pattern.exec(text)) !== null) {
        output += renderBoundaryText(text.slice(cursor, match.index));
        output += renderDetails("Tool response", match[1], true);
        cursor = pattern.lastIndex;
      }

      output += renderBoundaryText(text.slice(cursor));
      if (!output && text) output = renderDetails("Tool response", text, true);
      return output || '<div class="subtle">Empty tool response.</div>';
    }

    function renderBoundaryText(text) {
      if (!text || !text.trim()) return "";
      return `<pre class="subtle">${escapeHtml(text.trim())}</pre>`;
    }

    function renderDetails(label, text, open) {
      return `
        <details ${open ? "open" : ""}>
          <summary>${escapeHtml(label)}</summary>
          <pre>${escapeHtml(text || "")}</pre>
        </details>`;
    }

    document.getElementById("refresh").addEventListener("click", loadTraces);
    document.getElementById("clear-traces").addEventListener("click", clearTraceList);
    document.getElementById("reset-filters").addEventListener("click", () => {
      state.filters = {search: "", iter: "", reward: "", truncated: ""};
      syncFilterInputs();
      renderTraceList();
    });

    for (const [id, key] of [
      ["filter-search", "search"],
      ["filter-iter", "iter"],
      ["filter-reward", "reward"],
      ["filter-truncated", "truncated"],
    ]) {
      document.getElementById(id).addEventListener("input", (event) => {
        state.filters[key] = event.target.value;
        renderTraceList();
        statusEl.textContent = `${getFilteredTraces().length}/${state.traces.length} trace(s) shown`;
      });
    }

    function syncFilterInputs() {
      document.getElementById("filter-search").value = state.filters.search;
      document.getElementById("filter-iter").value = state.filters.iter;
      document.getElementById("filter-reward").value = state.filters.reward;
      document.getElementById("filter-truncated").value = state.filters.truncated;
    }

    async function clearTraceList() {
      if (!confirm("Clear currently loaded traces from this viewer session? Trace files are not deleted.")) {
        return;
      }
      try {
        const payload = await fetchJson("/api/traces/clear", {method: "POST"});
        state.selectedId = null;
        selectedEl.textContent = "Select a trace";
        detailEl.innerHTML = '<div class="empty">Select a trace to inspect the transcript.</div>';
        await loadTraces();
        statusEl.textContent = `Cleared ${payload.hidden_count || 0} trace(s) from this viewer session.`;
      } catch (error) {
        statusEl.textContent = `Failed to clear traces: ${error.message}`;
      }
    }

    loadTraces();
  </script>
</body>
</html>
"""


class RolloutViewerHandler(BaseHTTPRequestHandler):
    """HTTP handler for the HTML UI and trace JSON API."""

    trace_store: TraceStore
    server_version = "RolloutTraceViewer/1.0"

    def do_GET(self) -> None:
        path = urlparse(self.path).path

        if path in {"/", "/index.html"}:
            self._send_bytes(
                INDEX_HTML.encode("utf-8"),
                content_type="text/html; charset=utf-8",
            )
            return

        if path == "/api/traces":
            self._handle_trace_list()
            return

        if path.startswith("/api/traces/"):
            trace_id = unquote(path[len("/api/traces/"):])
            self._handle_trace_detail(trace_id)
            return

        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/traces/clear":
            hidden_count = self.trace_store.clear_visible_traces()
            self._send_json({"hidden_count": hidden_count})
            return

        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _handle_trace_list(self) -> None:
        try:
            traces = self.trace_store.list_summaries()
        except TraceLoadError as exc:
            self._send_json(
                {"error": str(exc)},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        self._send_json(
            {
                "trace_dir": str(self.trace_store.trace_dir),
                "count": len(traces),
                "traces": traces,
            }
        )

    def _handle_trace_detail(self, trace_id: str) -> None:
        if not trace_id:
            self._send_json(
                {"error": "missing trace id"},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        try:
            record = self.trace_store.get_record(trace_id)
        except TraceLoadError as exc:
            self._send_json(
                {"error": str(exc)},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        if record is None:
            self._send_json(
                {"error": f"trace not found: {trace_id}"},
                status=HTTPStatus.NOT_FOUND,
            )
            return

        self._send_json({"trace": record})

    def _send_json(
        self,
        payload: dict[str, Any],
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(
            body,
            status=status,
            content_type="application/json; charset=utf-8",
        )

    def _send_bytes(
        self,
        body: bytes,
        *,
        status: HTTPStatus = HTTPStatus.OK,
        content_type: str,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def make_server(
    trace_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    """Create a local viewer server."""
    store = TraceStore(Path(trace_dir))
    store.reload_if_changed()

    class Handler(RolloutViewerHandler):
        trace_store = store

    return ThreadingHTTPServer((host, port), Handler)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a local rollout trace viewer for a directory of JSONL files.",
    )
    parser.add_argument("trace_dir", help="Directory containing *.jsonl trace files")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        server = make_server(args.trace_dir, host=args.host, port=args.port)
    except (FileNotFoundError, NotADirectoryError, TraceLoadError) as exc:
        print(f"rollout_viewer: {exc}", file=sys.stderr)
        return 2

    host, port = server.server_address[:2]
    print(f"Serving rollout traces from {Path(args.trace_dir).resolve()}")
    print(f"Open http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping rollout viewer.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
