from __future__ import annotations

import json
import threading
from urllib.parse import quote
from urllib.request import Request, urlopen

import pytest

from qwen3_rl.scripts.rollout_viewer import (
    TraceLoadError,
    build_trace_summary,
    load_trace_records,
    make_server,
)


def _sample_record(**overrides):
    record = {
        "version": 1,
        "run_id": "run-a",
        "iter_id": 3,
        "group_id": 1,
        "rollout_id": 2,
        "seed_env": 11,
        "seed_rollout": 22,
        "reward": 1.0,
        "advantage": 0.25,
        "seq_len": 128,
        "n_trainable": 42,
        "mask_summary": {"prompt": 20, "gen": 42, "env": 10, "trainable": 42},
        "truncation": None,
        "turns": [
            {
                "kind": "prompt",
                "start": 0,
                "end": 20,
                "mask_any": False,
                "mask_all": False,
                "text": "Solve 2+2",
                "parsed": {},
            },
            {
                "kind": "gen",
                "start": 20,
                "end": 62,
                "mask_any": True,
                "mask_all": True,
                "text": "Compute it\n</think>\n\nThe answer is 4.",
                "parsed": {},
            },
        ],
        "raw_text": "Solve 2+2\nCompute it\n</think>\n\nThe answer is 4.",
    }
    record.update(overrides)
    return record


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_load_trace_records_reads_sorted_jsonl_and_adds_source(tmp_path):
    _write_jsonl(tmp_path / "b.jsonl", [_sample_record(run_id="run-b")])
    (tmp_path / "a.jsonl").write_text(
        "\n" + json.dumps(_sample_record(run_id="run-a")) + "\n",
        encoding="utf-8",
    )

    records = load_trace_records(tmp_path)

    assert [record["run_id"] for record in records] == ["run-a", "run-b"]
    assert records[0]["_id"] == "a.jsonl:2"
    assert records[0]["_file"] == "a.jsonl"
    assert records[0]["_line"] == 2


def test_load_trace_records_skips_unfinished_final_line(tmp_path):
    valid = json.dumps(_sample_record())
    (tmp_path / "trace.jsonl").write_text(
        valid + "\n" + '{"version": 1, "run_id":',
        encoding="utf-8",
    )

    records = load_trace_records(tmp_path)

    assert len(records) == 1
    assert records[0]["run_id"] == "run-a"


def test_load_trace_records_errors_on_invalid_complete_line(tmp_path):
    (tmp_path / "trace.jsonl").write_text('{"version": 1,\n', encoding="utf-8")

    with pytest.raises(TraceLoadError):
        load_trace_records(tmp_path)


def test_build_trace_summary_counts_turns_and_metadata():
    record = _sample_record(
        _id="trace.jsonl:1",
        _file="trace.jsonl",
        _line=1,
        truncation={"reason": "max_total_tokens"},
    )

    summary = build_trace_summary(record)

    assert summary["id"] == "trace.jsonl:1"
    assert summary["truncated"] is True
    assert summary["turn_count"] == 2
    assert summary["turn_kinds"] == {"prompt": 1, "gen": 1}
    assert summary["raw_chars"] == len(record["raw_text"])
    assert summary["mask_summary"]["trainable"] == 42


def test_http_api_serves_trace_list_detail_and_html(tmp_path):
    _write_jsonl(tmp_path / "trace.jsonl", [_sample_record()])
    server = make_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    host, port = server.server_address[:2]
    base_url = f"http://{host}:{port}"

    try:
        with urlopen(f"{base_url}/api/traces") as response:
            listing = json.loads(response.read().decode("utf-8"))

        assert listing["count"] == 1
        trace_id = listing["traces"][0]["id"]
        assert trace_id == "trace.jsonl:1"
        assert listing["traces"][0]["reward"] == 1.0

        with urlopen(f"{base_url}/api/traces/{quote(trace_id, safe='')}") as response:
            detail = json.loads(response.read().decode("utf-8"))

        assert detail["trace"]["raw_text"].endswith("The answer is 4.")
        assert detail["trace"]["turns"][1]["kind"] == "gen"

        with urlopen(f"{base_url}/") as response:
            html = response.read().decode("utf-8")

        assert "Rollout Trace Viewer" in html
        assert "/api/traces" in html
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_http_api_clear_hides_current_records_without_deleting_files(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    _write_jsonl(trace_path, [_sample_record()])
    server = make_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    host, port = server.server_address[:2]
    base_url = f"http://{host}:{port}"

    try:
        request = Request(f"{base_url}/api/traces/clear", method="POST")
        with urlopen(request) as response:
            result = json.loads(response.read().decode("utf-8"))

        assert result["hidden_count"] == 1
        assert trace_path.exists()

        with urlopen(f"{base_url}/api/traces") as response:
            listing = json.loads(response.read().decode("utf-8"))

        assert listing["count"] == 0
        assert listing["traces"] == []
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
