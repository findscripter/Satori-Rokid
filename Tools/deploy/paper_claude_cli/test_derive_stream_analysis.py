"""Offline regressions for segmented single-turn parsing and immutable lineage."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from deploy.paper_claude_cli import derive_stream_analysis as derive
import run_paper_parallel as parallel
import satori_paper_benchmark as bench


def events(text='{"answer":"cup","evidence_frame_ids":["v000"]}'):
    return [
        {"type": "system", "subtype": "init", "model": derive.MODEL, "tool_count": 0, "mcp_server_count": 0},
        {"type": "system", "subtype": "thinking_tokens", "model": None, "tool_count": 0, "mcp_server_count": 0},
        {"type": "system", "subtype": "thinking_tokens", "model": None, "tool_count": 0, "mcp_server_count": 0},
        {"type": "assistant", "message": {"model": derive.MODEL, "stop_reason": None,
         "usage": {"input_tokens": 2, "output_tokens": 3},
         "content": [{"type": "thinking", "omitted_sha256": "a" * 64}]}},
        {"type": "assistant", "message": {"model": derive.MODEL, "stop_reason": None,
         "usage": {"input_tokens": 2, "output_tokens": 3}, "content": [{"type": "text", "text": text}]}},
        {"type": "rate_limit_event", "subtype": None},
        {"type": "result", "subtype": "success", "is_error": False, "num_turns": 1, "stop_reason": "end_turn",
         "result": text, "usage": {"input_tokens": 2, "output_tokens": 192},
         "modelUsage": {derive.MODEL: {"inputTokens": 2, "outputTokens": 192, "thinkingTokens": 156}}},
    ]


def trace():
    return {"cli_invocations": 1, "cli_exit_code": 0, "cli_binary_sha256": derive.CLI_BINARY_SHA,
            "cli_stderr_sha256": derive.STDERR_WITNESS_SHA, "cli_stderr_bytes": 16,
            "requested_model": derive.MODEL, "expected_response_model": derive.MODEL,
            "generation_parameters": {"max_tokens": 4096, "effort": "high"},
            "transport": "claude_code_cli", "retry_controls": deepcopy(derive.RETRY_CONTROLS),
            "cli_stdout_sha256": "b" * 64, "cli_stdin_sha256": "c" * 64, "cli_events_redacted": events()}


@pytest.fixture
def sample():
    return {"sample_id": "s000", "task_kind": "video_qa", "candidate_group": None,
            "question": "What is visible?", "cutoff_ms": 3000,
            "frames": [{"frame_id": f"v{i:03d}", "timestamp_ms": i * 1000, "image_path": "frame.jpg", "sha256": "d" * 64}
                       for i in range(4)]}


@pytest.fixture
def taxonomy():
    return {"schema_version": 1, "goals": [], "actions": []}


def test_real_high_thinking_then_text_segments_form_one_turn(sample, taxonomy):
    raw = trace()
    before = deepcopy(raw)
    parsed = derive.parse_recorded_turn(raw, sample, taxonomy)
    assert parsed["raw_json"]["answer"] == "cup"
    assert parsed["cli_assistant_segment_count"] == 2 and parsed["cli_num_turns"] == 1
    assert parsed["usage"]["output_tokens"] == 192
    assert raw == before


@pytest.mark.parametrize("bad", ["two_turns", "other_model", "mismatch", "length", "tool", "retry", "unknown_stderr",
                                  "invalid_json", "invalid_schema", "over_budget", "fallback_model", "nonzero_exit"])
def test_unsafe_or_incomplete_streams_are_rejected(sample, taxonomy, bad):
    raw = trace()
    e = raw["cli_events_redacted"]
    if bad == "two_turns":
        e[-1]["num_turns"] = 2
    elif bad == "other_model":
        e[3]["message"]["model"] = "claude-sonnet-other"
    elif bad == "mismatch":
        e[-1]["result"] += " "
    elif bad == "length":
        e[-1]["stop_reason"] = "max_tokens"
    elif bad == "tool":
        e[3]["message"]["content"][0]["type"] = "tool_use"
    elif bad == "retry":
        e[1]["subtype"] = "api_retry"
    elif bad == "unknown_stderr":
        raw["cli_stderr_sha256"] = "0" * 64
    elif bad == "invalid_json":
        raw["cli_events_redacted"] = events("not JSON")
    elif bad == "invalid_schema":
        raw["cli_events_redacted"] = events('{"answer":"cup","evidence_frame_ids":[]}')
    elif bad == "over_budget":
        e[-1]["usage"]["output_tokens"] = 4097
    elif bad == "fallback_model":
        e[-1]["modelUsage"]["claude-haiku-other"] = {}
    elif bad == "nonzero_exit":
        raw["cli_exit_code"] = 1
    with pytest.raises(derive.StreamRejected):
        derive.parse_recorded_turn(raw, sample, taxonomy)


def _json(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def make_run(tmp_path, name, *, fail_code=None):
    root = tmp_path / (name + "-inputs")
    root.mkdir()
    image = root / "frame.jpg"
    image.write_bytes(b"opaque offline test bytes, never sent to a model")
    sample = {"sample_id": "s000", "task_kind": "video_qa", "candidate_group": None,
              "question": "What is visible?", "cutoff_ms": 3000,
              "frames": [{"frame_id": f"v{i:03d}", "timestamp_ms": i * 1000, "image_path": image.name,
                          "sha256": bench._sha(image.read_bytes())} for i in range(4)]}
    manifest = {"schema_version": 1, "benchmark_id": name, "samples": [sample]}
    taxonomy = {"schema_version": 1, "goals": [], "actions": []}
    mp, tp = _json(root / "infer-manifest.json", manifest), _json(root / "taxonomy.json", taxonomy)
    protocol = {"provider_sha256": derive.COLLECTOR_SHA, "source_manifest_sha256": bench._sha(mp.read_bytes()),
                "source_taxonomy_sha256": bench._sha(tp.read_bytes())}
    pp = _json(root / "protocol.json", protocol)
    class Provider:
        synthetic = False
        model = expected_response_model = derive.MODEL
        base_url = "cli://claude-cli/" + derive.MODEL
        generation_parameters = {"max_tokens": 4096, "effort": "high"}
        timeout = 180
        prompt = bench.PROMPT
        def __init__(self):
            self.traces = []
        def predict(self, sample, taxonomy):
            item = trace()
            text = item["cli_events_redacted"][-1]["result"]
            item.update(status="error" if fail_code else "success", returned_model=self.model, finish_reason="stop",
                        raw_json_text=text, raw_json=bench._strict_response_json(text), task_kind=sample["task_kind"],
                        image_sha256={f["frame_id"]: f["sha256"] for f in sample["frames"]}, latency_ms=10,
                        system_prompt_sha256=bench._sha(self.prompt.encode()),
                        payload_sha256=bench._sha(bench._json_text(bench.visible_payload(sample, taxonomy)).encode()),
                        timeout_seconds=180)
            if fail_code:
                item["error_code"] = fail_code
            self.traces.append(item)
            if fail_code:
                raise RuntimeError("offline collector failure")
            return item["raw_json"]
    output = tmp_path / name
    parallel.infer_parallel(mp, tp, output, Provider, protocol_path=pp)
    return output


def test_derivation_preserves_source_and_binds_exact_output(tmp_path):
    source = make_run(tmp_path, "source", fail_code="cli_unexpected_system_event")
    witness = make_run(tmp_path, "witness")
    before = derive._tree_hashes(source)
    output = tmp_path / "derived"
    meta = derive.derive_run(source, witness, output)
    assert derive._tree_hashes(source) == before
    assert meta["collector_corrected_samples"] == 1 and meta["successful_samples"] == 1
    assert meta["analysis_new_model_calls"] == 0 and meta["api_calls"] == 1
    assert derive.validate_derived_run(output)["source_lineage_verified"]
    (source / "added-after-freeze.txt").write_text("mutation")
    with pytest.raises(ValueError, match="Immutable source"):
        derive.validate_derived_run(output)


@pytest.mark.parametrize("error", ["http_error", "cli_timeout", "cli_nonzero_exit", "invalid_complete_prediction"])
def test_noncollector_errors_never_upgraded_even_with_valid_retained_text(tmp_path, error):
    source = make_run(tmp_path, "source", fail_code=error)
    witness = make_run(tmp_path, "witness")
    output = tmp_path / "derived"
    meta = derive.derive_run(source, witness, output)
    assert meta["collector_corrected_samples"] == 0 and meta["failed_samples"] == 1
    assert derive._records(output / "api-calls.jsonl")[0]["error_code"] == error


def test_unfinalized_source_is_refused(tmp_path):
    source = make_run(tmp_path, "source")
    meta, _ = bench._read(source / "run.json")
    meta["status"] = "running"
    _json(source / "run.json", meta)
    with pytest.raises(ValueError, match="finalized"):
        derive.derive_run(source, tmp_path / "missing-witness", tmp_path / "derived")
