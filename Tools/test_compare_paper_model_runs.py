"""Independent cross-machine model comparison and trace tamper regressions."""

from copy import deepcopy
import hashlib
import json

import pytest

import compare_paper_model_runs as compare
import satori_paper_benchmark as bench


def write(path, value, *, crlf=False):
    raw = (json.dumps(value, indent=2) + "\n").encode()
    path.write_bytes(raw.replace(b"\n", b"\r\n") if crlf else raw)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def frozen(tmp_path):
    data = tmp_path / "data"
    run = tmp_path / "run"
    data.mkdir()
    run.mkdir()
    image = data / "frame.jpg"
    image.write_bytes(b"frozen fake image; no model called")
    frames = [{"frame_id": f"v{i:03d}", "timestamp_ms": i * 1000,
               "image_path": image.name, "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}
              for i in range(4)]
    samples = [{"sample_id": f"s{i:03d}", "task_kind": "video_qa", "question": "What?",
                "candidate_group": None, "cutoff_ms": 3000, "frames": deepcopy(frames)} for i in range(4)]
    manifest = {"schema_version": 1, "benchmark_id": "unit", "samples": samples}
    taxonomy = {"schema_version": 1, "goals": [], "actions": []}
    gt = {"schema_version": 1, "benchmark_id": "unit", "provenance": {"unit": True}, "samples": [
        {"sample_id": s["sample_id"], "answers": ["cup"], "metadata": {"category": ["world"]}}
        for s in samples]}
    protocol = {"schema_version": 1, "source_manifest_sha256": write(data / "infer-manifest.json", manifest, crlf=True),
                "source_taxonomy_sha256": write(data / "taxonomy.json", taxonomy, crlf=True),
                "source_gt_sha256": write(data / "gt.json", gt, crlf=True)}
    write(data / "protocol.json", protocol, crlf=True)
    meta = {"schema_version": 1, "synthetic": False, "status": "aborted", "benchmark_id": "unit",
            "model": "test-model", "expected_response_model": "test-model", "endpoint": "https://example.invalid/v1",
            "generation_parameters": {"max_tokens": 4096, "temperature": 0.0}, "timeout_seconds": 180,
            "prompt_sha256": hashlib.sha256(bench.PROMPT.encode()).hexdigest(), "automatic_retry": False,
            "planned_samples": 4, "processed_samples": 3, "successful_samples": 2,
            "failed_samples": 1, "not_run_samples": 1, "api_calls": 3}
    for name, source, value in [("manifest", "infer-manifest.json", manifest), ("taxonomy", "taxonomy.json", taxonomy),
                                ("protocol", "protocol.json", protocol)]:
        meta[name + "_snapshot_sha256"] = write(run / (name + ".json"), value)
        meta["source_" + name + "_sha256"] = hashlib.sha256((data / source).read_bytes()).hexdigest()
        meta["source_" + name + "_path"] = "/remote/not-on-this-machine/" + source
    traces, predictions = [], []
    for i, answer in enumerate(["The cup.", None, None]):
        sample = samples[i]
        value = {"answer": answer, "evidence_frame_ids": ["v000"] if i != 2 else []}
        trace = {"sample_id": sample["sample_id"], "status": "success" if i < 2 else "error",
                 "task_kind": sample["task_kind"], "requested_model": "test-model", "returned_model": "test-model",
                 "expected_response_model": "test-model", "finish_reason": "stop", "raw_json": value,
                 "raw_json_text": json.dumps(value), "generation_parameters": meta["generation_parameters"],
                 "timeout_seconds": 180, "latency_ms": 10,
                 "image_sha256": {f["frame_id"]: f["sha256"] for f in frames},
                 "system_prompt_sha256": meta["prompt_sha256"],
                 "payload_sha256": hashlib.sha256(bench._json_text(bench.visible_payload(sample, taxonomy)).encode()).hexdigest()}
        if i == 2:
            trace["error_code"] = "invalid_complete_prediction"
        traces.append(trace)
        predictions.append({"sample_id": sample["sample_id"], "status": "completed" if i < 2 else "failed",
                            "response": value if i < 2 else None, "error_code": None if i < 2 else "RuntimeError",
                            "request_attempted": True, "latency_ms": 10})
    write(run / "run.json", meta)
    (run / "api-calls.jsonl").write_text("".join(json.dumps(t) + "\n" for t in traces), encoding="utf-8")
    (run / "predictions.jsonl").write_text("".join(json.dumps(t) + "\n" for t in predictions), encoding="utf-8")
    # A forged metrics file must never influence the independently recomputed score.
    write(run / "metrics.json", {"accuracy": 1.0})
    return data, run


def mutate_trace(run, callback):
    traces = [json.loads(l) for l in (run / "api-calls.jsonl").read_text().splitlines()]
    callback(traces)
    (run / "api-calls.jsonl").write_text("".join(json.dumps(t) + "\n" for t in traces), encoding="utf-8")


def test_cross_platform_sources_and_fixed_denominator(frozen):
    data, run = frozen
    result = compare.audit_and_score(run, data)
    assert result["audit"]["passed"] is True
    assert result["planned_samples"] == 4
    assert result["correct"] == 1 and result["accuracy"] == .25
    assert result["strict_exact_correct"] == 0
    assert result["valid_nonnull_answers"] == 1
    assert result["valid_null_abstentions"] == 1
    assert result["raw_null_answers_any_status"] == 2
    assert result["raw_null_failed_schema"] == 1
    assert result["failed_samples"] == 1 and result["not_run_samples"] == 1
    assert result["by_task"]["video_qa"]["samples"] == 4


@pytest.mark.parametrize("accounting", [False, None, "missing"])
def test_incomplete_parallel_request_accounting_rejected(frozen, accounting):
    data, run = frozen
    meta = json.loads((run / "run.json").read_text())
    meta["execution"] = {"runner": "run_paper_parallel"}
    if accounting != "missing":
        meta["request_accounting_complete"] = accounting
    write(run / "run.json", meta)
    with pytest.raises(ValueError, match="accounting"):
        compare.audit_and_score(run, data)


@pytest.mark.parametrize("field,value", [
    ("requested_model", "wrong"), ("returned_model", "wrong"), ("expected_response_model", "wrong"),
    ("system_prompt_sha256", "wrong"), ("payload_sha256", "wrong"),
    ("image_sha256", {}), ("generation_parameters", {"max_tokens": 10, "temperature": 0}),
    ("timeout_seconds", 1), ("finish_reason", "length"), ("latency_ms", -1),
    ("raw_json_text", '{"answer":"plate","evidence_frame_ids":["v000"]}'),
])
def test_trace_tampering_rejected(frozen, field, value):
    data, run = frozen
    mutate_trace(run, lambda traces: traces[0].__setitem__(field, value))
    with pytest.raises(ValueError):
        compare.audit_and_score(run, data)


def test_duplicate_requests_rejected(frozen):
    data, run = frozen
    mutate_trace(run, lambda traces: traces.append(deepcopy(traces[0])))
    with pytest.raises(ValueError, match="unique"):
        compare.audit_and_score(run, data)


def test_image_change_rejected(frozen):
    data, run = frozen
    (data / "frame.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="image"):
        compare.audit_and_score(run, data)


def test_snapshot_semantic_change_even_with_rehashed_metadata_rejected(frozen):
    data, run = frozen
    value = json.loads((run / "manifest.json").read_text())
    value["samples"][0]["question"] = "changed after inference"
    digest = write(run / "manifest.json", value)
    meta = json.loads((run / "run.json").read_text())
    meta["manifest_snapshot_sha256"] = digest
    write(run / "run.json", meta)
    with pytest.raises(ValueError, match="semantic"):
        compare.audit_and_score(run, data)


def test_gt_change_rejected(frozen):
    data, run = frozen
    gt = json.loads((data / "gt.json").read_text())
    gt["samples"][0]["answers"] = ["plate"]
    write(data / "gt.json", gt)
    with pytest.raises(ValueError, match="GT"):
        compare.audit_and_score(run, data)


def test_comparison_rejects_mismatched_input_fingerprints(frozen):
    data, run = frozen
    a = compare.audit_and_score(run, data)
    b = deepcopy(a)
    b["input_fingerprint"]["manifest_sha256"] = "different"
    with pytest.raises(ValueError, match="same frozen"):
        compare.compare_results([a, b])


def test_pairing_counts_each_frozen_sample_once(frozen):
    data, run = frozen
    a = compare.audit_and_score(run, data)
    b = deepcopy(a)
    b["model"] = "second-model"
    result = compare.compare_results([a, b])
    assert len(result["paired"]) == 4
    assert result["planned_samples"] == 4
    assert len(result["models"]) == 2


def test_external_protocol_requires_explicit_mapping_and_same_frozen_bytes(frozen):
    data, run = frozen
    external = data.parent / "new-protocol-source.json"
    (data / "protocol.json").replace(external)
    with pytest.raises(FileNotFoundError):
        compare.audit_and_score(run, data)
    result = compare.audit_and_score(run, data, protocol_source=external)
    assert result["audit"]["passed"]
    external.write_bytes(external.read_bytes() + b" ")
    with pytest.raises(ValueError, match="source bytes"):
        compare.audit_and_score(run, data, protocol_source=external)


def test_failed_raw_null_distinguishes_schema_from_other_response_failures(frozen):
    data, run = frozen
    def change(traces):
        value = {"answer": None, "evidence_frame_ids": ["v000"]}
        traces[2].update(raw_json=value, raw_json_text=json.dumps(value), finish_reason="length")
    mutate_trace(run, change)
    result = compare.audit_and_score(run, data)
    assert result["raw_null_failed_schema"] == 0
    assert result["raw_null_failed_response"] == 1
    assert result["failed_samples"] == 1
