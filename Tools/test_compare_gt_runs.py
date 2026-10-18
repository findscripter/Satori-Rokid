"""Negative audit cases: inconsistent evidence must never become silent exclusion."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_gt_runs as comparison
import gt_benchmark as benchmark


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_lines(path, values):
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values), encoding="utf-8")


@pytest.fixture
def evidence(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "frame.jpg").write_bytes(b"test fixture bytes; audit checks hashes, not visual content")
    image_hash = comparison.digest(data / "frame.jpg")
    taxonomy = {"schema_version": 1,
                "scenes": [{"id": f"s{i:03d}", "label": label} for i, label in
                           enumerate(("Bedroom", "Kitchen", "Living room", "Study"))],
                "actions": [{"id": f"c{i:03d}", "label": f"Action {i}"} for i in range(157)]}
    manifest = {"schema_version": 1, "benchmark_id": "comparison-test", "samples": [
        {"sample_id": f"e{index:03d}", "frames": [
            {"frame_id": f"v{frame:03d}", "timestamp_ms": frame * 100,
             "image_path": "frame.jpg", "sha256": image_hash} for frame in range(8)]} for index in range(32)]}
    truth = {"schema_version": 1, "benchmark_id": "comparison-test", "provenance": {}, "labels": [
        {"sample_id": sample["sample_id"], "source_video_id": f"source-{index}",
         "scene_id": "s000", "action_ids": ["c000"]} for index, sample in enumerate(manifest["samples"])]}
    for name, value in (("infer-manifest.json", manifest), ("taxonomy.json", taxonomy), ("gt.json", truth),
                        ("baseline-protocol.json", {"scene_label": "Bedroom", "action_ids": ["c000"]})):
        write(data / name, value)
    runs = []
    for model in ("Mage-VL", "gpt-test"):
        run = tmp_path / model
        run.mkdir()
        write(run / "manifest.json", manifest)
        write(run / "taxonomy.json", taxonomy)
        meta = {"status": "completed", "synthetic": False, "benchmark_id": manifest["benchmark_id"],
                "protocol": "closed-set eight-sparse-frame video-level scene and multilabel action classification",
                "planned_samples": 32, "processed_samples": 32, "api_calls": 32, "not_run_samples": 0,
                "successful_samples": 32, "failed_samples": 0, "automatic_retry": False, "frame_count": 8,
                "generation_parameters": deepcopy(comparison.EXPECTED_GENERATION), "timeout_seconds": 180,
                "prompt_sha256": hashlib.sha256(benchmark.PROMPT.encode()).hexdigest(),
                "source_manifest_sha256": comparison.digest(data / "infer-manifest.json"),
                "source_taxonomy_sha256": comparison.digest(data / "taxonomy.json"),
                "manifest_snapshot_sha256": comparison.digest(run / "manifest.json"),
                "taxonomy_snapshot_sha256": comparison.digest(run / "taxonomy.json"),
                "model": model, "expected_response_model": model, "endpoint": "fixture"}
        write(run / "run.json", meta)
        predictions, traces = [], []
        for sample in manifest["samples"]:
            answer = {"scene_id": "s000", "action_ids": ["c000"], "evidence_frame_ids": ["v000"], "answer": "测试观察。"}
            predictions.append({"sample_id": sample["sample_id"], "status": "completed", "response": answer,
                                "error_code": None, "request_attempted": True, "latency_ms": 100.0})
            visible = {"scene_candidates": taxonomy["scenes"], "action_candidates": taxonomy["actions"],
                       "frames": [{"frame_id": frame["frame_id"], "timestamp_ms": frame["timestamp_ms"]}
                                  for frame in sample["frames"]]}
            traces.append({"sample_id": sample["sample_id"], "stage": "charades_ego_classification", "status": "success",
                           "requested_model": model, "expected_response_model": model, "returned_model": model,
                           "generation_parameters": deepcopy(meta["generation_parameters"]), "timeout_seconds": 180,
                           "system_prompt_sha256": meta["prompt_sha256"],
                           "image_ids": [frame["frame_id"] for frame in sample["frames"]],
                           "image_sha256": {frame["frame_id"]: frame["sha256"] for frame in sample["frames"]},
                           "payload_sha256": hashlib.sha256(benchmark._json_text(visible).encode()).hexdigest(),
                           "latency_ms": 100.0, "response_model_matches": True, "finish_reason": "stop",
                           "response_refused": False, "raw_json": answer, "raw_json_text": json.dumps(answer),
                           "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}})
        write_lines(run / "predictions.jsonl", predictions)
        write_lines(run / "api-calls.jsonl", traces)
        runs.append(run)
    return runs[0], runs[1], data


def test_paired_complete_comparison_and_sidecars(evidence, tmp_path):
    result = comparison.compare(*evidence)
    assert result["audit_passed"] is True
    assert result["mage"]["metrics"]["scene_top1"] == 1
    assert result["gpt_minus_mage"]["action_micro_f1"] == 0
    assert len(result["paired_samples"]) == 32
    output = tmp_path / "comparison.json"
    assert len(comparison.save(result, output)) == 4
    with pytest.raises(ValueError, match="already exists"):
        comparison.save(result, output)


def test_failed_request_is_retained_with_all_false_negatives(evidence):
    mage, gpt, data = evidence
    rows = comparison.records(gpt / "predictions.jsonl")
    rows[0].update(status="failed", response=None, error_code="RuntimeError")
    write_lines(gpt / "predictions.jsonl", rows)
    traces = comparison.records(gpt / "api-calls.jsonl")
    traces[0].update(status="error", returned_model=None, error_code="connection_error", usage=None)
    for field in ("raw_json", "raw_json_text", "response_model_matches", "finish_reason", "response_refused"):
        traces[0].pop(field)
    write_lines(gpt / "api-calls.jsonl", traces)
    meta = comparison.read(gpt / "run.json")
    meta.update(successful_samples=31, failed_samples=1)
    write(gpt / "run.json", meta)
    result = comparison.compare(mage, gpt, data)
    assert result["gpt"]["metrics"]["scene_top1"] == 31 / 32
    assert result["gpt"]["metrics"]["action_fn"] == 1
    assert result["gpt"]["usage"]["response_envelope"]["total_tokens"]["missing_samples"] == 1
    assert result["paired_samples"][0]["gpt"]["fn"] == 1


@pytest.mark.parametrize("field,value,match", [
    ("returned_model", "unexpected-model", "actual response model"),
    ("timeout_seconds", 200, "timeout changed"),
    ("generation_parameters", {"max_tokens": 1024}, "generation parameters changed"),
    ("payload_sha256", "0" * 64, "visible context changed"),
    ("raw_json_text", "{}", "Raw JSON text differs"),
])
def test_reject_inconsistent_trace(evidence, field, value, match):
    _, gpt, _ = evidence
    traces = comparison.records(gpt / "api-calls.jsonl")
    traces[0][field] = value
    write_lines(gpt / "api-calls.jsonl", traces)
    with pytest.raises(ValueError, match=match):
        comparison.compare(*evidence)


@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_reject_missing_or_repeated_request(evidence, change):
    mage, _, _ = evidence
    traces = comparison.records(mage / "api-calls.jsonl")
    traces = traces[:-1] if change == "missing" else traces + [traces[0]]
    write_lines(mage / "api-calls.jsonl", traces)
    with pytest.raises(ValueError, match="precisely one ordered request trace"):
        comparison.compare(*evidence)


def test_reject_mutated_image_bytes(evidence):
    (evidence[2] / "frame.jpg").write_bytes(b"changed after inference")
    with pytest.raises(ValueError, match="Source image bytes changed"):
        comparison.compare(*evidence)


def test_reject_changed_saved_scores(evidence):
    write(evidence[0] / "scores.json", {"not": "the recomputed scores"})
    with pytest.raises(ValueError, match="Saved scores do not match"):
        comparison.compare(*evidence)


def test_reject_unfrozen_output_budget(evidence):
    path = evidence[1] / "run.json"
    meta = comparison.read(path)
    meta["generation_parameters"]["max_tokens"] = 8192
    write(path, meta)
    with pytest.raises(ValueError, match="Generation parameters differ"):
        comparison.compare(*evidence)


def mark_identity_failure(run):
    rows = comparison.records(run / "predictions.jsonl")
    rows[0].update(status="failed", response=None, error_code="RuntimeError")
    write_lines(run / "predictions.jsonl", rows)
    traces = comparison.records(run / "api-calls.jsonl")
    traces[0].update(status="error", returned_model="unexpected-model", error_code="invalid_complete_prediction",
                     response_model_matches=False, response_issues=["unexpected_model"])
    traces[0].pop("raw_json")
    traces[0].pop("raw_json_text")
    write_lines(run / "api-calls.jsonl", traces)
    meta = comparison.read(run / "run.json")
    meta.update(successful_samples=31, failed_samples=1)
    write(run / "run.json", meta)


def test_diagnosed_identity_failure_remains_in_fixed_denominator(evidence):
    mark_identity_failure(evidence[1])
    result = comparison.compare(*evidence)
    assert result["gpt"]["failed_samples"] == 1
    assert result["gpt"]["metrics"]["action_fn"] == 1
    assert result["gpt"]["metrics"]["scene_top1"] == 31 / 32
    assert result["gpt"]["identity_anomalies"] == [{
        "sample_id": "e000", "expected_response_model": "gpt-test", "returned_model": "unexpected-model",
        "error_code": "invalid_complete_prediction", "response_issues": ["unexpected_model"], "scored_status": "failed"}]
    assert result["paired_samples"][0]["gpt"]["returned_model"] == "unexpected-model"


@pytest.mark.parametrize("field,value", [("error_code", "connection_error"),
                                       ("response_issues", []), ("response_model_matches", True)])
def test_reject_undiagnosed_identity_failure(evidence, field, value):
    mark_identity_failure(evidence[1])
    path = evidence[1] / "api-calls.jsonl"
    traces = comparison.records(path)
    traces[0][field] = value
    write_lines(path, traces)
    with pytest.raises(ValueError, match="identity diagnostics"):
        comparison.compare(*evidence)


def make_freeze(evidence, tmp_path):
    mage, gpt, data = evidence
    for run in (mage, gpt):
        meta = comparison.read(run / "run.json")
        meta["started_utc"] = "2026-10-16T13:30:00+00:00"
        write(run / "run.json", meta)
    protocol = tmp_path / "MAGE-GPT-COMPARISON-PROTOCOL-20261016.md"
    protocol.write_text("冻结的单次调用对照协议。\n", encoding="utf-8")
    root = Path(comparison.__file__).resolve().parents[1]
    value = {"frozen_utc": "2026-10-16T13:26:00+00:00", "stage": "before_formal_inference",
             "planned_samples_per_model": 32, "frames_per_sample": 8, "attempts_per_sample": 1,
             "generation_parameters": comparison.EXPECTED_GENERATION, "timeout_seconds": 180,
             "automatic_retry": False,
             "data_sha256": {name: comparison.digest(data / name) for name in
                             ("infer-manifest.json", "taxonomy.json", "gt.json", "baseline-protocol.json")},
             "code_sha256": {name: comparison.digest(root / name) for name in
                             ("Tools/gt_benchmark.py", "Tools/run_mage_gt.py", "Tools/run_gpt_gt.py")},
             "prompt_sha256": hashlib.sha256(benchmark.PROMPT.encode()).hexdigest(),
             "protocol_sha256": comparison.digest(protocol),
             "formal_runs": {"mage": str(mage), "gpt": str(gpt)},
             "smokes": {"mage": {"returned_model": "Mage-VL"}, "gpt": {"returned_model": "gpt-test"}}}
    path = tmp_path / "mage-gpt-comparison-freeze-20261016.json"
    write(path, value)
    return path, value, protocol


def test_verify_freeze_contract(evidence, tmp_path):
    path, _, _ = make_freeze(evidence, tmp_path)
    result = comparison.compare(*evidence, freeze=path)
    assert result["freeze_verified"] is True
    assert result["freeze_evidence"]["sha256"] == comparison.digest(path)


@pytest.mark.parametrize("mutation,match", [("code", "Frozen inference code changed"),
                                          ("data", "Frozen data changed"),
                                          ("model", "Exact model ID differs"),
                                          ("run", "Formal run path differs"),
                                          ("protocol", "Frozen written protocol changed")])
def test_reject_changed_freeze_contract(evidence, tmp_path, mutation, match):
    path, value, protocol = make_freeze(evidence, tmp_path)
    if mutation == "code":
        value["code_sha256"]["Tools/gt_benchmark.py"] = "0" * 64
    elif mutation == "data":
        value["data_sha256"]["gt.json"] = "0" * 64
    elif mutation == "model":
        value["smokes"]["gpt"]["returned_model"] = "another-gpt"
    elif mutation == "run":
        value["formal_runs"]["gpt"] = "Research/Backend/runs/another-run"
    else:
        protocol.write_text("changed", encoding="utf-8")
    write(path, value)
    with pytest.raises(ValueError, match=match):
        comparison.compare(*evidence, freeze=path)
