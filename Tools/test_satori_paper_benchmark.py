"""Contract regressions for independent Satori-paper-data module probes."""

from copy import deepcopy
import hashlib
import json
import base64
from urllib import error

import pytest

import satori_paper_benchmark as bench


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def data(tmp_path):
    raw = b"frozen image bytes; fake provider only"
    image = tmp_path / "frame.png"
    image.write_bytes(raw)
    frames = [{"frame_id": f"v{i:03d}", "timestamp_ms": i * 1000,
               "image_path": image.name, "sha256": hashlib.sha256(raw).hexdigest()}
              for i in range(4)]
    sample = {"sample_id": "s000", "task_kind": "video_qa", "question": "What is on the table?",
              "cutoff_ms": 3000, "candidate_group": None, "frames": frames}
    manifest = {"schema_version": 1, "benchmark_id": "unit", "samples": [sample]}
    taxonomy = {"schema_version": 1, "goals": [{"id": "g0", "label": "make tea"}],
                "actions": [{"id": "a0", "label": "open jar"}, {"id": "a1", "label": "pour water"}]}
    protocol = {"protocol_id": "unit-v1", "scope": "module probe", "limitations": ["not paper reproduction"]}
    gt = {"schema_version": 1, "benchmark_id": "unit", "samples": [
        {"sample_id": "s000", "answers": ["a cup"], "metadata": {"subject": "S1"}}],
        "provenance": {"source": "unit-test"}}
    return {"root": tmp_path, "manifest": manifest, "taxonomy": taxonomy, "protocol": protocol, "gt": gt}


class FakeProvider:
    model = "fake"
    base_url = "https://example.invalid/v1"
    prompt = "test"
    synthetic = True
    generation_parameters = {"max_tokens": 20, "temperature": 0}
    timeout = 1

    def __init__(self, answers=None, fail=False):
        self.traces = []
        self.calls = []
        self.answers = answers or ["Cup."]
        self.fail = fail

    def predict(self, sample, taxonomy):
        self.calls.append((deepcopy(sample), deepcopy(taxonomy)))
        should_fail = bool(self.fail) if isinstance(self.fail, bool) else (len(self.calls) - 1) in self.fail
        value = {"answer": self.answers[len(self.calls) - 1] if not should_fail else None, "evidence_frame_ids": ["v000"]}
        self.traces.append({"status": "error" if should_fail else "success", "latency_ms": 10,
                            "requested_model": self.model, "returned_model": self.model,
                            "expected_response_model": self.model,
                            "image_sha256": {frame["frame_id"]: frame["sha256"] for frame in sample["frames"]},
                            "system_prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest(),
                            "payload_sha256": hashlib.sha256(bench._json_text(bench.visible_payload(sample, taxonomy)).encode()).hexdigest(),
                            "generation_parameters": self.generation_parameters, "timeout_seconds": self.timeout,
                            "raw_json": value if not should_fail else None})
        if should_fail:
            raise RuntimeError("no retry")
        return value


def run(data, provider, progress=None):
    root = data["root"]
    return bench.infer(write_json(root / "manifest-source.json", data["manifest"]),
                       write_json(root / "taxonomy-source.json", data["taxonomy"]),
                       root / "run", provider,
                       protocol_path=write_json(root / "protocol-source.json", data["protocol"]), progress=progress)


def test_normalization_is_literal_not_semantic():
    assert bench.normalize_answer("  The RED, cup! ") == "red cup"
    assert bench.normalize_answer("isn't") == "isnt"
    assert bench.normalize_answer("two") != bench.normalize_answer("2")
    assert bench.normalize_answer("cup and plate") != bench.normalize_answer("plate and cup")


def test_manifest_rejects_hidden_gt_and_future_frames(data):
    altered = deepcopy(data["manifest"])
    altered["samples"][0]["answer"] = "a cup"
    with pytest.raises(ValueError):
        bench.validate_manifest(altered, data["taxonomy"])
    altered = deepcopy(data["manifest"])
    altered["samples"][0]["cutoff_ms"] = 2999
    with pytest.raises(ValueError):
        bench.validate_manifest(altered, data["taxonomy"])


def test_manifest_requires_fixed_frame_counts_unique_ids_and_closed_set(data):
    sample = data["manifest"]["samples"][0]
    sample["frames"].pop()
    with pytest.raises(ValueError):
        bench.validate_manifest(data["manifest"], data["taxonomy"])
    sample["frames"].append(deepcopy(sample["frames"][-1]))
    with pytest.raises(ValueError):
        bench.validate_manifest(data["manifest"], data["taxonomy"])


def test_class_predictions_require_exact_candidate_ids(data):
    sample = data["manifest"]["samples"][0]
    sample.update(task_kind="action_forecast", candidate_group="actions")
    for invalid in ("open jar", "A0", ["a0"], "undefined"):
        with pytest.raises(ValueError):
            bench.validate_prediction({"answer": invalid, "evidence_frame_ids": ["v000"]}, sample, data["taxonomy"])
    assert bench.validate_prediction({"answer": "a0", "evidence_frame_ids": ["v000"]}, sample, data["taxonomy"])["answer"] == "a0"


def test_predictions_reject_empty_or_invented_evidence(data):
    for evidence in ([], ["v999"], ["v000", "v000"]):
        with pytest.raises(ValueError):
            bench.validate_prediction({"answer": "cup", "evidence_frame_ids": evidence},
                                      data["manifest"]["samples"][0], data["taxonomy"])


def test_inference_has_no_gt_parameter_and_one_attempt(data):
    provider = FakeProvider(fail=True)
    meta = run(data, provider)
    assert meta["failed_samples"] == 1
    assert meta["api_calls"] == 1
    assert len(provider.calls) == 1
    assert provider.calls[0][0].keys() == data["manifest"]["samples"][0].keys()
    assert "metadata" not in provider.calls[0][0]
    assert meta["synthetic"] is True


def test_changed_input_fails_without_network(data):
    (data["root"] / "frame.png").write_bytes(b"changed")
    provider = FakeProvider()
    meta = run(data, provider)
    assert meta["failed_samples"] == 1 and meta["api_calls"] == 0
    assert not provider.calls


def test_scoring_fixed_denominator_includes_failed_and_not_run(data):
    for i in (1, 2):
        sample = deepcopy(data["manifest"]["samples"][0])
        sample["sample_id"] = f"s{i:03d}"
        data["manifest"]["samples"].append(sample)
        data["gt"]["samples"].append({"sample_id": sample["sample_id"], "answers": ["cup"], "metadata": {}})
    def interrupt_after_two(row):
        if row["processed"] == 2:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(data, FakeProvider(["cup", "plate", "spoon"], fail={1}), progress=interrupt_after_two)
    result = bench.score(data["root"] / "run", write_json(data["root"] / "gt.json", data["gt"]))
    assert result["planned_samples"] == 3
    assert result["correct"] == 1
    assert result["accuracy"] == pytest.approx(1 / 3)
    assert result["failed_samples"] == 1 and result["not_run_samples"] == 1


def test_score_checks_snapshot_and_exact_gt_alignment(data):
    run(data, FakeProvider())
    gt_path = write_json(data["root"] / "gt.json", data["gt"])
    result = bench.score(data["root"] / "run", gt_path)
    assert result["accuracy"] == 1
    data["gt"]["samples"][0]["sample_id"] = "other"
    write_json(gt_path, data["gt"])
    with pytest.raises(ValueError):
        bench.score(data["root"] / "run", gt_path)
    write_json(data["root"] / "run" / "protocol.json", {"changed": True})
    with pytest.raises(ValueError):
        bench.score(data["root"] / "run", gt_path)


def test_abstention_counts_as_wrong(data):
    run(data, FakeProvider([None]))
    result = bench.score(data["root"] / "run", write_json(data["root"] / "gt.json", data["gt"]))
    assert result["accuracy"] == 0 and result["abstained_samples"] == 1


def test_action_gt_rejects_target_frame_leakage(data):
    sample = data["manifest"]["samples"][0]
    sample.update(task_kind="action_forecast", candidate_group="actions")
    data["gt"]["samples"][0].update(answers=["a0"], metadata={"target_onset_ms": 3000, "look_ahead_ms": 0})
    run(data, FakeProvider(["a0"]))
    with pytest.raises(ValueError, match="onset"):
        bench.score(data["root"] / "run", write_json(data["root"] / "gt.json", data["gt"]))


def test_classification_uses_literal_ids_and_macro_recall(data):
    sample = data["manifest"]["samples"][0]
    sample.update(task_kind="action_forecast", candidate_group="actions")
    data["gt"]["samples"][0].update(answers=["a0"], metadata={"target_onset_ms": 4000, "look_ahead_ms": 1000, "subject": "S1"})
    run(data, FakeProvider(["a0"]))
    result = bench.score(data["root"] / "run", write_json(data["root"] / "gt.json", data["gt"]))
    assert result["by_task"]["action_forecast"]["balanced_accuracy_present_classes"] == 1
    assert result["by_look_ahead_ms"]["1000"]["correct"] == 1


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit):
        return self.payload[:limit]


def make_transport(monkeypatch, data, *, finish="stop", returned_model="unit", content=None):
    captured = []
    payload = {"model": returned_model, "usage": {"total_tokens": 10}, "choices": [
        {"finish_reason": finish, "message": {"content": json.dumps({"answer": "cup", "evidence_frame_ids": ["v000"]}) if content is None else content}}]}
    raw = (data["root"] / "frame.png").read_bytes()
    monkeypatch.setattr(bench, "_image_data_url", lambda _: "data:image/png;base64," + base64.b64encode(raw).decode())
    monkeypatch.setenv("UNIT_TEST_CREDENTIAL", "unit-only-not-a-real-key")

    class Opener:
        def open(self, req, timeout):
            captured.append((req, timeout))
            return FakeResponse(payload)

    monkeypatch.setattr(bench.request, "build_opener", lambda *args: Opener())
    provider = bench.PaperBenchmarkProvider(model="unit", base_url="https://example.invalid/v1",
                                            api_key_env="UNIT_TEST_CREDENTIAL")
    sample = deepcopy(data["manifest"]["samples"][0])
    for frame in sample["frames"]:
        frame["image_path"] = str(data["root"] / frame["image_path"])
    return provider, sample, captured


def test_http_sends_only_question_candidate_labels_and_causal_images(monkeypatch, data):
    provider, sample, captured = make_transport(monkeypatch, data)
    result = provider.predict(sample, data["taxonomy"])
    assert result["answer"] == "cup"
    assert len(captured) == len(provider.traces) == 1
    req, timeout = captured[0]
    body = json.loads(req.data)
    assert body["max_tokens"] == 4096 and body["temperature"] == 0
    assert timeout == 180
    visible = json.loads(body["messages"][1]["content"][0]["text"])
    assert visible == {"task_kind": "video_qa", "question": sample["question"], "cutoff_ms": 3000, "candidates": None,
                       "frames": [{"frame_id": f"v{i:03d}", "timestamp_ms": i * 1000} for i in range(4)]}
    assert "frame.png" not in req.data.decode()
    assert "metadata" not in req.data.decode()
    assert "ground_truth" not in req.data.decode()
    assert provider.traces[0]["status"] == "success"
    assert "unit-only-not-a-real-key" not in json.dumps(provider.traces)


@pytest.mark.parametrize("finish,returned,content", [
    ("length", "unit", None),
    ("stop", "other", None),
    ("stop", "unit", "```json\n{}\n```"),
    ("stop", "unit", '{"answer":"cup","evidence_frame_ids":["v000"],"explanation":"extra"}'),
    ("stop", "unit", '{"answer":"unit-only-not-a-real-key","evidence_frame_ids":["v000"]}'),
    ("stop", "unit", '{"answer":"plate","answer":"cup","evidence_frame_ids":["v000"]}'),
])
def test_http_rejects_truncation_model_mismatch_bad_json_or_secret(monkeypatch, data, finish, returned, content):
    provider, sample, captured = make_transport(monkeypatch, data, finish=finish, returned_model=returned, content=content)
    with pytest.raises(RuntimeError):
        provider.predict(sample, data["taxonomy"])
    assert len(captured) == len(provider.traces) == 1
    assert provider.traces[0]["status"] == "error"
    assert "unit-only-not-a-real-key" not in json.dumps(provider.traces)


def test_http_error_not_retried_and_body_not_logged(monkeypatch, data):
    provider, sample, captured = make_transport(monkeypatch, data)

    class FailedOpener:
        def open(self, req, timeout):
            captured.append((req, timeout))
            raise error.HTTPError(req.full_url, 502, "upstream private error", {}, None)

    monkeypatch.setattr(bench.request, "build_opener", lambda *args: FailedOpener())
    with pytest.raises(RuntimeError):
        provider.predict(sample, data["taxonomy"])
    assert len(captured) == len(provider.traces) == 1
    assert provider.traces[0]["http_status"] == 502
    assert "private error" not in json.dumps(provider.traces)


def test_pending_missing_video_manifest_is_rejected_before_requests(data):
    data["manifest"]["samples"][0]["frames"] = []
    provider = FakeProvider()
    with pytest.raises(ValueError):
        run(data, provider)
    assert not provider.calls
    assert not (data["root"] / "run").exists()


def test_qa_reports_strict_em_and_overlapping_official_categories(data):
    data["gt"]["samples"][0]["metadata"].update(category=["world", "multiagent"], reference="indirect")
    run(data, FakeProvider())
    result = bench.score(data["root"] / "run", write_json(data["root"] / "gt.json", data["gt"]))
    assert result["accuracy"] == 1
    assert result["strict_exact_accuracy"] == 0
    assert result["by_category"]["world"]["samples"] == 1
    assert result["by_category"]["multiagent"]["samples"] == 1
    assert result["by_reference"]["indirect"]["samples"] == 1


def test_scoring_rejects_prediction_that_differs_from_api_trace(data):
    run(data, FakeProvider())
    path = data["root"] / "run" / "predictions.jsonl"
    row = json.loads(path.read_text())
    row["response"]["answer"] = "plate"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="trace"):
        bench.score(data["root"] / "run", write_json(data["root"] / "gt.json", data["gt"]))


def test_infer_checks_protocol_source_hash_before_requests(data):
    data["protocol"]["source_manifest_sha256"] = "0" * 64
    provider = FakeProvider()
    with pytest.raises(ValueError, match="Protocol source hash"):
        run(data, provider)
    assert not provider.calls
    assert not (data["root"] / "run").exists()


def test_score_checks_frozen_gt_hash(data):
    gt_path = write_json(data["root"] / "gt.json", data["gt"])
    data["protocol"]["source_gt_sha256"] = hashlib.sha256(gt_path.read_bytes()).hexdigest()
    run(data, FakeProvider())
    assert bench.score(data["root"] / "run", gt_path)["audit"]["passed"]
    data["gt"]["samples"][0]["answers"] = ["plate"]
    write_json(gt_path, data["gt"])
    with pytest.raises(ValueError, match="GT source hash"):
        bench.score(data["root"] / "run", gt_path)


def test_audit_rechecks_source_files_and_actual_image_bytes(data):
    run(data, FakeProvider())
    assert bench.audit_run(data["root"] / "run")["passed"]
    (data["root"] / "frame.png").write_bytes(b"changed after inference")
    with pytest.raises(ValueError, match="image bytes"):
        bench.audit_run(data["root"] / "run")


def test_audit_rejects_duplicate_calls_and_counter_changes(data):
    run(data, FakeProvider())
    trace_path = data["root"] / "run" / "api-calls.jsonl"
    original = trace_path.read_text()
    trace_path.write_text(original + original, encoding="utf-8")
    with pytest.raises(ValueError, match="one call"):
        bench.audit_run(data["root"] / "run")
    trace_path.write_text(original, encoding="utf-8")
    run_path = data["root"] / "run" / "run.json"
    meta = json.loads(run_path.read_text())
    meta["api_calls"] = 2
    write_json(run_path, meta)
    with pytest.raises(ValueError, match="counters"):
        bench.audit_run(data["root"] / "run")


def test_audit_rejects_trace_request_mismatch(data):
    run(data, FakeProvider())
    trace_path = data["root"] / "run" / "api-calls.jsonl"
    trace = json.loads(trace_path.read_text())
    trace["payload_sha256"] = "0" * 64
    trace_path.write_text(json.dumps(trace) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen request"):
        bench.audit_run(data["root"] / "run")


def test_forecast_system_prompt_respects_question_lookahead():
    assert "look-ahead offset specified in the question" in bench.PROMPT
    assert "even if other actions could" in bench.PROMPT

