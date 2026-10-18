"""Offline checks for bounded one-attempt inference and request accounting."""

from copy import deepcopy
import hashlib
import json
import signal
from pathlib import Path
import threading
import time

import pytest

import compare_paper_model_runs as compare
import satori_paper_benchmark as bench
import run_paper_parallel as parallel


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def dataset(tmp_path):
    raw = b"Frozen image bytes for offline providers only"
    (tmp_path / "frame.png").write_bytes(raw)
    samples = [{"sample_id": f"s{i:03d}", "task_kind": "video_qa", "question": "What is visible?",
                "candidate_group": None, "cutoff_ms": 3000,
                "frames": [{"frame_id": f"v{j:03d}", "timestamp_ms": j * 1000,
                            "image_path": "frame.png", "sha256": hashlib.sha256(raw).hexdigest()}
                           for j in range(4)]} for i in range(8)]
    manifest = {"schema_version": 1, "benchmark_id": "parallel-unit", "samples": samples}
    taxonomy = {"schema_version": 1, "goals": [], "actions": []}
    gt = {"schema_version": 1, "benchmark_id": "parallel-unit", "provenance": {"source": "test"},
          "samples": [{"sample_id": s["sample_id"], "answers": ["cup"], "metadata": {}} for s in samples]}
    mp = dump(tmp_path / "infer-manifest.json", manifest)
    tp = dump(tmp_path / "taxonomy.json", taxonomy)
    gp = dump(tmp_path / "gt.json", gt)
    protocol = {"protocol_id": "offline-test", "source_manifest_sha256": bench._sha(mp.read_bytes()),
                "source_taxonomy_sha256": bench._sha(tp.read_bytes()), "source_gt_sha256": bench._sha(gp.read_bytes())}
    pp = dump(tmp_path / "protocol.json", protocol)
    return tmp_path, mp, tp, pp, gp


class Factory:
    def __init__(self, *, delays=None, failures=(), missing=(), duplicate=(), interrupt=(), mutate=None):
        self.delays, self.failures = delays or {}, set(failures)
        self.missing, self.duplicate, self.interrupt = set(missing), set(duplicate), set(interrupt)
        self.mutate = mutate
        self.calls, self.instances = [], []
        self.lock = threading.Lock()
        self.active = self.peak = 0

    def __call__(self):
        provider = FakeProvider(self)
        self.instances.append(provider)
        return provider


class FakeProvider:
    model = expected_response_model = "offline-test"
    base_url = "https://example.invalid/v1"
    prompt = bench.PROMPT
    synthetic = False  # Complete mocked HTTP traces exercise the real comparator.
    generation_parameters = {"max_tokens": 4096, "temperature": 0.0}
    timeout = 180

    def __init__(self, factory):
        self.factory, self.traces = factory, []

    def predict(self, sample, taxonomy):
        f, sid = self.factory, sample["sample_id"]
        with f.lock:
            f.calls.append(sid)
            f.active += 1
            f.peak = max(f.peak, f.active)
        try:
            time.sleep(f.delays.get(sid, 0.005))
            value = {"answer": "cup", "evidence_frame_ids": ["v000"]}
            failed = sid in f.failures or sid in f.interrupt
            trace = {"status": "error" if failed else "success", "latency_ms": 10,
                     "requested_model": self.model, "returned_model": self.model,
                     "expected_response_model": self.expected_response_model, "task_kind": sample["task_kind"],
                     "image_sha256": {a["frame_id"]: a["sha256"] for a in sample["frames"]},
                     "system_prompt_sha256": bench._sha(self.prompt.encode()),
                     "payload_sha256": bench._sha(bench._json_text(bench.visible_payload(sample, taxonomy)).encode()),
                     "generation_parameters": deepcopy(self.generation_parameters), "timeout_seconds": self.timeout,
                     "finish_reason": "stop", "raw_json": value, "raw_json_text": bench._json_text(value)}
            if failed:
                trace.update(error_code="http_error", http_status=503)
            if f.mutate:
                f.mutate(trace)
            if sid not in f.missing:
                self.traces.append(trace)
            if sid in f.duplicate:
                self.traces.append(deepcopy(trace))
            if sid in f.interrupt:
                raise KeyboardInterrupt
            if sid in f.failures:
                raise RuntimeError("Simulated HTTP failure")
            return value
        finally:
            with f.lock:
                f.active -= 1


def run(dataset, factory, **kwargs):
    root, manifest, taxonomy, protocol, _ = dataset
    return parallel.infer_parallel(manifest, taxonomy, root / "run", factory,
                                   protocol_path=protocol, max_workers=4, **kwargs)


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_out_of_order_is_complete_bounded_and_compatible(dataset):
    factory = Factory(delays={"s000": 0.12})
    meta = run(dataset, factory)
    root = dataset[0]
    rows = records(root / "run" / "predictions.jsonl")
    assert rows[0]["sample_id"] != "s000"
    assert len(set(factory.calls)) == len(factory.calls) == 8
    assert 1 < factory.peak <= 4
    assert len({id(p) for p in factory.instances}) == 4
    assert meta["input_sha256_audit_before_start"]["passed"]
    assert meta["execution"]["max_in_flight"] == 4
    assert meta["execution"]["submission_policy"] == "refill_only_after_completed_record_is_persisted"
    assert bench.audit_run(root / "run")["api_calls"] == 8
    assert bench.score(root / "run", dataset[4])["correct"] == 8
    assert compare.audit_and_score(root / "run", root)["correct"] == 8
    assert len(list((root / "run" / "request-records").glob("*.json"))) == 8


def test_http_failure_never_retried_and_keeps_denominator(dataset):
    f = Factory(failures={"s002", "s005"})
    meta = run(dataset, f)
    result = bench.score(dataset[0] / "run", dataset[4])
    assert meta["api_calls"] == 8
    assert len(set(f.calls)) == len(f.calls) == 8
    assert result["planned_samples"] == 8 and result["failed_samples"] == 2
    assert result["correct"] == 6


@pytest.mark.parametrize("option", ["missing", "duplicate"])
def test_broken_trace_accounting_raises_and_drains_started_requests(dataset, option):
    f = Factory(**{option: {"s000"}}, delays={"s000": 0.001, "s001": 0.03, "s002": 0.04, "s003": 0.05})
    with pytest.raises(parallel.TraceContractError):
        run(dataset, f)
    meta = json.loads((dataset[0] / "run" / "run.json").read_text())
    assert meta["status"] == "failed" and meta["request_accounting_complete"] is False
    assert set(f.calls) == {"s000", "s001", "s002", "s003"}
    assert meta["processed_samples"] == 4 and meta["not_run_samples"] == 4
    assert len(records(dataset[0] / "run" / "predictions.jsonl")) == 4
    with pytest.raises(parallel.TraceContractError, match="incomplete"):
        parallel.validate_parallel_run(dataset[0] / "run")


@pytest.mark.parametrize("field,value", [("generation_parameters", {"max_tokens": 1, "temperature": 0}),
                                          ("image_sha256", {}), ("payload_sha256", "wrong"),
                                          ("latency_ms", "not a number")])
def test_changed_request_fingerprint_rejected(dataset, field, value):
    f = Factory(mutate=lambda trace: trace.update({field: value}))
    with pytest.raises(parallel.TraceContractError):
        run(dataset, f)


def test_all_images_validated_before_first_request(dataset):
    (dataset[0] / "frame.png").write_bytes(b"corrupted")
    f = Factory()
    with pytest.raises(ValueError, match="image"):
        run(dataset, f)
    assert f.calls == []


def test_progress_keyboard_interrupt_drains_bounded_inflight(dataset):
    f = Factory(delays={"s000": 0.002, "s001": 0.03, "s002": 0.04, "s003": 0.05})
    def progress(value):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run(dataset, f, progress=progress)
    meta = json.loads((dataset[0] / "run" / "run.json").read_text())
    assert meta["status"] == "aborted"
    assert len(f.calls) == 4 and meta["not_run_samples"] == 4
    assert bench.audit_run(dataset[0] / "run")["api_calls"] == 4
    assert bench.score(dataset[0] / "run", dataset[4])["accuracy"] == 0.5


def test_progress_exception_drains_and_preserves_results(dataset):
    f = Factory(delays={"s000": 0.002, "s001": 0.03, "s002": 0.04, "s003": 0.05})
    def progress(value):
        raise RuntimeError("status sink failed")
    with pytest.raises(RuntimeError, match="status sink"):
        run(dataset, f, progress=progress)
    meta = json.loads((dataset[0] / "run" / "run.json").read_text())
    assert meta["status"] == "failed" and meta["processed_samples"] == 4
    assert bench.audit_run(dataset[0] / "run")["api_calls"] == 4


def test_worker_interrupt_drains_other_workers(dataset):
    f = Factory(interrupt={"s000"}, delays={"s000": 0.001, "s001": 0.03, "s002": 0.04, "s003": 0.05})
    with pytest.raises(KeyboardInterrupt):
        run(dataset, f)
    meta = json.loads((dataset[0] / "run" / "run.json").read_text())
    assert meta["status"] == "aborted" and meta["processed_samples"] == 4
    result = bench.score(dataset[0] / "run", dataset[4])
    assert result["not_run_samples"] == 4 and result["failed_samples"] == 1


def test_sigint_stops_submission_and_restores_handler(dataset):
    original = signal.getsignal(signal.SIGINT)
    f = Factory(delays={"s000": 0.002, "s001": 0.03, "s002": 0.04, "s003": 0.05})
    def progress(value):
        signal.raise_signal(signal.SIGINT)
    with pytest.raises(KeyboardInterrupt):
        run(dataset, f, progress=progress)
    assert signal.getsignal(signal.SIGINT) == original
    assert set(f.calls) == {"s000", "s001", "s002", "s003"}
    assert bench.audit_run(dataset[0] / "run")["api_calls"] == 4


def test_snapshots_exist_before_first_call_and_fingerprint_binds_runner(dataset):
    f = Factory()
    original_factory = f.__call__
    def factory():
        p = original_factory()
        original_predict = p.predict
        def predict(sample, taxonomy):
            target = dataset[0] / "run"
            meta = json.loads((target / "run.json").read_text())
            assert meta["status"] == "running"
            assert len(json.loads((target / "manifest.json").read_text())["samples"]) == 8
            assert meta["execution"]["runner_sha256"] == bench._sha(Path(parallel.__file__).read_bytes())
            assert meta["execution"]["benchmark_source_sha256"] == bench._sha(Path(bench.__file__).read_bytes())
            assert len(meta["run_fingerprint_sha256"]) == 64
            return original_predict(sample, taxonomy)
        p.predict = predict
        return p
    assert run(dataset, factory)["api_calls"] == 8


def test_preexisting_output_refuses_resumption(dataset):
    (dataset[0] / "run").mkdir()
    f = Factory()
    with pytest.raises(FileExistsError):
        run(dataset, f)
    assert f.calls == []


def test_unexpected_future_exception_keeps_traces_and_drains(dataset, monkeypatch):
    original = parallel._invoke
    def invoke(provider, sample, *args):
        result = original(provider, sample, *args)
        if sample["sample_id"] == "s000":
            raise RuntimeError("Unexpected driver error after request")
        return result
    monkeypatch.setattr(parallel, "_invoke", invoke)
    f = Factory(delays={"s000": 0.002, "s001": 0.03, "s002": 0.04, "s003": 0.05})
    with pytest.raises(RuntimeError, match="Unexpected"):
        run(dataset, f)
    meta = json.loads((dataset[0] / "run" / "run.json").read_text())
    assert meta["api_calls"] == 4 and meta["processed_samples"] == 4
    assert meta["request_accounting_complete"] is False
    assert len(records(dataset[0] / "run" / "api-calls.jsonl")) == 4


def test_storage_failure_stops_submission_drains_others_and_refuses_scoring(dataset, monkeypatch):
    original = parallel._atomic_json
    def failing_write(path, value):
        if path.name == "s000.json":
            raise OSError("Simulated disk failure for one result")
        return original(path, value)
    monkeypatch.setattr(parallel, "_atomic_json", failing_write)
    f = Factory(delays={"s000": 0.002, "s001": 0.03, "s002": 0.04, "s003": 0.05})
    with pytest.raises(OSError, match="disk failure"):
        run(dataset, f)
    meta = json.loads((dataset[0] / "run" / "run.json").read_text())
    assert len(f.calls) == 4 and f.active == 0
    assert meta["status"] == "failed" and meta["request_accounting_complete"] is False
    assert meta["processed_samples"] == 4 and meta["api_calls"] == 4
    assert len(list((dataset[0] / "run" / "request-records").glob("*.json"))) == 3
    with pytest.raises(parallel.TraceContractError, match="incomplete"):
        parallel.validate_parallel_run(dataset[0] / "run")


@pytest.mark.parametrize("workers", [0, 5, True])
def test_worker_limit_is_explicit(dataset, workers):
    with pytest.raises(ValueError):
        parallel.infer_parallel(dataset[1], dataset[2], dataset[0] / "run", Factory(),
                                protocol_path=dataset[3], max_workers=workers)
