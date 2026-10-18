"""Offline evidence that continuation never retries or rewrites parent calls."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

import continue_paper_run as continuation
import run_paper_parallel as parallel
import satori_paper_benchmark as bench
import compare_paper_model_runs as compare
from test_run_paper_parallel import dataset, Factory, records


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def parent_run(dataset):
    root, manifest, taxonomy, protocol, _ = dataset
    def stop(row):
        if row["processed"] == 2:
            raise KeyboardInterrupt
    factory = Factory(failures={"s001"}, mutate=lambda trace: trace.update(http_status=429) if trace.get("http_status") else None)
    with pytest.raises(KeyboardInterrupt):
        parallel.infer_parallel(manifest, taxonomy, root / "parent", factory,
                                protocol_path=protocol, max_workers=1, progress=stop)
    return root / "parent"


def prepared(dataset):
    parent = parent_run(dataset)
    protocol = dataset[0] / "continuation-protocol.json"
    continuation.prepare_continuation_protocol(parent, protocol, min_start_interval=5, rate_limit_cooldown=60)
    return parent, protocol


def test_only_missing_ids_called_once_and_parent_errors_preserved(dataset, monkeypatch):
    parent, protocol = prepared(dataset)
    original_hashes = continuation._parent_hashes(parent)
    clock = Clock()
    monkeypatch.setattr(continuation, "time", clock)
    f = Factory(failures={"s003"}, mutate=lambda trace: trace.update(http_status=429) if trace.get("http_status") else None)
    starts = []
    def factory():
        p = f()
        predict = p.predict
        def wrapped(sample, taxonomy):
            starts.append((sample["sample_id"], clock.now))
            return predict(sample, taxonomy)
        p.predict = wrapped
        return p
    out = dataset[0] / "child"
    meta = continuation.continue_infer(parent, out, factory, protocol_path=protocol)
    assert f.calls == [f"s{i:03d}" for i in range(2, 8)]
    assert meta["api_calls"] == 8 and meta["continuation_inherited_api_calls"] == 2
    assert meta["continuation_new_api_calls"] == 6 and meta["planned_samples"] == 8
    assert starts[0][1] >= 60
    assert starts[1][1] - starts[0][1] >= 5
    assert starts[2][1] - starts[1][1] >= 60
    assert continuation._parent_hashes(parent) == original_hashes
    assert (out / "api-calls.jsonl").read_bytes().startswith((parent / "api-calls.jsonl").read_bytes())
    assert continuation.validate_continuation_run(out)["lineage_verified"]
    assert bench.score(out, dataset[4])["correct"] == 6
    assert compare.audit_and_score(out, dataset[0], protocol_source=protocol)["correct"] == 6


def test_parent_file_added_after_freeze_refuses_without_request(dataset):
    parent, protocol = prepared(dataset)
    (parent / "late-report.txt").write_text("changed parent")
    f = Factory()
    with pytest.raises(ValueError, match="parent files changed"):
        continuation.continue_infer(parent, dataset[0] / "child", f, protocol_path=protocol)
    assert f.calls == []


@pytest.mark.parametrize("change", ["unfinalized", "incomplete", "duplicate"])
def test_parent_must_have_finalized_complete_unique_accounting(dataset, change):
    parent = parent_run(dataset)
    path = parent / "run.json"
    meta = json.loads(path.read_text())
    if change == "unfinalized":
        meta["status"] = "running"
    elif change == "incomplete":
        meta["request_accounting_complete"] = False
    else:
        p = parent / "api-calls.jsonl"
        with p.open("a") as handle:
            handle.write(p.read_text().splitlines()[0] + "\n")
    path.write_text(json.dumps(meta))
    with pytest.raises((ValueError, parallel.TraceContractError)):
        continuation.prepare_continuation_protocol(parent, dataset[0] / "next.json")


def test_interrupt_keeps_parent_and_new_request_accounting(dataset, monkeypatch):
    parent, protocol = prepared(dataset)
    monkeypatch.setattr(continuation, "time", Clock())
    def interrupt(value):
        raise KeyboardInterrupt
    f = Factory()
    out = dataset[0] / "child"
    with pytest.raises(KeyboardInterrupt):
        continuation.continue_infer(parent, out, f, protocol_path=protocol, progress=interrupt)
    assert f.calls == ["s002"]
    audit = continuation.validate_continuation_run(out)
    assert audit["api_calls"] == 3 and audit["not_run_samples"] == 5
    meta = json.loads((out / "run.json").read_text())
    assert meta["status"] == "aborted" and meta["continuation_new_api_calls"] == 1


def test_model_or_payload_changes_refused(dataset, monkeypatch):
    parent, protocol = prepared(dataset)
    f = Factory()
    def factory():
        p = f()
        p.generation_parameters = {"max_tokens": 100, "temperature": 0}
        return p
    with pytest.raises(ValueError, match="exact parent"):
        continuation.continue_infer(parent, dataset[0] / "child", factory, protocol_path=protocol)
    assert f.calls == []


def test_paced_provider_uses_same_cooldown_without_retry(dataset, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(continuation, "time", clock)
    f = Factory(failures={"s000"}, mutate=lambda trace: trace.update(http_status=429) if trace.get("http_status") else None)
    p = continuation.PacedProvider(f(), min_start_interval=5, rate_limit_cooldown=60)
    manifest, _ = bench._read(dataset[1])
    taxonomy, _ = bench._read(dataset[2])
    with pytest.raises(RuntimeError):
        p.predict(manifest["samples"][0], taxonomy)
    p.predict(manifest["samples"][1], taxonomy)
    assert clock.now >= 60 and f.calls == ["s000", "s001"]
