"""Explicit missing-GT binding exception and concurrent trace ordering checks."""

from copy import deepcopy
import json
import shutil

import pytest

import compare_paper_model_runs as compare
from test_compare_paper_model_runs import frozen, write


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_records(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_target_binding(data, run, value="missing"):
    protocol = load(run / "protocol.json")
    if value == "missing":
        del protocol["source_gt_sha256"]
    else:
        protocol["source_gt_sha256"] = value
    source = data / "target-protocol.json"
    source_hash = write(source, protocol, crlf=True)
    snapshot_hash = write(run / "protocol.json", protocol)
    meta = load(run / "run.json")
    meta.update(source_protocol_path="/remote/target-protocol.json", source_protocol_sha256=source_hash,
                protocol_snapshot_sha256=snapshot_hash, started_utc="2026-10-18T05:00:00+00:00")
    write(run / "run.json", meta)


@pytest.fixture
def witness_pair(frozen):
    data, run = frozen
    witness = run.parent / "strict-prior-run"
    shutil.copytree(run, witness)
    meta = load(witness / "run.json")
    meta.update(status="completed", started_utc="2026-10-17T08:00:00+00:00", finished_utc="2026-10-17T09:00:00+00:00",
                processed_samples=4, api_calls=4, failed_samples=2, not_run_samples=0)
    write(witness / "run.json", meta)
    for filename in ("api-calls.jsonl", "predictions.jsonl"):
        rows = records(witness / filename)
        added = deepcopy(rows[-1])
        added["sample_id"] = "s003"
        rows.append(added)
        write_records(witness / filename, rows)
    make_target_binding(data, run)
    return data, run, witness


def test_missing_gt_rejected_by_default(witness_pair):
    data, run, _ = witness_pair
    with pytest.raises(ValueError, match="Missing per-run GT binding"):
        compare.audit_and_score(run, data)


def test_explicit_prior_witness_is_audited_and_disclosed(witness_pair):
    data, run, witness = witness_pair
    original = {path.name: path.read_bytes() for path in run.iterdir() if path.is_file()}
    result = compare.audit_and_score(run, data, gt_witness_run=witness)
    audit = result["audit"]
    assert audit["passed"] and audit["per_run_gt_binding_missing"] is True
    assert audit["gt_binding_verification"] == "prior_strict_run_witness"
    assert audit["gt_witness"]["run_dir"] == str(witness.resolve())
    assert audit["gt_witness"]["source_protocol_sha256"] == load(witness / "run.json")["source_protocol_sha256"]
    assert audit["gt_witness"]["gt_sha256"] == result["input_fingerprint"]["gt_sha256"]
    assert result["correct"] == 1 and result["failed_samples"] == 1
    assert original == {path.name: path.read_bytes() for path in run.iterdir() if path.is_file()}


@pytest.mark.parametrize("value", [None, "wrong-gt-hash"])
def test_present_but_incorrect_binding_cannot_be_overridden(witness_pair, value):
    data, run, witness = witness_pair
    make_target_binding(data, run, value)
    with pytest.raises(ValueError, match="GT bytes differ"):
        compare.audit_and_score(run, data, gt_witness_run=witness)


def test_actual_gt_change_is_rejected_by_strict_witness(witness_pair):
    data, run, witness = witness_pair
    gt = load(data / "gt.json")
    gt["samples"][0]["answers"] = ["altered"]
    write(data / "gt.json", gt)
    with pytest.raises(ValueError, match="GT bytes differ"):
        compare.audit_and_score(run, data, gt_witness_run=witness)


@pytest.mark.parametrize("finished", ["2026-10-18T05:00:00+00:00", "2026-10-18T06:00:00+00:00"])
def test_same_or_later_witness_is_rejected(witness_pair, finished):
    data, run, witness = witness_pair
    meta = load(witness / "run.json")
    meta["finished_utc"] = finished
    write(witness / "run.json", meta)
    with pytest.raises(ValueError, match="finished before"):
        compare.audit_and_score(run, data, gt_witness_run=witness)


@pytest.mark.parametrize("field,value", [("synthetic", True), ("status", "aborted"),
                                        ("source_manifest_sha256", "wrong-input"),
                                        ("source_taxonomy_sha256", "wrong-taxonomy")])
def test_invalid_witness_is_rejected(witness_pair, field, value):
    data, run, witness = witness_pair
    meta = load(witness / "run.json")
    meta[field] = value
    write(witness / "run.json", meta)
    with pytest.raises(ValueError):
        compare.audit_and_score(run, data, gt_witness_run=witness)


@pytest.mark.parametrize("timestamp", [None, "not-a-date", "2026-10-18T05:00:00"])
def test_missing_or_ambiguous_target_time_is_rejected(witness_pair, timestamp):
    data, run, witness = witness_pair
    meta = load(run / "run.json")
    meta["started_utc"] = timestamp
    write(run / "run.json", meta)
    with pytest.raises(ValueError, match="timestamp"):
        compare.audit_and_score(run, data, gt_witness_run=witness)


def test_self_witness_is_rejected(witness_pair):
    data, run, _ = witness_pair
    with pytest.raises(ValueError, match="own GT witness"):
        compare.audit_and_score(run, data, gt_witness_run=run)


def test_witness_cannot_chain_missing_bindings(witness_pair):
    data, run, witness = witness_pair
    protocol = load(witness / "protocol.json")
    del protocol["source_gt_sha256"]
    meta = load(witness / "run.json")
    meta["source_protocol_sha256"] = write(data / "protocol.json", protocol, crlf=True)
    meta["protocol_snapshot_sha256"] = write(witness / "protocol.json", protocol)
    write(witness / "run.json", meta)
    with pytest.raises(ValueError, match="Missing per-run GT binding"):
        compare.audit_and_score(run, data, gt_witness_run=witness)


def test_unnecessary_exception_on_bound_run_is_rejected(frozen):
    data, run = frozen
    with pytest.raises(ValueError, match="allowed only"):
        compare.audit_and_score(run, data, gt_witness_run=run.parent / "unused")


def test_parallel_completion_order_does_not_change_pairing(frozen):
    data, run = frozen
    before = compare.audit_and_score(run, data)
    for filename in ("api-calls.jsonl", "predictions.jsonl"):
        rows = records(run / filename)
        write_records(run / filename, list(reversed(rows)))
    after = compare.audit_and_score(run, data)
    assert after["paired"] == before["paired"]
    assert after["correct"] == before["correct"]
