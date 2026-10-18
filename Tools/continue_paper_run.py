"""Auditable paced continuation of missing IDs from an immutable parent run.

Completed and failed parent records are inherited verbatim. Only IDs entirely
absent from the parent predictions can produce new calls. Aggregate api_calls
includes inherited calls; continuation_new_api_calls counts this process only.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import threading
import time

import run_paper_parallel as parallel
import satori_paper_benchmark as bench


def _records(path):
    return [bench._strict_response_json(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def _parent_hashes(parent):
    result = {}
    for path in sorted(parent.rglob("*")):
        if path.is_symlink():
            raise ValueError("Parent lineage cannot contain symlinks")
        if path.is_file():
            result[path.relative_to(parent).as_posix()] = bench._sha(path.read_bytes())
    return result


def _validate_parent(parent):
    meta, _ = bench._read(parent / "run.json")
    if meta.get("status") not in ("completed", "aborted", "failed"):
        raise ValueError("Parent must be finalized before continuation")
    parallel.validate_parallel_run(parent)
    if "continuation" in meta:
        validate_continuation_run(parent)
    return meta


def prepare_continuation_protocol(parent_run, protocol_path, *, min_start_interval=5, rate_limit_cooldown=60):
    """Freeze immutable lineage and the fixed paced scheduling amendment."""
    _validate_schedule(min_start_interval, rate_limit_cooldown)
    parent, destination = Path(parent_run).resolve(), Path(protocol_path).resolve()
    if destination == parent or parent in destination.parents:
        raise ValueError("Continuation protocol must be outside the immutable parent")
    meta = _validate_parent(parent)
    protocol, _ = bench._read(parent / "protocol.json")
    records = _records(parent / "predictions.jsonl")
    traces = _records(parent / "api-calls.jsonl")
    lineage = {"parent_run_path": str(parent), "parent_files_sha256": _parent_hashes(parent),
               "inherited_sample_ids": [row["sample_id"] for row in records],
               "inherited_api_sample_ids": [row["sample_id"] for row in traces],
               "parent_api_calls": meta["api_calls"],
               "record_policy": "Copy all completed and failed parent records unchanged; call only IDs absent from parent predictions.",
               "aggregate_api_calls": "Includes inherited calls; continuation_new_api_calls is this child process only."}
    execution = {"max_inflight_requests": 1, "min_request_start_interval_seconds": min_start_interval,
                 "http_429_future_request_cooldown_seconds": rate_limit_cooldown,
                 "initial_delay_seconds": rate_limit_cooldown if traces and traces[-1].get("http_status") == 429 else min_start_interval,
                 "automatic_retry": False, "requests_per_sample_across_lineage": 1,
                 "runner_sha256": bench._sha(Path(__file__).read_bytes()),
                 "parallel_helper_sha256": bench._sha(Path(parallel.__file__).read_bytes()),
                 "benchmark_source_sha256": bench._sha(Path(bench.__file__).read_bytes())}
    protocol = deepcopy(protocol)
    protocol.update(protocol_id=str(protocol.get("protocol_id", "paper")) + "-paced-continuation",
                    continuation=lineage, continuation_execution=execution,
                    parent_protocol_sha256=bench._sha((parent / "protocol.json").read_bytes()),
                    max_inflight_requests=1, concurrency_policy=f"One request at a time; starts at least {min_start_interval} seconds apart; after HTTP 429 wait at least {rate_limit_cooldown} seconds before any future new sample. Never retry a sample.",
                    automatic_retry=False, created_utc=datetime.now(timezone.utc).isoformat())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    return protocol


def _validate_lineage(parent, protocol):
    lineage = protocol.get("continuation", {})
    if lineage.get("parent_run_path") != str(parent) or lineage.get("parent_files_sha256") != _parent_hashes(parent):
        raise ValueError("Immutable parent files changed after continuation protocol freeze")
    meta = _validate_parent(parent)
    predictions, traces = _records(parent / "predictions.jsonl"), _records(parent / "api-calls.jsonl")
    if (lineage.get("inherited_sample_ids") != [row["sample_id"] for row in predictions]
            or lineage.get("inherited_api_sample_ids") != [row["sample_id"] for row in traces]
            or lineage.get("parent_api_calls") != meta["api_calls"]):
        raise ValueError("Frozen inherited record IDs differ from the parent")
    return meta, predictions, traces


def _wait_until(deadline, stop):
    while not stop.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.2, remaining))
    raise KeyboardInterrupt


def _validate_schedule(interval, cooldown):
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in (interval, cooldown)):
        raise ValueError("Pacing intervals must be finite and positive")


class RequestPacer:
    """Shared no-retry pacing for continuation and fresh sequential runs."""
    def __init__(self, *, min_start_interval=5, rate_limit_cooldown=60, initial_delay=0):
        _validate_schedule(min_start_interval, rate_limit_cooldown)
        self.interval, self.cooldown = min_start_interval, rate_limit_cooldown
        self.next_allowed = time.monotonic() + initial_delay

    def wait(self, stop=None):
        _wait_until(self.next_allowed, stop or threading.Event())

    def started(self):
        self.next_allowed = time.monotonic() + self.interval

    def observe(self, traces):
        if any(trace.get("http_status") == 429 for trace in traces):
            self.next_allowed = max(self.next_allowed, time.monotonic() + self.cooldown)


class PacedProvider:
    """Wrap one provider for max_workers=1; leave actual payload/traces intact."""
    def __init__(self, provider, *, min_start_interval=5, rate_limit_cooldown=60, initial_delay=0):
        self.provider = provider
        self.pacer = RequestPacer(min_start_interval=min_start_interval,
                                 rate_limit_cooldown=rate_limit_cooldown, initial_delay=initial_delay)

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def predict(self, sample, taxonomy):
        self.pacer.wait()
        self.pacer.started()
        before = len(self.provider.traces)
        try:
            return self.provider.predict(sample, taxonomy)
        finally:
            self.pacer.observe(self.provider.traces[before:])


def validate_continuation_run(output_dir):
    """Audit aggregate records plus byte-exact parent prefixes and provenance."""
    output = Path(output_dir).resolve()
    audit = parallel.validate_parallel_run(output)
    meta, _ = bench._read(output / "run.json")
    protocol, _ = bench._read(output / "protocol.json")
    lineage = protocol["continuation"]
    parent = Path(lineage["parent_run_path"])
    _, inherited, inherited_traces = _validate_lineage(parent, protocol)
    for name in ("predictions.jsonl", "api-calls.jsonl"):
        original = (parent / name).read_bytes()
        if not (output / name).read_bytes().startswith(original):
            raise ValueError("Inherited parent JSONL bytes changed")
    predictions, traces = _records(output / "predictions.jsonl"), _records(output / "api-calls.jsonl")
    inherited_ids = {row["sample_id"] for row in inherited}
    new_ids = {row["sample_id"] for row in predictions} - inherited_ids
    provenance, _ = bench._read(output / "record-provenance.json")
    if provenance.get("parent_files_sha256") != lineage["parent_files_sha256"]:
        raise ValueError("Record provenance differs from frozen parent file hashes")
    origins = provenance.get("sample_origins")
    expected = {sid: "parent" for sid in inherited_ids}
    expected.update({sid: "continuation_new" for sid in new_ids})
    if origins != expected:
        raise ValueError("Prediction provenance is incomplete or inconsistent")
    if (meta.get("continuation_inherited_api_calls") != len(inherited_traces)
            or meta.get("continuation_new_api_calls") != len(traces) - len(inherited_traces)
            or meta.get("continuation_new_processed_samples") != len(new_ids)):
        raise ValueError("Aggregate and new call counters are inconsistent")
    return {**audit, "lineage_verified": True, "inherited_api_calls": len(inherited_traces),
            "new_api_calls": len(traces) - len(inherited_traces)}


def continue_infer(parent_run, output_dir, provider_factory, *, protocol_path, progress=None):
    """Run only missing parent IDs, preserving the original full denominator."""
    parent, output, protocol_path = map(lambda path: Path(path).resolve(), (parent_run, output_dir, protocol_path))
    if output == parent or parent in output.parents:
        raise ValueError("Child run must be outside the immutable parent directory")
    protocol, protocol_hash = bench._read(protocol_path)
    parent_meta, inherited, inherited_traces = _validate_lineage(parent, protocol)
    execution = protocol["continuation_execution"]
    expected_code = {"runner_sha256": bench._sha(Path(__file__).read_bytes()),
                     "parallel_helper_sha256": bench._sha(Path(parallel.__file__).read_bytes()),
                     "benchmark_source_sha256": bench._sha(Path(bench.__file__).read_bytes())}
    _validate_schedule(execution.get("min_request_start_interval_seconds"), execution.get("http_429_future_request_cooldown_seconds"))
    if (any(execution.get(k) != v for k, v in expected_code.items())
            or execution.get("max_inflight_requests") != 1
            or execution.get("initial_delay_seconds") not in (
                execution["min_request_start_interval_seconds"], execution["http_429_future_request_cooldown_seconds"])
            or execution.get("automatic_retry") is not False):
        raise ValueError("Continuation scheduling or source code differs from frozen protocol")
    manifest, _ = bench._read(parent / "manifest.json")
    taxonomy, _ = bench._read(parent / "taxonomy.json")
    manifest = bench.validate_manifest(manifest, taxonomy)
    bench._protocol_input_hashes(protocol, parent_meta)
    manifest_path = Path(parent_meta["source_manifest_path"])
    images = bench._image_audit(manifest, manifest_path)
    if not images["passed"]:
        raise ValueError("Frozen image bytes changed before continuation")
    provider = provider_factory()
    identity = parallel._identity(provider)
    if provider.traces or any(parent_meta.get(k) != v for k, v in identity.items()):
        raise ValueError("Continuation must preserve the exact parent model and request contract")
    inherited_ids = {row["sample_id"] for row in inherited}
    remaining = [row for row in manifest["samples"] if row["sample_id"] not in inherited_ids]
    output.mkdir(parents=True, exist_ok=False)
    (output / "request-records").mkdir()
    for name, value in (("manifest", manifest), ("taxonomy", taxonomy), ("protocol", protocol)):
        parallel._atomic_json(output / (name + ".json"), value)
    for name in ("predictions.jsonl", "api-calls.jsonl"):
        raw = (parent / name).read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ValueError("Parent JSONL requires a complete final line before byte-exact continuation")
        (output / name).write_bytes(raw)
    meta = {key: deepcopy(parent_meta[key]) for key in (
        "schema_version", "benchmark_id", "synthetic", "model", "endpoint", "expected_response_model",
        "generation_parameters", "timeout_seconds", "automatic_retry", "normalization", "prompt_sha256",
        "source_manifest_sha256", "source_taxonomy_sha256", "source_manifest_path", "source_taxonomy_path")}
    meta.update(status="running", planned_samples=len(manifest["samples"]), source_protocol_sha256=protocol_hash,
                source_protocol_path=str(protocol_path), started_utc=datetime.now(timezone.utc).isoformat(),
                request_accounting_complete=True, input_sha256_audit_before_start=images,
                continuation=deepcopy(protocol["continuation"]), execution=deepcopy(execution),
                continuation_inherited_api_calls=len(inherited_traces), continuation_new_api_calls=0,
                continuation_new_processed_samples=0)
    for name in ("manifest", "taxonomy", "protocol"):
        meta[name + "_snapshot_sha256"] = bench._sha((output / (name + ".json")).read_bytes())
    meta["run_fingerprint_sha256"] = bench._sha(bench._json_text(meta).encode("utf-8"))
    provenance = {"parent_run_path": str(parent), "parent_files_sha256": protocol["continuation"]["parent_files_sha256"],
                  "sample_origins": {sid: "parent" for sid in inherited_ids}}
    parallel._atomic_json(output / "record-provenance.json", provenance)
    parallel._atomic_json(output / "run.json", meta)
    stop = threading.Event()
    install = threading.current_thread() is threading.main_thread()
    previous = signal.getsignal(signal.SIGINT) if install else None
    if install:
        signal.signal(signal.SIGINT, lambda signum, frame: stop.set())
    completed = sum(row["status"] == "completed" for row in inherited)
    failed = sum(row["status"] == "failed" for row in inherited)
    attempts = len(inherited_traces)
    pacer = RequestPacer(min_start_interval=execution["min_request_start_interval_seconds"],
                         rate_limit_cooldown=execution["http_429_future_request_cooldown_seconds"],
                         initial_delay=execution["initial_delay_seconds"])
    try:
        with (output / "predictions.jsonl").open("a", encoding="utf-8", newline="\n") as predictions, \
             (output / "api-calls.jsonl").open("a", encoding="utf-8", newline="\n") as calls:
            for source_sample in remaining:
                pacer.wait(stop)
                pacer.started()
                row, emitted, worker_stop, violation = parallel._invoke(provider, source_sample, taxonomy, manifest_path, identity)
                try:
                    parallel._atomic_json(output / "request-records" / (row["sample_id"] + ".json"),
                                          {"prediction": row, "api_calls": emitted, "origin": "continuation_new"})
                    for trace in emitted:
                        calls.write(bench._json_text({"sample_id": row["sample_id"], **trace}) + "\n")
                    calls.flush()
                    os.fsync(calls.fileno())
                    predictions.write(bench._json_text(row) + "\n")
                    predictions.flush()
                    os.fsync(predictions.fileno())
                    provenance["sample_origins"][row["sample_id"]] = "continuation_new"
                    parallel._atomic_json(output / "record-provenance.json", provenance)
                except BaseException:
                    meta["request_accounting_complete"] = False
                    raise
                completed += row["status"] == "completed"
                failed += row["status"] == "failed"
                attempts += len(emitted)
                meta["continuation_new_api_calls"] += len(emitted)
                meta["continuation_new_processed_samples"] += 1
                if violation:
                    meta["request_accounting_complete"] = False
                if worker_stop:
                    raise worker_stop
                pacer.observe(emitted)
                if progress:
                    progress({"sample_id": row["sample_id"], "status": row["status"],
                              "processed": completed + failed, "planned": len(manifest["samples"]),
                              "inherited_api_calls": len(inherited_traces),
                              "new_api_calls": meta["continuation_new_api_calls"],
                              "next_new_request_delay_seconds": max(0, round(pacer.next_allowed - time.monotonic(), 3))})
                if stop.is_set():
                    raise KeyboardInterrupt
        meta["status"] = "completed"
    except KeyboardInterrupt:
        meta["status"] = "aborted"
        raise
    except BaseException:
        meta["status"] = "failed"
        raise
    finally:
        parent_unchanged = protocol["continuation"]["parent_files_sha256"] == _parent_hashes(parent)
        if not parent_unchanged:
            meta["request_accounting_complete"] = False
            meta["status"] = "failed"
        meta.update(finished_utc=datetime.now(timezone.utc).isoformat(), successful_samples=completed,
                    failed_samples=failed, processed_samples=completed + failed, api_calls=attempts,
                    not_run_samples=len(manifest["samples"]) - completed - failed,
                    input_sha256_audit_at_finish=bench._image_audit(manifest, manifest_path),
                    immutable_parent_audit_at_finish=parent_unchanged)
        parallel._atomic_json(output / "run.json", meta)
        if install:
            signal.signal(signal.SIGINT, previous)
    validate_continuation_run(output)
    return meta
