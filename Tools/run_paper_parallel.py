"""Bounded parallel driver for the unchanged frozen paper benchmark provider.

No retry, GT access, response repair, resume or eager hidden request queue.
SIGINT stops submissions and waits for every already submitted call to finish.
The original scorer and comparison tool consume the same run/JSONL contract.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from datetime import datetime, timezone
import getpass
import json
import math
import os
from pathlib import Path
import signal
import threading
import uuid

import satori_paper_benchmark as bench


class TraceContractError(RuntimeError):
    """A provider failed the one-call accounting contract; this run is invalid."""


def _identity(provider):
    return {"model": provider.model, "endpoint": provider.base_url,
            "expected_response_model": getattr(provider, "expected_response_model", provider.model),
            "generation_parameters": deepcopy(provider.generation_parameters),
            "timeout_seconds": provider.timeout, "synthetic": bool(getattr(provider, "synthetic", False)),
            "prompt_sha256": bench._sha(provider.prompt.encode("utf-8"))}


def _check_trace(trace, sample, taxonomy, identity):
    expected = {"requested_model": identity["model"],
                "expected_response_model": identity["expected_response_model"],
                "task_kind": sample["task_kind"],
                "image_sha256": {f["frame_id"]: f["sha256"] for f in sample["frames"]},
                "system_prompt_sha256": identity["prompt_sha256"],
                "payload_sha256": bench._sha(bench._json_text(bench.visible_payload(sample, taxonomy)).encode("utf-8")),
                "generation_parameters": identity["generation_parameters"],
                "timeout_seconds": identity["timeout_seconds"]}
    if any(trace.get(k) != v for k, v in expected.items()) or trace.get("status") not in ("success", "error"):
        raise TraceContractError("Request trace differs from frozen request contract")
    latency = trace.get("latency_ms")
    if type(latency) not in (float, int) or not math.isfinite(latency) or latency < 0:
        raise TraceContractError("Request trace has invalid latency")
    if trace["status"] == "success":
        if trace.get("returned_model") != identity["expected_response_model"]:
            raise TraceContractError("Response model differs from frozen model")
        if "raw_json_text" in trace:
            if bench._strict_response_json(trace["raw_json_text"]) != trace.get("raw_json"):
                raise TraceContractError("Raw response text differs from parsed response")
        elif not identity["synthetic"]:
            raise TraceContractError("Real response is missing original text")
        if not identity["synthetic"] and trace.get("finish_reason") != "stop":
            raise TraceContractError("Successful response is incomplete")
        bench.validate_prediction(trace.get("raw_json"), sample, taxonomy)


def _invoke(provider, source_sample, taxonomy, manifest_path, identity):
    """Catch control exceptions in workers so every future carries its trace."""
    row = {"sample_id": source_sample["sample_id"], "status": "failed", "response": None,
           "error_code": None, "request_attempted": False, "latency_ms": None}
    before = len(provider.traces)
    failure, called = None, False
    try:
        if _identity(provider) != identity:
            raise TraceContractError("Worker provider identity changed after freezing")
        sample = deepcopy(source_sample)
        for frame in sample["frames"]:
            frame["image_path"] = str((manifest_path.parent / frame["image_path"]).resolve())
            if bench._sha(Path(frame["image_path"]).read_bytes()) != frame["sha256"]:
                raise ValueError("Frozen input image changed before request")
        called = True
        value = provider.predict(sample, deepcopy(taxonomy))
        row["response"] = bench.validate_prediction(value, sample, taxonomy)
        row["status"] = "completed"
    except BaseException as exc:
        row.update(response=None, error_code=type(exc).__name__)
        failure = exc
    emitted = deepcopy(provider.traces[before:])
    row["request_attempted"] = bool(emitted)
    latencies = [t.get("latency_ms") for t in emitted]
    valid_latencies = all(type(value) in (float, int) and math.isfinite(value) and value >= 0 for value in latencies)
    row["latency_ms"] = sum(latencies) if emitted and valid_latencies else None
    violation = None
    try:
        if called and len(emitted) != 1:
            raise TraceContractError("Provider must emit exactly one trace for each invocation")
        for trace in emitted:
            _check_trace(trace, source_sample, taxonomy, identity)
        if row["status"] == "completed":
            if not emitted or emitted[0]["status"] != "success" or emitted[0].get("raw_json") != row["response"]:
                raise TraceContractError("Prediction differs from its unique successful trace")
        elif emitted and emitted[0]["status"] == "success":
            raise TraceContractError("Failed prediction retained a successful request trace")
    except Exception as exc:
        violation = TraceContractError("Request accounting or fingerprint validation failed: " + type(exc).__name__)
        row.update(status="failed", response=None, error_code="TraceContractError")
    stop = violation or (failure if isinstance(failure, (KeyboardInterrupt, SystemExit, TraceContractError)) else None)
    # A pre-request input failure must stop this run: its initial frozen audit
    # no longer describes what the remaining calls would receive.
    if failure and not called:
        stop = stop or failure
    return row, emitted, stop, violation is not None


def _atomic_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def validate_parallel_run(output_dir):
    """Reject incomplete request accounting before using the shared auditor."""
    meta, _ = bench._read(Path(output_dir) / "run.json")
    if meta.get("request_accounting_complete") is not True:
        raise TraceContractError("Parallel run request accounting is incomplete")
    return bench.audit_run(output_dir)


def infer_parallel(manifest_path, taxonomy_path, output_dir, provider_factory, *, protocol_path,
                   max_workers=1, progress=None):
    """Freeze all inputs, then issue at most max_workers one-attempt requests.

    provider_factory must return a fresh, equally configured provider per slot.
    All started calls are drained and persisted before an interrupt is raised.
    No GT is accepted. An invalid trace makes the run fail, never silently pass.
    """
    if type(max_workers) is not int or not 1 <= max_workers <= 4:
        raise ValueError("max_workers must be an integer between 1 and 4")
    manifest_path, taxonomy_path, protocol_path, output_dir = map(
        lambda path: Path(path).resolve(), (manifest_path, taxonomy_path, protocol_path, output_dir))
    source_manifest, manifest_hash = bench._read(manifest_path)
    source_taxonomy, taxonomy_hash = bench._read(taxonomy_path)
    protocol, protocol_hash = bench._read(protocol_path)
    if not isinstance(protocol, dict) or not protocol:
        raise ValueError("A frozen protocol object is required before inference")
    taxonomy = bench.validate_taxonomy(source_taxonomy)
    manifest = bench.validate_manifest(source_manifest, taxonomy)
    bench._protocol_input_hashes(protocol, {"source_manifest_sha256": manifest_hash, "source_taxonomy_sha256": taxonomy_hash})
    initial_audit = bench._image_audit(manifest, manifest_path)
    if not initial_audit["passed"]:
        raise ValueError("Frozen image audit failed before any request")
    providers = [provider_factory() for _ in range(max_workers)]
    identity = _identity(providers[0])
    if len({id(p) for p in providers}) != max_workers or any(p.traces or _identity(p) != identity for p in providers):
        raise ValueError("Workers require distinct, fresh, identically configured providers")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "request-records").mkdir()
    for name, value in (("manifest", manifest), ("taxonomy", taxonomy), ("protocol", protocol)):
        _atomic_json(output_dir / (name + ".json"), value)
    meta = {"schema_version": 1, "status": "running", "benchmark_id": manifest["benchmark_id"],
            **identity, "planned_samples": len(manifest["samples"]), "automatic_retry": False,
            "normalization": "lowercase; delete ASCII punctuation; remove English a/an/the; collapse whitespace; no synonyms or number conversion",
            "source_manifest_sha256": manifest_hash, "source_taxonomy_sha256": taxonomy_hash,
            "source_protocol_sha256": protocol_hash, "source_manifest_path": str(manifest_path),
            "source_taxonomy_path": str(taxonomy_path), "source_protocol_path": str(protocol_path),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "input_sha256_audit_before_start": initial_audit, "request_accounting_complete": True,
            "execution": {"runner": "run_paper_parallel", "max_in_flight": max_workers,
                          "submission_policy": "refill_only_after_completed_record_is_persisted",
                          "interruption_policy": "stop_submissions_then_drain_started_requests",
                          "runner_sha256": bench._sha(Path(__file__).read_bytes()),
                          "benchmark_source_sha256": bench._sha(Path(bench.__file__).read_bytes())}}
    for name in ("manifest", "taxonomy", "protocol"):
        meta[name + "_snapshot_sha256"] = bench._sha((output_dir / (name + ".json")).read_bytes())
    fingerprint = {name: meta[name] for name in ("model", "endpoint", "expected_response_model", "generation_parameters",
                   "timeout_seconds", "prompt_sha256", "source_manifest_sha256", "source_taxonomy_sha256",
                   "source_protocol_sha256", "execution")}
    meta["run_fingerprint_sha256"] = bench._sha(bench._json_text(fingerprint).encode("utf-8"))
    _atomic_json(output_dir / "run.json", meta)
    stop_event = threading.Event()
    signal_installed = threading.current_thread() is threading.main_thread()
    previous_handler = signal.getsignal(signal.SIGINT) if signal_installed else None
    if signal_installed:
        signal.signal(signal.SIGINT, lambda signum, frame: stop_event.set())
    active, next_index = {}, 0
    completed = failed = attempts = 0
    fatal = None
    executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="paper-once")
    try:
        with (output_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as predictions, \
             (output_dir / "api-calls.jsonl").open("w", encoding="utf-8", newline="\n") as calls:
            def submit(slot):
                nonlocal next_index
                sample = manifest["samples"][next_index]
                before = len(providers[slot].traces)
                future = executor.submit(_invoke, providers[slot], sample, taxonomy, manifest_path, identity)
                active[future] = (slot, sample, before)
                next_index += 1

            for slot in range(min(max_workers, len(manifest["samples"]))):
                if stop_event.is_set():
                    break
                submit(slot)
            while active:
                try:
                    done, _ = wait(active, timeout=0.2, return_when=FIRST_COMPLETED)
                except BaseException as exc:
                    fatal = fatal or exc
                    stop_event.set()
                    continue
                for future in done:
                    slot, sample, before = active[future]
                    try:
                        row, emitted, worker_stop, violation = future.result()
                    except BaseException as exc:
                        # Even a driver bug after a request must retain that
                        # worker's traces and drain the other started requests.
                        emitted = deepcopy(providers[slot].traces[before:])
                        row = {"sample_id": sample["sample_id"], "status": "failed", "response": None,
                               "error_code": type(exc).__name__, "request_attempted": bool(emitted),
                               "latency_ms": emitted[0].get("latency_ms") if len(emitted) == 1 else None}
                        worker_stop, violation = exc, True
                    # Atomic paired journal precedes the legacy JSONL projections.
                    # It also retains every raw trace if a provider violates its
                    # contract, so the failed run can be diagnosed without retry.
                    try:
                        _atomic_json(output_dir / "request-records" / (row["sample_id"] + ".json"),
                                     {"prediction": row, "api_calls": emitted})
                        for trace in emitted:
                            calls.write(bench._json_text({"sample_id": row["sample_id"], **trace}) + "\n")
                        calls.flush()
                        os.fsync(calls.fileno())
                        predictions.write(bench._json_text(row) + "\n")
                        predictions.flush()
                        os.fsync(predictions.fileno())
                    except BaseException as exc:
                        # Physical storage failure cannot guarantee durability.
                        # Stop issuing calls, keep draining/journaling other
                        # results, and explicitly refuse downstream scoring.
                        meta["request_accounting_complete"] = False
                        meta.setdefault("persistence_failures", []).append(
                            {"sample_id": row["sample_id"], "error_type": type(exc).__name__})
                        fatal = fatal or exc
                        stop_event.set()
                    del active[future]
                    completed += row["status"] == "completed"
                    failed += row["status"] == "failed"
                    attempts += len(emitted)
                    if violation:
                        meta["request_accounting_complete"] = False
                    if worker_stop:
                        fatal = fatal or worker_stop
                        stop_event.set()
                    if progress and not stop_event.is_set():
                        try:
                            progress({"sample_id": row["sample_id"], "status": row["status"],
                                      "processed": completed + failed, "planned": len(manifest["samples"])})
                        except BaseException as exc:
                            fatal = fatal or exc
                            stop_event.set()
                    if not stop_event.is_set() and next_index < len(manifest["samples"]):
                        try:
                            submit(slot)
                        except BaseException as exc:
                            fatal = fatal or exc
                            stop_event.set()
            if stop_event.is_set() and fatal is None:
                fatal = KeyboardInterrupt()
        if fatal:
            raise fatal
        meta["status"] = "completed"
    except KeyboardInterrupt:
        meta["status"] = "aborted"
        raise
    except BaseException:
        meta["status"] = "failed"
        raise
    finally:
        executor.shutdown(wait=True)
        meta.update(finished_utc=datetime.now(timezone.utc).isoformat(), successful_samples=completed,
                    failed_samples=failed, processed_samples=completed + failed, api_calls=attempts,
                    not_run_samples=len(manifest["samples"]) - completed - failed,
                    input_sha256_audit_at_finish=bench._image_audit(manifest, manifest_path))
        _atomic_json(output_dir / "run.json", meta)
        if signal_installed:
            signal.signal(signal.SIGINT, previous_handler)
    validate_parallel_run(output_dir)
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "taxonomy", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--expected-response-model")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--api-key-env", default="SATORI_API_KEY")
    parser.add_argument("--prompt-for-key", action="store_true")
    args = parser.parse_args(argv)
    process_env = "SATORI_PARALLEL_" + uuid.uuid4().hex.upper()
    try:
        key = getpass.getpass("API credential: ") if args.prompt_for_key else os.environ.get(args.api_key_env)
        if not key or not key.isascii() or any(c.isspace() for c in key):
            raise ValueError("A valid process credential is required")
        os.environ[process_env] = key
        def factory():
            return bench.PaperBenchmarkProvider(model=args.model, base_url=args.base_url, api_key_env=process_env,
                timeout=args.timeout, max_output_tokens=args.max_output_tokens, temperature=args.temperature,
                expected_response_model=args.expected_response_model)
        result = infer_parallel(args.manifest, args.taxonomy, args.output, factory, protocol_path=args.protocol,
                                max_workers=args.max_workers, progress=lambda r: print(json.dumps(r), flush=True))
        print(json.dumps({k: result[k] for k in ("status", "planned_samples", "successful_samples", "failed_samples", "api_calls")}))
        return 0 if result["failed_samples"] == 0 else 2
    except KeyboardInterrupt:
        print(json.dumps({"status": "aborted", "started_requests_drained": True}), flush=True)
        return 130
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}), flush=True)
        return 2
    finally:
        os.environ.pop(process_env, None)


if __name__ == "__main__":
    raise SystemExit(main())
