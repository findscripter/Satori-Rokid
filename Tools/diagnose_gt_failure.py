"""Diagnose exactly one historical eight-frame GT request without retrying.

Only structured HTTP failure codes, finish reasons, numeric token usage and
request fingerprints are retained. Raw error text, headers, credentials and
image data URLs never enter diagnostic artifacts. Old benchmark runs are read-only.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import gt_benchmark as benchmark
from unity_mage_request import PROJECT_ROOT, ensure_service, fetch_service_token

ENDPOINT = "http://127.0.0.1:18866/v1"
_MODEL = "Mage-VL"


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path):
    raw = Path(path).read_bytes()
    return benchmark._json_loads(raw.decode("utf-8-sig")), _hash(raw)


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare_source(source_run, manifest_path, sample_id="e003"):
    source_run, manifest_path = Path(source_run).resolve(), Path(manifest_path).resolve()
    meta, meta_hash = _read(source_run / "run.json")
    manifest_data, snapshot_hash = _read(source_run / "manifest.json")
    taxonomy_data, taxonomy_hash = _read(source_run / "taxonomy.json")
    original_data, original_hash = _read(manifest_path)
    if (meta.get("synthetic") is not False or meta.get("model") != _MODEL or meta.get("endpoint") != ENDPOINT
            or meta.get("source_manifest_sha256") != original_hash
            or meta.get("manifest_snapshot_sha256") != snapshot_hash
            or meta.get("taxonomy_snapshot_sha256") != taxonomy_hash):
        raise ValueError("Source benchmark identity or frozen input hashes do not match")
    manifest = benchmark.validate_manifest(manifest_data, frame_count=8)
    if benchmark.validate_manifest(original_data, frame_count=8) != manifest:
        raise ValueError("Original unlabeled manifest differs from the saved snapshot")
    taxonomy = benchmark.validate_taxonomy(taxonomy_data)
    matches = [sample for sample in manifest["samples"] if sample["sample_id"] == sample_id]
    if len(matches) != 1:
        raise ValueError("Sample must occur exactly once in the original manifest")
    trace_path = source_run / "api-calls.jsonl"
    trace_bytes = trace_path.read_bytes()
    original_calls = [benchmark._json_loads(line) for line in trace_bytes.decode("utf-8-sig").splitlines() if line.strip()]
    old_calls = [entry for entry in original_calls if entry.get("sample_id") == sample_id]
    if len(old_calls) != 1 or old_calls[0].get("http_status") != 502 or old_calls[0].get("status") != "error":
        raise ValueError("Diagnostic source must contain exactly one HTTP 502 call for this sample")
    original = old_calls[0]
    frames = deepcopy(matches[0]["frames"])
    parts, image_hashes = [], {}
    for frame in frames:
        frame["image_path"] = str((manifest_path.parent / frame["image_path"]).resolve())
        image_bytes = Path(frame["image_path"]).read_bytes()
        digest = _hash(image_bytes)
        if digest != frame["sha256"]:
            raise ValueError("Original image bytes changed")
        image_hashes[frame["frame_id"]] = digest
        parts.extend([{"type": "text", "text": f"Frame ID: {frame['frame_id']}; timestamp_ms: {frame['timestamp_ms']}"},
                      {"type": "image_url", "image_url": {"url": benchmark._image_data_url(frame["image_path"])}}])
    visible = {"scene_candidates": taxonomy["scenes"], "action_candidates": taxonomy["actions"],
               "frames": [{"frame_id": f["frame_id"], "timestamp_ms": f["timestamp_ms"]} for f in frames]}
    context = benchmark._json_text(visible)
    prompt_hash, payload_hash = _hash(benchmark.PROMPT.encode("utf-8")), _hash(context.encode("utf-8"))
    fingerprints = {"requested_model": _MODEL, "image_ids": [f["frame_id"] for f in frames],
                    "image_sha256": image_hashes, "system_prompt_sha256": prompt_hash,
                    "payload_sha256": payload_hash}
    if meta.get("prompt_sha256") != prompt_hash or any(original.get(key) != value for key, value in fingerprints.items()):
        raise ValueError("Original image, prompt, model or payload fingerprint does not match")
    expected_body = {"model": _MODEL, "stream": False, "response_format": {"type": "json_object"},
                     "messages": [{"role": "system", "content": benchmark.PROMPT},
                                  {"role": "user", "content": [{"type": "text", "text": context}, *parts]}]}
    expected_wire = benchmark._json_text(expected_body).encode("utf-8")
    source_hashes = {"run.json": meta_hash, "manifest.json": snapshot_hash,
                     "taxonomy.json": taxonomy_hash, "api-calls.jsonl": _hash(trace_bytes)}
    return frames, taxonomy, fingerprints, _hash(expected_wire), source_hashes


class _ExactRequestOpener:
    def __init__(self, delegate, expected_wire_hash, counter):
        self.delegate, self.expected_wire_hash, self.counter = delegate, expected_wire_hash, counter

    def open(self, req, **kwargs):
        if (req.get_method() != "POST" or req.full_url != ENDPOINT + "/chat/completions"
                or _hash(req.data) != self.expected_wire_hash):
            raise ValueError("Reconstructed HTTP request differs from the original eight-frame format")
        if self.counter["attempts"]:
            raise RuntimeError("Diagnostic forbids a second model request")
        self.counter.update(attempts=1, verified=True)
        return self.delegate.open(req, **kwargs)


def _usage(value):
    if not isinstance(value, dict):
        return None
    selected = {key: item for key, item in value.items()
                if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
                and type(item) is int and item >= 0}
    return selected or None


def _safe_trace(trace):
    fields = {"stage", "status", "requested_model", "returned_model", "http_status", "error_code",
              "latency_ms", "image_ids", "image_sha256", "system_prompt_sha256", "payload_sha256",
              "server_error_code", "server_finish_reason", "error_body_status"}
    result = {key: deepcopy(value) for key, value in trace.items() if key in fields}
    result["server_usage"] = _usage(trace.get("server_usage"))
    result["usage"] = _usage(trace.get("usage"))
    return result


def diagnose(source_run, manifest_path, output_dir, *, sample_id="e003"):
    frames, taxonomy, fingerprints, wire_hash, source_hashes = prepare_source(source_run, manifest_path, sample_id)
    source_run, output_dir = Path(source_run).resolve(), Path(output_dir).resolve()
    if output_dir == source_run or output_dir.is_relative_to(source_run):
        raise ValueError("Diagnostic output must be a separate new run")
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "sample_id": sample_id, "source_run": str(source_run),
              "source_hashes": source_hashes, "frame_count": 8, "model": _MODEL, "endpoint": ENDPOINT,
              "original_fingerprints_match": True, "reconstructed_request_sha256": wire_hash,
              "wire_request_verified": False, "model_requests": 0, "automatic_retry": False,
              "excluded_from_benchmark_scores": True, "started_utc": datetime.now(timezone.utc).isoformat()}
    _write(output_dir / "run.json", result)
    _write(output_dir / "input-fingerprints.json", fingerprints)
    provider, env_name = None, "SATORI_GT_DIAG_" + uuid.uuid4().hex.upper()
    counter = {"attempts": 0, "verified": False}
    original_opener = benchmark.request.build_opener
    traces = []
    try:
        ensure_service(PROJECT_ROOT)
        if env_name in os.environ:
            raise ValueError("Temporary credential environment already exists")
        os.environ[env_name] = fetch_service_token()
        provider = benchmark.BenchmarkProvider(model=_MODEL, base_url=ENDPOINT, api_key_env=env_name,
                                               timeout=180, frame_count=8)
        def guarded_opener(*handlers):
            return _ExactRequestOpener(original_opener(*handlers), wire_hash, counter)
        benchmark.request.build_opener = guarded_opener
        try:
            provider.classify(frames, taxonomy)
            result["diagnosis"] = "failure_not_reproduced"
        except Exception as exc:
            result["exception_type"] = type(exc).__name__
            result["diagnosis"] = "unclassified_failure"
        traces = [_safe_trace(trace) for trace in provider.traces]
        if len(traces) != 1 or counter["attempts"] != 1:
            raise ValueError("Expected exactly one model request and trace")
        trace = traces[0]
        if any(trace.get(key) != value for key, value in fingerprints.items()):
            raise ValueError("Executed request trace does not match the original fingerprints")
        result.update({key: trace.get(key) for key in ("http_status", "server_error_code", "server_finish_reason",
                                                      "server_usage", "error_body_status", "latency_ms")})
        if trace.get("status") == "success":
            result.update(http_status=200, diagnosis="failure_not_reproduced")
        elif trace.get("server_error_code") == "incomplete_json" and trace.get("server_finish_reason") == "length":
            result["diagnosis"] = "output_token_limit"
        elif trace.get("server_error_code") == "invalid_json_output":
            result["diagnosis"] = "invalid_json_output"
        elif trace.get("http_status"):
            result["diagnosis"] = "unclassified_http_failure"
        elif trace.get("error_code") == "invalid_complete_prediction":
            result["diagnosis"] = "client_schema_rejection"
        result["status"] = "completed"
    except Exception as exc:
        result.update(status="failed", exception_type=type(exc).__name__)
    finally:
        benchmark.request.build_opener = original_opener
        os.environ.pop(env_name, None)
        result.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                      model_requests=counter["attempts"], wire_request_verified=counter["verified"],
                      source_run_unchanged=all(_hash((source_run / name).read_bytes()) == digest
                                               for name, digest in source_hashes.items()))
        _write(output_dir / "api-call-summary.json", traces)
        _write(output_dir / "diagnostic.json", result)
        _write(output_dir / "run.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="One exact historical GT failure request; no retry or repair")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", default="e003")
    args = parser.parse_args(argv)
    try:
        result = diagnose(args.source_run, args.manifest, args.output, sample_id=args.sample)
        print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)
        return 0 if result["status"] == "completed" else 2
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "failed", "exception_type": type(exc).__name__}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
