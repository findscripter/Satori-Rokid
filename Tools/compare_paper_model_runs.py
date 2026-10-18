"""Read-only cross-machine audit and independent scoring of paper-data probes.

No network, model, credential or GPU access. Explicit local dataset mapping
replaces remote source path prefixes, while preserving their frozen file hashes.
Existing metrics files are never trusted or modified.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import satori_paper_benchmark as bench


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _load(path):
    raw = Path(path).read_bytes()
    return bench._strict_response_json(raw.decode("utf-8-sig")), _hash(raw)


def _records(path):
    return [bench._strict_response_json(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def _finite_latency(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _utc_timestamp(value, field):
    if not isinstance(value, str):
        raise ValueError(f"GT witness requires a recorded {field} timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"Invalid GT witness {field} timestamp") from None
    if parsed.tzinfo is None:
        raise ValueError(f"GT witness {field} timestamp must have a timezone")
    return parsed.astimezone(timezone.utc)


def _verify_gt_witness(run_dir, data_dir, meta, gt_hash, witness_run):
    """Explicit exception for a missing per-run binding, never a wrong binding.

    The witness is audited with the default strict contract, so exceptions cannot
    be chained. Its source bytes and finished time must precede the target run.
    """
    witness_run = Path(witness_run)
    if witness_run.resolve() == Path(run_dir).resolve():
        raise ValueError("A run cannot be its own GT witness")
    witness = audit_and_score(witness_run, data_dir)
    witness_meta, _ = _load(witness_run / "run.json")
    if witness_meta["status"] != "completed" or witness["not_run_samples"] != 0:
        raise ValueError("GT witness must be a fully completed prior run")
    started = _utc_timestamp(meta.get("started_utc"), "target started_utc")
    finished = _utc_timestamp(witness_meta.get("finished_utc"), "witness finished_utc")
    if finished >= started:
        raise ValueError("GT witness must have finished before the target run started")
    expected = {"manifest_sha256": meta["source_manifest_sha256"],
                "taxonomy_sha256": meta["source_taxonomy_sha256"], "gt_sha256": gt_hash}
    if any(witness["input_fingerprint"].get(key) != value for key, value in expected.items()):
        raise ValueError("GT witness must bind identical manifest, taxonomy and GT bytes")
    if witness["benchmark_id"] != meta["benchmark_id"]:
        raise ValueError("GT witness benchmark differs from the target run")
    return {"run_dir": str(witness_run.resolve()), "run_id": witness_run.name,
            "source_protocol_path": witness_meta["source_protocol_path"],
            "source_protocol_sha256": witness_meta["source_protocol_sha256"],
            "protocol_source_mapping": witness["audit"]["protocol_source_mapping"],
            "gt_sha256": gt_hash, "manifest_sha256": expected["manifest_sha256"],
            "taxonomy_sha256": expected["taxonomy_sha256"],
            "finished_utc": witness_meta["finished_utc"], "target_started_utc": meta["started_utc"],
            "strict_audit_passed": True,
            "limitation": "Target run omitted its per-run GT hash; a separately audited pre-existing run supplies the binding. Target artifacts were not changed."}


def _execution_contract(meta, prompt_hash, profile_path, expected_profile_sha256):
    """Use an externally pinned profile only; never infer exceptions from a run."""
    endpoint = meta.get("endpoint")
    if not isinstance(endpoint, str):
        raise ValueError("Run requires its actual endpoint")
    if (profile_path is None) != (expected_profile_sha256 is None):
        raise ValueError("Execution profile path and explicit SHA256 must be provided together")
    if profile_path is None:
        if endpoint.startswith("cli://"):
            raise ValueError("CLI execution requires an explicitly pinned execution profile")
        if meta.get("generation_parameters") != {"max_tokens": 4096, "temperature": 0.0}:
            raise ValueError("Generation parameters differ from the frozen comparison contract")
        native = endpoint.startswith("local://")
        if meta.get("timeout_seconds") != (None if native else 180):
            raise ValueError("Run timeout differs from the declared native/HTTP execution contract")
        return ("native_video_tensor" if native else "http_image_urls"), None
    if (not isinstance(expected_profile_sha256, str) or len(expected_profile_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_profile_sha256)):
        raise ValueError("Execution profile requires an explicit lowercase SHA256")
    profile, digest = _load(profile_path)
    if digest != expected_profile_sha256:
        raise ValueError("Execution profile bytes differ from the explicitly pinned SHA256")
    required = {"schema_version", "profile_id", "model", "expected_response_model", "endpoint",
                "generation_parameters", "timeout_seconds", "input_representation", "prompt_sha256"}
    if not isinstance(profile, dict) or not required <= set(profile) or profile["schema_version"] != 1:
        raise ValueError("Execution profile is missing its declared contract fields")
    bench._text(profile["profile_id"], "execution profile ID")
    if (profile["model"] != "claude-opus-5-5" or profile["endpoint"] != "cli://reclaude/claude-opus-5-5"
            or profile["generation_parameters"] != {"max_tokens": 4096, "effort": "high"}
            or profile["timeout_seconds"] != 180 or profile["input_representation"] != "claude_cli_image_blocks"
            or profile["prompt_sha256"] != prompt_hash):
        raise ValueError("Execution profile is outside the explicitly supported Claude high-effort contract")
    for key in ("model", "expected_response_model", "endpoint", "generation_parameters", "timeout_seconds", "prompt_sha256"):
        if meta.get(key) != profile[key]:
            raise ValueError(f"Run {key} differs from the pinned execution profile")
    return profile["input_representation"], {"path": str(Path(profile_path).resolve()), "sha256": digest,
                                             "profile_id": profile["profile_id"], "contract": profile}


def audit_and_score(run_dir, data_dir, *, protocol_source=None, gt_witness_run=None,
                    execution_profile=None, execution_profile_sha256=None):
    """Recompute scores after verifying actual sources, images and raw traces.

    The explicit data_dir must contain the original inference source filenames;
    snapshot names and line endings need not match across platforms.
    """
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    meta, _ = _load(run_dir / "run.json")
    parallel = isinstance(meta.get("execution"), dict) and meta["execution"].get("runner") == "run_paper_parallel"
    if ((parallel or "request_accounting_complete" in meta)
            and meta.get("request_accounting_complete") is not True):
        raise ValueError("Parallel request accounting is incomplete; this run cannot be scored")
    if (meta.get("schema_version") != 1 or meta.get("synthetic") is not False
            or meta.get("status") not in ("completed", "aborted", "failed")
            or meta.get("automatic_retry") is not False):
        raise ValueError("Run must be a finalized real one-attempt experiment")
    prompt_hash = _hash(bench.PROMPT.encode("utf-8"))
    if meta.get("prompt_sha256") != prompt_hash:
        raise ValueError("System prompt differs from the frozen comparison contract")
    input_representation, profile_info = _execution_contract(meta, prompt_hash, execution_profile, execution_profile_sha256)
    endpoint = meta["endpoint"]
    for name in ("model", "expected_response_model"):
        bench._text(meta.get(name), name)
    snapshots, paths = {}, {}
    for name in ("manifest", "taxonomy", "protocol"):
        snapshot, digest = _load(run_dir / (name + ".json"))
        if digest != meta.get(name + "_snapshot_sha256"):
            raise ValueError("Frozen snapshot bytes changed")
        source_path = meta.get("source_" + name + "_path")
        if not isinstance(source_path, str) or not source_path:
            raise ValueError("Missing recorded source path")
        basename = source_path.replace("\\", "/").rsplit("/", 1)[-1]
        if basename in ("", ".", ".."):
            raise ValueError("Invalid source basename")
        mapped_path = Path(protocol_source) if name == "protocol" and protocol_source is not None else data_dir / basename
        source, source_hash = _load(mapped_path)
        if source_hash != meta.get("source_" + name + "_sha256"):
            raise ValueError("Mapped original source bytes changed")
        if source != snapshot:
            raise ValueError("Snapshot semantic content differs from its actual source")
        snapshots[name], paths[name] = snapshot, mapped_path
    if profile_info is not None and snapshots["protocol"].get("execution_profile_sha256") != profile_info["sha256"]:
        raise ValueError("Run protocol does not bind the explicitly pinned execution profile SHA256")
    taxonomy = bench.validate_taxonomy(snapshots["taxonomy"])
    manifest = bench.validate_manifest(snapshots["manifest"], taxonomy)
    if meta.get("benchmark_id") != manifest["benchmark_id"]:
        raise ValueError("Run benchmark identity differs from the actual inputs")
    if any(len(s["frames"]) != 4 for s in manifest["samples"]):
        raise ValueError("Comparison contract requires four frozen input frames")
    bench._protocol_input_hashes(snapshots["protocol"], meta)
    images = bench._image_audit(manifest, paths["manifest"])
    if not images["passed"]:
        raise ValueError("Frozen image bytes differ from the source frame hashes")
    gt, gt_hash = _load(data_dir / "gt.json")
    gt_binding_missing = "source_gt_sha256" not in snapshots["protocol"]
    gt_witness = None
    if gt_binding_missing:
        if gt_witness_run is None:
            raise ValueError("Missing per-run GT binding; an explicit prior strict GT witness is required")
        gt_witness = _verify_gt_witness(run_dir, data_dir, meta, gt_hash, gt_witness_run)
    else:
        if snapshots["protocol"]["source_gt_sha256"] != gt_hash:
            raise ValueError("GT bytes differ from the pre-inference frozen source hash")
        if gt_witness_run is not None:
            raise ValueError("GT witness is allowed only when the per-run GT binding field is missing")
    labels = bench._validate_gt(gt, manifest, taxonomy)
    inputs = {sample["sample_id"]: sample for sample in manifest["samples"]}
    traces, raw_null, raw_null_failed, raw_null_failed_schema = {}, 0, 0, 0
    for trace in _records(run_dir / "api-calls.jsonl"):
        sid = trace.get("sample_id")
        if not isinstance(sid, str) or sid not in inputs or sid in traces:
            raise ValueError("Request sample IDs must be known and unique")
        sample = inputs[sid]
        expected = {"requested_model": meta["model"], "expected_response_model": meta["expected_response_model"],
                    "system_prompt_sha256": prompt_hash,
                    "payload_sha256": _hash(bench._json_text(bench.visible_payload(sample, taxonomy)).encode("utf-8")),
                    "image_sha256": {f["frame_id"]: f["sha256"] for f in sample["frames"]},
                    "generation_parameters": meta["generation_parameters"], "timeout_seconds": meta["timeout_seconds"],
                    "task_kind": sample["task_kind"]}
        if any(trace.get(key) != value for key, value in expected.items()):
            raise ValueError("Request trace differs from frozen model, prompt, input or generation parameters")
        if trace.get("status") not in ("success", "error") or not _finite_latency(trace.get("latency_ms")):
            raise ValueError("Request status or latency is invalid")
        if "raw_json_text" in trace:
            try:
                raw = bench._strict_response_json(trace["raw_json_text"])
            except (ValueError, TypeError):
                raw = None
                if trace["status"] == "success":
                    raise ValueError("Successful trace has invalid original JSON text") from None
            if "raw_json" in trace and raw != trace["raw_json"]:
                raise ValueError("Parsed trace differs from its original response text")
            if isinstance(raw, dict) and "answer" in raw and raw["answer"] is None:
                raw_null += 1
                raw_null_failed += trace["status"] == "error"
                if trace["status"] == "error":
                    try:
                        bench.validate_prediction(raw, sample, taxonomy)
                    except ValueError:
                        raw_null_failed_schema += 1
        if trace["status"] == "success":
            if (trace.get("returned_model") != meta["expected_response_model"] or trace.get("finish_reason") != "stop"
                    or not isinstance(trace.get("raw_json_text"), str)):
                raise ValueError("Successful trace requires the expected model and complete original response")
            bench.validate_prediction(trace.get("raw_json"), sample, taxonomy)
        traces[sid] = trace
    predictions = {}
    for record in _records(run_dir / "predictions.jsonl"):
        bench._fields(record, {"sample_id", "status", "response", "error_code", "request_attempted", "latency_ms"},
                      "prediction record")
        sid = record["sample_id"]
        if not isinstance(sid, str) or sid not in inputs or sid in predictions:
            raise ValueError("Prediction sample IDs must be known and unique")
        trace = traces.get(sid)
        if record["status"] not in ("completed", "failed") or type(record["request_attempted"]) is not bool:
            raise ValueError("Prediction status or request flag is invalid")
        if record["request_attempted"] is not bool(trace):
            raise ValueError("Prediction request flag differs from recorded requests")
        if record["latency_ms"] != (trace["latency_ms"] if trace else None):
            raise ValueError("Prediction latency differs from recorded request")
        if record["status"] == "completed":
            if (not trace or trace["status"] != "success" or record["response"] != trace.get("raw_json")
                    or record["error_code"] is not None):
                raise ValueError("Scored prediction differs from the successful raw response")
            bench.validate_prediction(record["response"], inputs[sid], taxonomy)
        elif record["response"] is not None or (trace and trace["status"] == "success"):
            raise ValueError("Failed prediction cannot retain a scoreable successful response")
        predictions[sid] = record
    if set(traces) - set(predictions):
        raise ValueError("Request has no corresponding prediction")
    counts = {"planned_samples": len(inputs), "processed_samples": len(predictions), "api_calls": len(traces),
              "successful_samples": sum(p["status"] == "completed" for p in predictions.values()),
              "failed_samples": sum(p["status"] == "failed" for p in predictions.values()),
              "not_run_samples": len(inputs) - len(predictions)}
    if any(meta.get(key) != value for key, value in counts.items()):
        raise ValueError("Recorded run counters differ from independent request/prediction counts")
    rows = []
    for sample in manifest["samples"]:
        sid = sample["sample_id"]
        p, target = predictions.get(sid), labels[sid]
        answer = p["response"]["answer"] if p and p["status"] == "completed" else None
        strict = answer is not None and answer in target["answers"]
        correct = strict if sample["candidate_group"] else answer is not None and bench.normalize_answer(answer) in {
            bench.normalize_answer(t) for t in target["answers"]}
        rows.append({"sample_id": sid, "task_kind": sample["task_kind"], "target": target["answers"],
                     "metadata": target["metadata"], "prediction": answer, "correct": bool(correct),
                     "strict_exact_correct": bool(strict), "status": p["status"] if p else "not_run"})
    result = {"schema_version": 1, "benchmark_id": manifest["benchmark_id"], "run_id": run_dir.name,
              "run_dir": str(run_dir.resolve()), "model": meta["model"], "expected_response_model": meta["expected_response_model"],
              "endpoint": endpoint, "generation_parameters": meta["generation_parameters"], "timeout_seconds": meta["timeout_seconds"],
              "input_representation": input_representation, "execution_profile": profile_info, **counts,
              "correct": sum(r["correct"] for r in rows), "accuracy": sum(r["correct"] for r in rows) / len(rows),
              "strict_exact_correct": sum(r["strict_exact_correct"] for r in rows),
              "strict_exact_accuracy": sum(r["strict_exact_correct"] for r in rows) / len(rows),
              "valid_null_abstentions": sum(p["status"] == "completed" and p["response"]["answer"] is None for p in predictions.values()),
              "raw_null_answers_any_status": raw_null, "raw_null_failed_response": raw_null_failed,
              "raw_null_failed_schema": raw_null_failed_schema,
              "error_codes": dict(Counter(t.get("error_code", "unspecified") for t in traces.values() if t["status"] == "error")),
              "http_statuses": dict(Counter(str(t["http_status"]) for t in traces.values() if "http_status" in t)),
              "finish_reasons": dict(Counter(str(t["finish_reason"]) for t in traces.values() if "finish_reason" in t)),
              "audit": {"passed": True, "images": images, "source_mapping": str(data_dir.resolve()),
                        "protocol_source_mapping": str(paths["protocol"].resolve()),
                        "metrics_file_used": False, "per_run_gt_binding_missing": gt_binding_missing,
                        "gt_binding_verification": "prior_strict_run_witness" if gt_binding_missing else "per_run_protocol",
                        "gt_witness": gt_witness}, "paired": rows,
              "input_fingerprint": {"manifest_sha256": meta["source_manifest_sha256"],
                                    "taxonomy_sha256": meta["source_taxonomy_sha256"], "gt_sha256": gt_hash,
                                    "prompt_sha256": prompt_hash, "generation_parameters": meta["generation_parameters"]}}
    result["valid_nonnull_answers"] = counts["successful_samples"] - result["valid_null_abstentions"]
    for key in ("task_kind", "subject", "look_ahead_ms", "category", "type", "semantic", "structural", "reference"):
        grouped = defaultdict(list)
        for row in rows:
            value = row["task_kind"] if key == "task_kind" else row["metadata"].get(key)
            values = value if isinstance(value, list) else [value]
            for group in dict.fromkeys(v for v in values if isinstance(v, (str, int))):
                grouped[str(group)].append(row)
        if grouped:
            result["by_task" if key == "task_kind" else "by_" + key] = {
                name: bench._metrics(group, taxonomy) for name, group in sorted(grouped.items())}
    return result


def compare_results(results, *, comparison_mode="strict"):
    if not results:
        raise ValueError("At least one audited run is required")
    reference = results[0]
    if comparison_mode not in ("strict", "same_observations"):
        raise ValueError("Unknown comparison mode")
    if any(r["input_fingerprint"].get("generation_parameters") != r["generation_parameters"] for r in results):
        raise ValueError("Run generation parameters differ from its audited input fingerprint")
    observation = lambda r: {key: value for key, value in r["input_fingerprint"].items() if key != "generation_parameters"}
    if any(observation(r) != observation(reference) or r["benchmark_id"] != reference["benchmark_id"] for r in results):
        raise ValueError("Models must use the same frozen inputs, GT and prompt")
    generation_equal = all(r["generation_parameters"] == reference["generation_parameters"] for r in results)
    if comparison_mode == "strict" and not generation_equal:
        raise ValueError("Models must use the same frozen inputs, GT, prompt and generation parameters")
    execution_keys = ("generation_parameters", "timeout_seconds", "input_representation")
    execution_equal = all(all(r.get(key) == reference.get(key) for key in execution_keys) for r in results)
    run_ids = [r["run_id"] for r in results]
    if len(set(run_ids)) != len(run_ids):
        # Programmatic callers can use distinct model names to compare independent copies.
        run_ids = [r["model"] for r in results]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Run labels must be unique")
    rows = [{"sample_id": row["sample_id"], "task_kind": row["task_kind"], "target": row["target"],
             "metadata": row["metadata"], "models": {}} for row in reference["paired"]]
    for label, result in zip(run_ids, results):
        if [r["sample_id"] for r in result["paired"]] != [r["sample_id"] for r in rows]:
            raise ValueError("Frozen sample ordering differs")
        for paired, row in zip(rows, result["paired"]):
            if paired["target"] != row["target"] or paired["metadata"] != row["metadata"]:
                raise ValueError("Frozen target or metadata differs")
            paired["models"][label] = {k: row[k] for k in ("prediction", "status", "correct", "strict_exact_correct")}
    return {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
            "benchmark_id": reference["benchmark_id"], "planned_samples": reference["planned_samples"],
            "comparison_mode": comparison_mode, "generation_parameters_equal": generation_equal,
            "execution_parameters_equal": execution_equal,
            "execution_parameter_fields": list(execution_keys),
            "observation_fingerprint": observation(reference),
            "input_fingerprint": reference["input_fingerprint"] if generation_equal else observation(reference),
            "models": [{k: v for k, v in r.items() if k != "paired"} for r in results], "paired": rows,
            "limitations": ["Same observations, different native/HTTP/CLI model preprocessing; no fair speed or compute ranking.",
                            "Cross-provider generation controls differ when generation_parameters_equal is false; no claim of matched decoding conditions.",
                            "Module diagnostic, not the unpublished original Satori evaluation or an independent final test.",
                            "Failed, null, unrun and malformed predictions remain in each fixed denominator."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--protocol-source", action="append", default=[], metavar="RUN_ID=PATH",
                        help="Explicit original protocol source mapping for a run stored outside --data; repeat as needed")
    parser.add_argument("--gt-witness", action="append", default=[], metavar="RUN_ID=WITNESS_RUN",
                        help="Explicit exception for a missing per-run GT hash; witness must be an earlier strictly audited run")
    parser.add_argument("--execution-profile", action="append", nargs=3, default=[], metavar=("RUN_ID", "PATH", "SHA256"),
                        help="Explicit pre-frozen execution profile and externally pinned file hash")
    parser.add_argument("--comparison-mode", choices=("strict", "same_observations"), default="strict")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    protocols = {}
    for mapping in args.protocol_source:
        run_id, separator, path = mapping.partition("=")
        if not separator or not run_id or not path or run_id in protocols or run_id not in {r.name for r in args.run}:
            parser.error("--protocol-source requires a unique selected RUN_ID=PATH")
        protocols[run_id] = Path(path)
    witnesses = {}
    for mapping in args.gt_witness:
        run_id, separator, path = mapping.partition("=")
        if not separator or not run_id or not path or run_id in witnesses or run_id not in {r.name for r in args.run}:
            parser.error("--gt-witness requires a unique selected RUN_ID=WITNESS_RUN")
        witnesses[run_id] = Path(path)
    profiles = {}
    for run_id, path, digest in args.execution_profile:
        if run_id in profiles or run_id not in {r.name for r in args.run}:
            parser.error("--execution-profile requires a unique selected run ID")
        profiles[run_id] = (Path(path), digest)
    result = compare_results([audit_and_score(run, args.data, protocol_source=protocols.get(run.name),
                                              gt_witness_run=witnesses.get(run.name),
                                              execution_profile=profiles.get(run.name, (None, None))[0],
                                              execution_profile_sha256=profiles.get(run.name, (None, None))[1]) for run in args.run],
                             comparison_mode=args.comparison_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"benchmark_id": result["benchmark_id"], "planned_samples": result["planned_samples"],
                      "models": [{k: r[k] for k in ("model", "correct", "accuracy", "strict_exact_correct", "failed_samples",
                                                   "valid_null_abstentions", "raw_null_failed_schema", "not_run_samples")}
                                 for r in result["models"]]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
