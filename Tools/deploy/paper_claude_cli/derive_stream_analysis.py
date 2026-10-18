"""Offline collector correction for Claude single-turn segmented CLI streams.

No subprocess, network, credentials, GT access, answer repair or model retry.
Source runs remain immutable. All recorded samples use one parser; only two
known collector event-shape errors can be upgraded to a validated completion.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys

TOOLS = Path(__file__).resolve().parents[2]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import satori_paper_benchmark as bench
from run_paper_parallel import _atomic_json, validate_parallel_run

VERSION = "claude-cli-single-turn-segments-v1"
COLLECTOR_SHA = "3314cf0311519e06c1b458ae119861e2b90c58104770e180892f61d4724a0102"
CLI_BINARY_SHA = "d866d40ef7e477844977793b5a044b006dc340684a11d867234bd1d29377a7a2"
MODEL = "claude-opus-5-5"
STDERR_WITNESS_SHA = "e1ee308a55f0a2279ce8a3117d4a7a36896ff85c81ec9e922438a48951150639"
UPGRADABLE_ERRORS = {"cli_unexpected_system_event", "cli_multiple_or_missing_model_calls"}
RETRY_CONTROLS = {"CLAUDE_CODE_MAX_RETRIES": "0", "CLAUDE_CODE_NONSTREAMING_TIMEOUT_RETRIES": "0",
                  "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK": "1", "CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK": "1",
                  "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1", "CLAUDE_CODE_RETRY_WATCHDOG": "removed"}
RULES = [
    "Post-collection correction of CLI event parsing; no new model calls and no answer text repair.",
    "Apply the identical parser to every recorded success and every eligible collector failure, without reading GT.",
    "Allow system thinking_tokens and ordered assistant thinking/redacted_thinking/text segments for one final model turn.",
    "Concatenated assistant text must equal final result.result byte-for-byte as UTF-8; strict JSON/schema unchanged.",
    "Only the two declared collector event-shape failures may be upgraded. All other source errors stay failed.",
    "Require known preflight stderr witness, disabled retry/fallback controls, exact model, one final end_turn and budget <=4096.",
    "Hash and retain the complete source run, original collector/protocol identities and raw-stream/body hashes.",
]


class StreamRejected(ValueError):
    pass


def _records(path):
    return [bench._strict_response_json(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _tree_hashes(root):
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Evidence directories cannot contain symlinks")
        if path.is_file():
            hashes[path.relative_to(root).as_posix()] = bench._sha(path.read_bytes())
    return hashes


def _load_finalized(root):
    meta, _ = bench._read(root / "run.json")
    if meta.get("status") not in ("completed", "aborted", "failed"):
        raise ValueError("Source must be finalized before offline derivation")
    validate_parallel_run(root)
    protocol, _ = bench._read(root / "protocol.json")
    if (meta.get("model") != MODEL or meta.get("expected_response_model") != MODEL
            or meta.get("endpoint") != "cli://reclaude/" + MODEL
            or meta.get("generation_parameters") != {"max_tokens": 4096, "effort": "high"}
            or protocol.get("provider_sha256") != COLLECTOR_SHA):
        raise ValueError("Source must use the identified frozen Claude collector contract")
    return meta, protocol


def parse_recorded_turn(trace, sample, taxonomy):
    """Parse already recorded sanitized events without reconstructing an answer."""
    required = {"cli_invocations": 1, "cli_exit_code": 0, "cli_binary_sha256": CLI_BINARY_SHA,
                "cli_stderr_sha256": STDERR_WITNESS_SHA, "cli_stderr_bytes": 16,
                "requested_model": MODEL, "expected_response_model": MODEL,
                "generation_parameters": {"max_tokens": 4096, "effort": "high"},
                "transport": "claude_code_cli"}
    if any(trace.get(key) != value for key, value in required.items()):
        raise StreamRejected("unknown_transport_model_budget_or_stderr")
    if trace.get("retry_controls") != RETRY_CONTROLS:
        raise StreamRejected("unverified_retry_controls")
    for field in ("cli_stdout_sha256", "cli_stdin_sha256"):
        if not isinstance(trace.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", trace[field]):
            raise StreamRejected("missing_original_stream_hash")
    events = trace.get("cli_events_redacted")
    if not isinstance(events, list) or not events or any(not isinstance(e, dict) for e in events):
        raise StreamRejected("missing_recorded_events")
    init, assistants, final = [], [], []
    for event in events:
        kind, subtype = event.get("type"), event.get("subtype")
        if re.search(r"retry|fallback|tool|compact", str(kind) + ":" + str(subtype), re.I):
            raise StreamRejected("retry_tool_or_fallback_event")
        if kind == "system":
            if subtype not in ("init", "thinking_tokens"):
                raise StreamRejected("unknown_system_event")
            if event.get("tool_count") != 0 or event.get("mcp_server_count") != 0:
                raise StreamRejected("tools_or_mcp_enabled")
            if subtype == "init":
                init.append(event)
                if event.get("model") != MODEL:
                    raise StreamRejected("init_model_mismatch")
            elif event.get("model") not in (None, MODEL):
                raise StreamRejected("thinking_model_mismatch")
        elif kind == "assistant":
            assistants.append(event)
        elif kind == "result":
            final.append(event)
        elif kind == "rate_limit_event":
            if subtype is not None:
                raise StreamRejected("unknown_rate_limit_event")
        else:
            raise StreamRejected("unknown_event_type")
    if len(init) != 1 or events[0] is not init[0] or len(final) != 1 or events[-1] is not final[0] or not assistants:
        raise StreamRejected("invalid_init_or_final_turn_count")
    result = final[0]
    if (result.get("subtype") != "success" or result.get("is_error") is not False
            or result.get("num_turns") != 1 or result.get("stop_reason") != "end_turn"):
        raise StreamRejected("noncanonical_or_multiple_turn_result")
    model_usage = result.get("modelUsage")
    if not isinstance(model_usage, dict) or set(model_usage) != {MODEL}:
        raise StreamRejected("multiple_or_unknown_usage_models")
    usage = result.get("usage")
    if (not isinstance(usage, dict) or any(type(usage.get(key)) not in (int, float)
            or not math.isfinite(usage[key]) or usage[key] < 0 for key in ("input_tokens", "output_tokens"))
            or usage["output_tokens"] > 4096):
        raise StreamRejected("invalid_or_excessive_final_usage")
    text_parts = []
    for event in assistants:
        message = event.get("message")
        if not isinstance(message, dict) or message.get("model") != MODEL or message.get("stop_reason") not in (None, "end_turn"):
            raise StreamRejected("assistant_model_or_stop_mismatch")
        parts = message.get("content")
        if not isinstance(parts, list) or not parts:
            raise StreamRejected("empty_assistant_segment")
        for part in parts:
            if not isinstance(part, dict):
                raise StreamRejected("invalid_assistant_content")
            if part.get("type") == "text":
                if set(part) != {"type", "text"} or not isinstance(part["text"], str):
                    raise StreamRejected("invalid_text_segment")
                text_parts.append(part["text"])
            elif part.get("type") in ("thinking", "redacted_thinking"):
                if (set(part) != {"type", "omitted_sha256"} or not isinstance(part["omitted_sha256"], str)
                        or not re.fullmatch(r"[0-9a-f]{64}", part["omitted_sha256"])):
                    raise StreamRejected("unverified_thinking_redaction")
            else:
                raise StreamRejected("tool_or_unknown_assistant_content")
    text = "".join(text_parts)
    if not text or text != result.get("result"):
        raise StreamRejected("assistant_text_differs_from_final_result")
    try:
        raw = bench._strict_response_json(text)
        prediction = bench.validate_prediction(raw, sample, taxonomy)
    except (ValueError, TypeError, KeyError):
        raise StreamRejected("invalid_original_json_or_prediction_schema") from None
    return {"raw_json_text": text, "raw_json": prediction, "returned_model": MODEL,
            "native_stop_reason": "end_turn", "finish_reason": "stop", "usage": deepcopy(usage),
            "cli_model_usage": deepcopy(model_usage), "cli_num_turns": 1,
            "cli_assistant_segment_count": len(assistants), "body_sha256": bench._sha(text.encode("utf-8"))}


def _witness(root):
    meta, _ = _load_finalized(root)
    traces = _records(root / "api-calls.jsonl")
    manifest, _ = bench._read(root / "manifest.json")
    taxonomy, _ = bench._read(root / "taxonomy.json")
    samples = {s["sample_id"]: s for s in manifest["samples"]}
    if meta["status"] != "completed" or len(traces) != 1 or traces[0].get("status") != "success":
        raise ValueError("Stderr witness must be a successful single-sample finalized preflight")
    parse_recorded_turn(traces[0], samples[traces[0]["sample_id"]], taxonomy)
    return {"run_path": str(root), "files_sha256": _tree_hashes(root), "stderr_sha256": STDERR_WITNESS_SHA,
            "stderr_bytes": 16, "evidence_scope": "Exact stderr bytes equal the successful v2 non-GT preflight; raw stderr is not exported."}


def _transform(source):
    manifest, _ = bench._read(source / "manifest.json")
    taxonomy, _ = bench._read(source / "taxonomy.json")
    samples = {s["sample_id"]: s for s in manifest["samples"]}
    predictions = _records(source / "predictions.jsonl")
    source_traces = _records(source / "api-calls.jsonl")
    traces = {t["sample_id"]: deepcopy(t) for t in source_traces}
    decisions = []
    for row in predictions:
        original_row = deepcopy(row)
        trace = traces.get(row["sample_id"])
        decision = {"sample_id": row["sample_id"], "source_prediction": original_row,
                    "source_trace_sha256": bench._sha(bench._json_text(trace).encode()) if trace else None,
                    "source_error_code": trace.get("error_code") if trace else None,
                    "source_status": trace.get("status") if trace else None,
                    "source_stdout_sha256": trace.get("cli_stdout_sha256") if trace else None}
        eligible = trace and (trace["status"] == "success" or trace.get("error_code") in UPGRADABLE_ERRORS)
        if not eligible:
            decision["outcome"] = "preserved_noncollector_failure"
        else:
            try:
                parsed = parse_recorded_turn(trace, samples[row["sample_id"]], taxonomy)
            except StreamRejected as exc:
                decision.update(outcome="rejected_by_canonical_parser", reason=str(exc))
                trace.update(status="error", error_code="offline_stream_rejected:" + str(exc))
                row.update(status="failed", response=None, error_code="OfflineStreamRejected")
            else:
                decision.update(outcome="confirmed_success" if trace["status"] == "success" else "corrected_collector_failure",
                                original_body_sha256=parsed.pop("body_sha256"))
                trace.update(parsed)
                trace.update(status="success", offline_parser_version=VERSION,
                             collector_original_status=decision["source_status"], collector_original_error_code=decision["source_error_code"])
                trace.pop("error_code", None)
                row.update(status="completed", response=deepcopy(parsed["raw_json"]), error_code=None)
        decisions.append(decision)
    return predictions, [traces[t["sample_id"]] for t in source_traces], decisions


def derive_run(source_run, witness_run, output_dir):
    source, witness, output = map(lambda p: Path(p).resolve(), (source_run, witness_run, output_dir))
    if output in (source, witness) or source in output.parents or witness in output.parents:
        raise ValueError("Derived artifacts must be outside immutable source and witness runs")
    source_meta, protocol = _load_finalized(source)
    source_hashes = _tree_hashes(source)
    witness_binding = _witness(witness)
    binding = {"kind": "offline_collector_bug_correction", "parser_version": VERSION,
               "parser_sha256": bench._sha(Path(__file__).read_bytes()), "source_run_path": str(source),
               "source_files_sha256": source_hashes, "collector_provider_sha256": COLLECTOR_SHA,
               "original_collection_protocol_sha256": source_meta["source_protocol_sha256"],
               "stderr_witness": witness_binding, "new_model_calls": 0,
               "post_hoc_parser_correction": True, "upgradable_source_errors": sorted(UPGRADABLE_ERRORS), "rules": RULES}
    protocol = deepcopy(protocol)
    protocol["offline_derivation"] = binding
    output.mkdir(parents=True, exist_ok=False)
    for name in ("manifest", "taxonomy"):
        (output / (name + ".json")).write_bytes((source / (name + ".json")).read_bytes())
    _atomic_json(output / "protocol-source.json", protocol)
    _atomic_json(output / "protocol.json", protocol)
    predictions, traces, decisions = _transform(source)
    for name, rows in (("predictions.jsonl", predictions), ("api-calls.jsonl", traces)):
        (output / name).write_text("".join(bench._json_text(row) + "\n" for row in rows), encoding="utf-8", newline="\n")
    _atomic_json(output / "collector-corrections.json", {"binding": binding, "samples": decisions})
    meta = deepcopy(source_meta)
    meta.update(status="completed", source_protocol_path=str(output / "protocol-source.json"),
                source_protocol_sha256=bench._sha((output / "protocol-source.json").read_bytes()),
                successful_samples=sum(row["status"] == "completed" for row in predictions),
                failed_samples=sum(row["status"] == "failed" for row in predictions),
                offline_derivation=binding, analysis_created_utc=datetime.now(timezone.utc).isoformat(),
                analysis_new_model_calls=0, inherited_source_api_calls=source_meta["api_calls"],
                collector_original_successful_samples=source_meta["successful_samples"],
                collector_original_failed_samples=source_meta["failed_samples"],
                collector_corrected_samples=sum(d["outcome"] == "corrected_collector_failure" for d in decisions),
                canonical_parser_rejected_samples=sum(d["outcome"] == "rejected_by_canonical_parser" for d in decisions))
    for name in ("manifest", "taxonomy", "protocol"):
        meta[name + "_snapshot_sha256"] = bench._sha((output / (name + ".json")).read_bytes())
    meta["corrections_sha256"] = bench._sha((output / "collector-corrections.json").read_bytes())
    _atomic_json(output / "run.json", meta)
    if _tree_hashes(source) != source_hashes or _tree_hashes(witness) != witness_binding["files_sha256"]:
        raise ValueError("Immutable source or witness changed during derivation")
    validate_derived_run(output)
    return meta


def validate_derived_run(output_dir):
    output = Path(output_dir).resolve()
    audit = validate_parallel_run(output)
    meta, _ = bench._read(output / "run.json")
    protocol, _ = bench._read(output / "protocol.json")
    binding = meta.get("offline_derivation")
    if not isinstance(binding, dict) or binding != protocol.get("offline_derivation"):
        raise ValueError("Derived run is missing its bound correction protocol")
    if (binding.get("parser_version") != VERSION or binding.get("parser_sha256") != bench._sha(Path(__file__).read_bytes())
            or binding.get("new_model_calls") != 0 or meta.get("analysis_new_model_calls") != 0):
        raise ValueError("Derived parser identity or zero-new-call declaration changed")
    source, witness = Path(binding["source_run_path"]), Path(binding["stderr_witness"]["run_path"])
    original_meta, _ = _load_finalized(source)
    if _tree_hashes(source) != binding["source_files_sha256"] or _witness(witness) != binding["stderr_witness"]:
        raise ValueError("Immutable source or stderr witness changed")
    predictions, traces, decisions = _transform(source)
    if _records(output / "predictions.jsonl") != predictions or _records(output / "api-calls.jsonl") != traces:
        raise ValueError("Derived records differ from deterministic source re-parsing")
    corrections, correction_hash = bench._read(output / "collector-corrections.json")
    if correction_hash != meta.get("corrections_sha256") or corrections != {"binding": binding, "samples": decisions}:
        raise ValueError("Per-sample correction evidence changed")
    if meta.get("inherited_source_api_calls") != original_meta["api_calls"] or meta["api_calls"] != original_meta["api_calls"]:
        raise ValueError("Derived calls must equal actual source calls with no additions")
    return {**audit, "source_lineage_verified": True, "new_model_calls": 0,
            "collector_corrected_samples": sum(d["outcome"] == "corrected_collector_failure" for d in decisions)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    derive = sub.add_parser("derive")
    derive.add_argument("--source-run", required=True, type=Path)
    derive.add_argument("--witness-run", required=True, type=Path)
    derive.add_argument("--output", required=True, type=Path)
    audit = sub.add_parser("audit")
    audit.add_argument("--run", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "audit":
        result = validate_derived_run(args.run)
    else:
        meta = derive_run(args.source_run, args.witness_run, args.output)
        result = {key: meta[key] for key in ("planned_samples", "successful_samples", "failed_samples",
                  "api_calls", "analysis_new_model_calls", "collector_corrected_samples")}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
