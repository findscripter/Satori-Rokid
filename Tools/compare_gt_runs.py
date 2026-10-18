"""Read-only audit of the frozen 32-video Mage/GPT comparison.

Inference, network access, credentials, retries and answer repair are deliberately
absent. Original runs are never modified. Missing or inconsistent evidence is an
error, rather than a reason to omit a sample from the comparison.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import gt_benchmark as benchmark
from summarize_gt_run import evaluate


SAMPLE_COUNT = 32
EXPECTED_GENERATION = {"max_tokens": 4096, "temperature": 0.0}
EXPECTED_TIMEOUT = 180


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return benchmark._json_loads(Path(path).read_text(encoding="utf-8-sig"))


def records(path):
    result = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        require(bool(line.strip()), f"Blank JSONL record: {path}:{number}")
        item = benchmark._json_loads(line)
        require(isinstance(item, dict), f"Non-object JSONL record: {path}:{number}")
        result.append(item)
    return result


def distribution(values):
    for value in values:
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                "Reported usage or latency must be finite and nonnegative")
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "mean": None, "p50": None, "p95": None}

    def quantile(fraction):
        position = (len(ordered) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {"count": len(values), "mean": statistics.fmean(values),
            "p50": quantile(.5), "p95": quantile(.95)}


def usage_summary(traces, field):
    result = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = []
        for trace in traces:
            usage = trace.get(field)
            require(usage is None or isinstance(usage, dict), "Usage must be an object or null")
            if usage is not None and name in usage:
                require(type(usage[name]) is int and usage[name] >= 0,
                        f"Invalid integer usage field: {name}")
                values.append(usage[name])
        result[name] = {**distribution(values), "missing_samples": len(traces) - len(values)}
    return result


def _check_metrics(scores, independent):
    pairs = {
        "scene_top1": scores["scene"]["top1"],
        "action_micro_precision": scores["actions"]["micro_precision"],
        "action_micro_recall": scores["actions"]["micro_recall"],
        "action_micro_f1": scores["actions"]["micro_f1"],
        "action_exact_match": scores["actions"]["exact_match"],
        "action_sample_mean_f1": scores["actions"]["sample_mean_f1"],
        "action_mean_jaccard": scores["actions"]["mean_jaccard"],
    }
    for name, expected in pairs.items():
        require(abs(independent[name] - expected) < 1e-12,
                f"Independent scoring disagreement: {name}")


def audit_run(run_dir, data_dir, manifest, taxonomy, truth):
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    meta = read(run_dir / "run.json")
    require(meta.get("status") == "completed" and meta.get("synthetic") is False,
            f"Run must be completed and real: {run_dir}")
    for name in ("planned_samples", "processed_samples", "api_calls"):
        require(type(meta.get(name)) is int and meta[name] == SAMPLE_COUNT,
                f"Run must retain exactly 32 {name}: {run_dir}")
    require(meta.get("not_run_samples") == 0 and meta.get("automatic_retry") is False,
            f"Not-run records or retries invalidate the frozen comparison: {run_dir}")
    require(meta.get("frame_count") == 8, f"Expected eight frames: {run_dir}")
    require(meta.get("generation_parameters") == EXPECTED_GENERATION,
            f"Generation parameters differ from the frozen protocol: {run_dir}")
    require(meta.get("timeout_seconds") == EXPECTED_TIMEOUT,
            f"Timeout differs from the frozen protocol: {run_dir}")
    require(meta.get("prompt_sha256") == hashlib.sha256(benchmark.PROMPT.encode()).hexdigest(),
            f"System prompt differs from the frozen protocol: {run_dir}")
    for field, filename in (("source_manifest_sha256", "infer-manifest.json"),
                            ("source_taxonomy_sha256", "taxonomy.json")):
        require(meta.get(field) == digest(data_dir / filename), f"Source hash mismatch: {run_dir} {field}")
    require(read(run_dir / "manifest.json") == manifest, f"Manifest content mismatch: {run_dir}")
    require(read(run_dir / "taxonomy.json") == taxonomy, f"Taxonomy content mismatch: {run_dir}")
    require(isinstance(meta.get("model"), str) and bool(meta["model"]), "Missing requested model")
    require(isinstance(meta.get("expected_response_model"), str) and bool(meta["expected_response_model"]),
            "Missing frozen exact response model")
    predictions_list = records(run_dir / "predictions.jsonl")
    traces = records(run_dir / "api-calls.jsonl")
    ids = [sample["sample_id"] for sample in manifest["samples"]]
    for items, name in ((predictions_list, "prediction"), (traces, "request trace")):
        require(len(items) == SAMPLE_COUNT and [item.get("sample_id") for item in items] == ids,
                f"Expected precisely one ordered {name} for each of 32 samples: {run_dir}")
    predictions = {item["sample_id"]: item for item in predictions_list}
    identity_anomalies = []
    for sample, row, trace in zip(manifest["samples"], predictions_list, traces):
        sid = sample["sample_id"]
        require(row.get("request_attempted") is True, f"Request was not attempted: {sid}")
        require(trace.get("stage") == "charades_ego_classification", f"Wrong request stage: {sid}")
        require(trace.get("requested_model") == meta["model"], f"Requested model mismatch: {sid}")
        require(trace.get("expected_response_model") == meta["expected_response_model"],
                f"Frozen response model changed: {sid}")
        returned = trace.get("returned_model")
        require(trace.get("generation_parameters") == meta["generation_parameters"],
                f"Request generation parameters changed: {sid}")
        require(trace.get("timeout_seconds") == meta["timeout_seconds"], f"Request timeout changed: {sid}")
        require(trace.get("system_prompt_sha256") == meta["prompt_sha256"], f"Request prompt changed: {sid}")
        expected_ids = [frame["frame_id"] for frame in sample["frames"]]
        expected_hashes = {frame["frame_id"]: frame["sha256"] for frame in sample["frames"]}
        require(trace.get("image_ids") == expected_ids and trace.get("image_sha256") == expected_hashes,
                f"Request image IDs or bytes changed: {sid}")
        visible = {"scene_candidates": taxonomy["scenes"], "action_candidates": taxonomy["actions"],
                   "frames": [{"frame_id": frame["frame_id"], "timestamp_ms": frame["timestamp_ms"]}
                              for frame in sample["frames"]]}
        payload_hash = hashlib.sha256(benchmark._json_text(visible).encode()).hexdigest()
        require(trace.get("payload_sha256") == payload_hash, f"Request visible context changed: {sid}")
        require(type(trace.get("latency_ms")) in (int, float)
                and math.isfinite(trace["latency_ms"]) and trace["latency_ms"] >= 0
                and row.get("latency_ms") == trace["latency_ms"], f"Trace latency mismatch: {sid}")
        if row.get("status") == "completed":
            require(returned == meta["expected_response_model"],
                    f"Unexpected actual response model in successful prediction: {sid}")
            require(trace.get("status") == "success" and returned == meta["expected_response_model"]
                    and trace.get("response_model_matches") is True and trace.get("finish_reason") == "stop"
                    and trace.get("response_refused") is False and not trace.get("response_issues"),
                    f"Successful prediction lacks a matching complete response: {sid}")
            require(trace.get("raw_json") == row.get("response"), f"Scored answer differs from raw response: {sid}")
            require(isinstance(trace.get("raw_json_text"), str)
                    and benchmark._json_loads(trace["raw_json_text"]) == row["response"],
                    f"Raw JSON text differs from scored prediction: {sid}")
        else:
            require(row.get("status") == "failed" and trace.get("status") == "error"
                    and row.get("response") is None and isinstance(trace.get("error_code"), str),
                    f"Failed prediction lacks an error trace: {sid}")
            issues = trace.get("response_issues", [])
            require(isinstance(issues, list), f"Malformed failed-response diagnostics: {sid}")
            identity_mismatch = returned is not None and returned != meta["expected_response_model"]
            if identity_mismatch or "unexpected_model" in issues:
                require(trace.get("error_code") == "invalid_complete_prediction"
                        and "unexpected_model" in issues and trace.get("response_model_matches") is False,
                        f"Unexpected actual response model lacks explicit failed-response identity diagnostics: {sid}")
                identity_anomalies.append({"sample_id": sid, "expected_response_model": meta["expected_response_model"],
                                           "returned_model": returned, "error_code": trace["error_code"],
                                           "response_issues": issues, "scored_status": "failed"})
            elif returned == meta["expected_response_model"]:
                require(trace.get("response_model_matches") is True,
                        f"Failed-response identity flag disagrees with actual response model: {sid}")
    completed = sum(row["status"] == "completed" for row in predictions_list)
    require(meta.get("successful_samples") == completed
            and meta.get("failed_samples") == SAMPLE_COUNT - completed, "Run coverage counters disagree")

    # score validates snapshot checksums, GT alignment and every prediction schema.
    scores = benchmark.score(run_dir, data_dir / "gt.json")
    if (run_dir / "scores.json").exists():
        require(read(run_dir / "scores.json") == scores, f"Saved scores do not match fresh scoring: {run_dir}")
    scene_names = {entry["id"]: entry["label"] for entry in taxonomy["scenes"]}
    action_names = {entry["id"]: entry["label"] for entry in taxonomy["actions"]}
    independent = evaluate(truth["labels"], predictions, scene_names, action_names)
    _check_metrics(scores, independent)
    failures = Counter(trace.get("error_code") for trace in traces if trace["status"] == "error")
    return {"run_directory": str(run_dir.resolve()), "model": meta["model"],
            "expected_response_model": meta["expected_response_model"], "endpoint": meta.get("endpoint"),
            "planned_samples": SAMPLE_COUNT, "completed_samples": completed,
            "failed_samples": SAMPLE_COUNT - completed, "failure_rate": (SAMPLE_COUNT - completed) / SAMPLE_COUNT,
            "scene_abstained_samples": scores["scene"]["abstained_samples"],
            "action_abstained_samples": scores["actions"]["abstained_samples"],
            "metrics": {key: value for key, value in independent.items() if key != "rows"},
            "latency_ms": {"all_attempts": distribution([trace["latency_ms"] for trace in traces]),
                           "completed": distribution([trace["latency_ms"] for trace in traces if trace["status"] == "success"]),
                           "failed": distribution([trace["latency_ms"] for trace in traces if trace["status"] == "error"])},
            "usage": {"response_envelope": usage_summary(traces, "usage"),
                      "failed_server_metadata": usage_summary(traces, "server_usage")},
            "failure_codes": dict(failures), "identity_anomalies": identity_anomalies, "scores": scores,
            "artifact_sha256": {name: digest(run_dir / name) for name in
                                ("run.json", "manifest.json", "taxonomy.json", "predictions.jsonl", "api-calls.jsonl")},
            "rows": independent["rows"]}, predictions, traces, meta


def labelled(ids, names):
    return [{"id": key, "label": names[key]} for key in sorted(ids)]


def verify_freeze(freeze_path, data, runs, metadata):
    """Verify the immutable pre-inference contract, without editing any evidence."""
    freeze_path = Path(freeze_path).resolve()
    freeze = read(freeze_path)
    require(freeze.get("stage") == "before_formal_inference", "Freeze is not a pre-inference contract")
    for field, expected in (("planned_samples_per_model", 32), ("frames_per_sample", 8),
                            ("attempts_per_sample", 1), ("generation_parameters", EXPECTED_GENERATION),
                            ("timeout_seconds", EXPECTED_TIMEOUT), ("automatic_retry", False)):
        require(freeze.get(field) == expected, f"Freeze protocol field mismatch: {field}")
    required_data = {"infer-manifest.json", "taxonomy.json", "gt.json", "baseline-protocol.json"}
    require(isinstance(freeze.get("data_sha256"), dict) and set(freeze["data_sha256"]) == required_data,
            "Freeze must contain every required data hash")
    for name, expected in freeze["data_sha256"].items():
        require(digest(Path(data) / name) == expected, f"Frozen data changed: {name}")
    project_root = Path(__file__).resolve().parents[1]
    required_code = {"Tools/gt_benchmark.py", "Tools/run_mage_gt.py", "Tools/run_gpt_gt.py"}
    require(isinstance(freeze.get("code_sha256"), dict) and set(freeze["code_sha256"]) == required_code,
            "Freeze must contain exactly the three inference code hashes")
    for name, expected in freeze["code_sha256"].items():
        require(digest(project_root / name) == expected, f"Frozen inference code changed: {name}")
    prompt_hash = hashlib.sha256(benchmark.PROMPT.encode()).hexdigest()
    require(freeze.get("prompt_sha256") == prompt_hash, "Frozen prompt changed")
    date = freeze_path.stem.rsplit("-", 1)[-1]
    require(re.fullmatch(r"\d{8}", date) is not None, "Freeze filename must end in YYYYMMDD.json")
    protocol_path = freeze_path.with_name(f"MAGE-GPT-COMPARISON-PROTOCOL-{date}.md")
    require(digest(protocol_path) == freeze.get("protocol_sha256"), "Frozen written protocol changed")
    frozen_time = datetime.fromisoformat(freeze["frozen_utc"])
    require(frozen_time.tzinfo is not None, "Freeze timestamp must include timezone")
    for name in ("mage", "gpt"):
        frozen_run = Path(freeze["formal_runs"][name])
        if not frozen_run.is_absolute():
            frozen_run = project_root / frozen_run
        require(frozen_run.resolve() == Path(runs[name]).resolve(), f"Formal run path differs from freeze: {name}")
        expected_model = freeze["smokes"][name]["returned_model"]
        require(metadata[name]["expected_response_model"] == expected_model
                and metadata[name]["model"] == expected_model, f"Exact model ID differs from freeze: {name}")
        require(metadata[name]["prompt_sha256"] == prompt_hash, f"Run prompt differs from freeze: {name}")
        start_time = datetime.fromisoformat(metadata[name]["started_utc"])
        require(start_time.tzinfo is not None and start_time >= frozen_time,
                f"Run was not started after freeze: {name}")
    return {"path": str(freeze_path), "sha256": digest(freeze_path), "frozen_utc": freeze["frozen_utc"],
            "protocol_path": str(protocol_path), "protocol_sha256": freeze["protocol_sha256"],
            "inference_code_sha256": freeze["code_sha256"], "data_sha256": freeze["data_sha256"]}


def compare(mage_run, gpt_run, data, *, freeze=None):
    data = Path(data)
    manifest = benchmark.validate_manifest(read(data / "infer-manifest.json"), frame_count=8)
    taxonomy = benchmark.validate_taxonomy(read(data / "taxonomy.json"))
    truth = read(data / "gt.json")
    benchmark._validate_gt(truth, manifest, taxonomy)
    require(len(manifest["samples"]) == SAMPLE_COUNT, "Comparison requires exactly the frozen 32 samples")
    for sample in manifest["samples"]:
        for frame in sample["frames"]:
            require(digest((data / frame["image_path"]).resolve()) == frame["sha256"],
                    f"Source image bytes changed: {sample['sample_id']} {frame['frame_id']}")
    mage, mage_predictions, mage_traces, mage_meta = audit_run(mage_run, data, manifest, taxonomy, truth)
    gpt, gpt_predictions, gpt_traces, gpt_meta = audit_run(gpt_run, data, manifest, taxonomy, truth)
    require(mage_meta["model"] == "Mage-VL" and gpt_meta["model"] != "Mage-VL", "Model roles are not Mage versus GPT")
    same_fields = ("benchmark_id", "protocol", "source_manifest_sha256", "source_taxonomy_sha256",
                   "manifest_snapshot_sha256", "taxonomy_snapshot_sha256", "prompt_sha256", "frame_count",
                   "generation_parameters", "timeout_seconds")
    for field in same_fields:
        require(mage_meta.get(field) == gpt_meta.get(field), f"Cross-run protocol mismatch: {field}")
    freeze_evidence = verify_freeze(freeze, data, {"mage": mage_run, "gpt": gpt_run},
                                    {"mage": mage_meta, "gpt": gpt_meta}) if freeze else None
    scene_names = {entry["id"]: entry["label"] for entry in taxonomy["scenes"]}
    action_names = {entry["id"]: entry["label"] for entry in taxonomy["actions"]}
    paired, wrong_examples = [], []
    rows = [{row["sample_id"]: row for row in model["rows"]} for model in (mage, gpt)]
    traces = [{trace["sample_id"]: trace for trace in collection} for collection in (mage_traces, gpt_traces)]
    for gt in truth["labels"]:
        sid = gt["sample_id"]
        pair = {"sample_id": sid, "source_video_id": gt["source_video_id"],
                "gt_scene": {"id": gt["scene_id"], "label": scene_names[gt["scene_id"]]},
                "gt_actions": labelled(set(gt["action_ids"]), action_names)}
        notes = []
        for index, (name, predictions) in enumerate((("mage", mage_predictions), ("gpt", gpt_predictions))):
            row, prediction, trace = rows[index][sid], predictions[sid], traces[index][sid]
            answer = prediction["response"]
            actual = set(gt["action_ids"])
            predicted = set(answer["action_ids"] or []) if answer else set()
            scene = answer["scene_id"] if answer else None
            pair[name] = {"status": prediction["status"], "scene_id": scene,
                          "scene_label": scene_names.get(scene), "scene_correct": row["scene_correct"],
                          "scene_abstained": bool(answer and scene is None),
                          "action_abstained": bool(answer and answer["action_ids"] is None),
                          "predicted_actions": labelled(predicted, action_names),
                          "missed_actions": labelled(actual - predicted, action_names),
                          "extra_actions": labelled(predicted - actual, action_names),
                          "tp": row["true_positives"], "fp": row["false_positives"], "fn": row["false_negatives"],
                          "action_f1": row["action_f1"], "jaccard": row["jaccard"],
                          "answer": row["answer"], "latency_ms": row["latency_ms"],
                          "error_code": trace.get("error_code"), "http_status": trace.get("http_status"),
                          "expected_response_model": trace.get("expected_response_model"),
                          "returned_model": trace.get("returned_model"), "response_issues": trace.get("response_issues", []),
                          "usage": trace.get("usage"), "server_usage": trace.get("server_usage")}
            if prediction["status"] != "completed":
                notes.append(f"{name} 请求失败（{trace.get('error_code')}），全部 GT 动作计为漏检。")
            else:
                if scene is None:
                    notes.append(f"{name} 对场景弃答。")
                elif not row["scene_correct"]:
                    notes.append(f"{name} 将 {scene_names[gt['scene_id']]} 判为 {scene_names[scene]}。")
                if answer["action_ids"] is None:
                    notes.append(f"{name} 对动作弃答。")
                if actual - predicted or predicted - actual:
                    notes.append(f"{name} 动作命中 {len(actual & predicted)}、漏检 {len(actual - predicted)}、误报 {len(predicted - actual)}。")
        pair["gpt_minus_mage_scene_correct"] = int(pair["gpt"]["scene_correct"]) - int(pair["mage"]["scene_correct"])
        pair["gpt_minus_mage_action_f1"] = pair["gpt"]["action_f1"] - pair["mage"]["action_f1"]
        pair["gpt_minus_mage_latency_ms"] = pair["gpt"]["latency_ms"] - pair["mage"]["latency_ms"]
        pair["中文说明"] = " ".join(notes) if notes else "两边场景与完整 GT 动作集合均匹配。"
        paired.append(pair)
        if notes:
            wrong_examples.append(pair)
    prior = read(data / "baseline-protocol.json")
    scene_ids = {label: sid for sid, label in scene_names.items()}
    baseline_predictions = {gt["sample_id"]: {"status": "completed", "latency_ms": None,
        "response": {"scene_id": scene_ids[prior["scene_label"]], "action_ids": prior["action_ids"],
                     "answer": "固定训练集频率先验，不调用视觉模型。"}} for gt in truth["labels"]}
    baseline = evaluate(truth["labels"], baseline_predictions, scene_names, action_names)
    baseline.pop("rows")
    metric_fields = ("scene_correct", "scene_top1", "action_tp", "action_fp", "action_fn", "action_micro_precision",
                     "action_micro_recall", "action_micro_f1", "action_exact_match", "action_sample_mean_f1", "action_mean_jaccard")
    return {"success": True, "audit_passed": True, "freeze_verified": freeze_evidence is not None,
            "freeze_evidence": freeze_evidence, "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": {field: mage_meta[field] for field in same_fields},
            "source_gt_sha256": digest(data / "gt.json"), "fixed_denominator": SAMPLE_COUNT,
            "audit_checks": ["32 planned, attempted and scored samples on both sides; exactly one trace each",
                             "source/snapshot/prompt/context/image-byte hashes match",
                             "4096 max_tokens, temperature 0, timeout 180; no automatic retry",
                             "successful model equals each exact ID; diagnosed identity mismatch retained as failed sample",
                             "raw successful answer equals scored answer",
                             "two independent scorers agree; failures and abstentions remain in fixed denominator"],
            "mage": mage, "gpt": gpt,
            "gpt_minus_mage": {name: gpt["metrics"][name] - mage["metrics"][name] for name in metric_fields},
            "paired_outcomes": {"scene_gpt_only_correct": sum(row["gpt_minus_mage_scene_correct"] == 1 for row in paired),
                                "scene_mage_only_correct": sum(row["gpt_minus_mage_scene_correct"] == -1 for row in paired),
                                "action_f1_gpt_better": sum(row["gpt_minus_mage_action_f1"] > 1e-12 for row in paired),
                                "action_f1_mage_better": sum(row["gpt_minus_mage_action_f1"] < -1e-12 for row in paired),
                                "action_f1_tied": sum(abs(row["gpt_minus_mage_action_f1"]) <= 1e-12 for row in paired)},
            "fixed_training_prior_baseline": {"protocol": prior, "metrics": baseline},
            "paired_samples": paired, "wrong_examples": wrong_examples,
            "limitations": ["32 段已观察过结果的开发诊断集，不能据此推断完整官方测试集性能。",
                            "整片 GT 包含未必被八张稀疏帧覆盖的动作；当前指标不是官方 mAP。",
                            "GPT 为第三方 API 的请求/返回标识，未独立验证其底层模型身份。",
                            "延迟是含传输与失败等待的接口端到端时间；tokenizer 不同，token 数不直接等价。",
                            "usage 只统计服务器实际报告值；缺失不填零，失败附带的 server_usage 单列。"]}


def _write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def save(report, output):
    output = Path(output)
    files = (output, output.with_suffix(".paired.csv"), output.with_suffix(".metrics.csv"),
             output.with_suffix(".wrong-examples.json"))
    require(len(set(files)) == 4, "Choose an output JSON filename distinct from generated sidecar paths")
    require(not any(path.exists() for path in files), "Comparison output already exists; refusing replacement")
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = []
    for name in ("mage", "gpt"):
        model = report[name]
        metrics.append({"provider": name, "model": model["model"], "response_model": model["expected_response_model"],
                        **{key: value for key, value in model["metrics"].items() if key != "scene_by_gt_class"},
                        "completed_samples": model["completed_samples"], "failed_samples": model["failed_samples"],
                        "scene_abstained_samples": model["scene_abstained_samples"],
                        "action_abstained_samples": model["action_abstained_samples"],
                        "latency_mean_ms": model["latency_ms"]["all_attempts"]["mean"],
                        "latency_p50_ms": model["latency_ms"]["all_attempts"]["p50"],
                        "latency_p95_ms": model["latency_ms"]["all_attempts"]["p95"]})
    with files[0].open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    _write_csv(files[1], report["paired_samples"])
    _write_csv(files[2], metrics)
    with files[3].open("x", encoding="utf-8") as stream:
        json.dump(report["wrong_examples"], stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return [str(path.resolve()) for path in files]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mage-run", "gpt-run", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args(argv)
    try:
        report = compare(args.mage_run, args.gpt_run, args.data, freeze=args.freeze)
        print(json.dumps({"success": True, "audit_passed": True, "freeze_verified": report["freeze_verified"],
                          "output_files": save(report, args.output),
                          "mage": report["mage"]["metrics"], "gpt": report["gpt"]["metrics"],
                          "gpt_minus_mage": report["gpt_minus_mage"]}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"success": False, "audit_passed": False, "error_type": type(exc).__name__,
                          "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
