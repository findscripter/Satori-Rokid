"""Read-only N-model comparison on the same frozen eight-unique-frame pilot.

Snapshots may use different newline bytes, but each must match its own checksum
and the identical original-source JSON. No API calls, retries or score repair.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Research/Backend"))

import gt_benchmark as benchmark
from summarize_gt_run import evaluate
from compare_gt_runs import require, digest, read, records, distribution, usage_summary, _check_metrics

SAMPLE_COUNT = 32
EXPECTED_GENERATION = {"max_tokens": 4096, "temperature": 0.0}
ALLOWED_TIMEOUTS = (180, 240)
LARGE_ENUMERATION_THRESHOLD = 32


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
    generation = meta["generation_parameters"]
    require(type(generation["max_tokens"]) is int
            and type(generation["temperature"]) in (int, float), "Generation budget/temperature must be numeric, not boolean")
    require(meta.get("timeout_seconds") in ALLOWED_TIMEOUTS,
            f"Timeout must be 180 or 240 seconds: {run_dir}")
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
        require(type(trace["generation_parameters"]["max_tokens"]) is int
                and type(trace["generation_parameters"]["temperature"]) in (int, float),
                f"Request generation budget/temperature has invalid types: {sid}")
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
            "success_rate": completed / SAMPLE_COUNT, "timeout_seconds": meta["timeout_seconds"],
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


def current_source_files(model):
    """Record available file bytes; this is not a historical code-freeze claim."""
    paths = ["Tools/gt_benchmark.py", "Tools/summarize_gt_run.py", "Tools/compare_gt_runs.py",
             "Tools/compare_video_models.py"]
    if model == "Mage-VL":
        paths += ["Tools/run_mage_gt.py", "Tools/deploy/mage_vl/server.py"]
    elif model == "VideoChat3-4B":
        paths += ["Tools/deploy/video_eval/server.py", "Tools/deploy/video_eval/run_cpu_benchmark.py",
                  "Tools/deploy/videochat3/backend.py"]
    elif model == "Dispider":
        paths += ["Tools/deploy/video_eval/server.py", "Tools/deploy/video_eval/run_cpu_benchmark.py",
                  "Tools/deploy/dispider/backend.py"]
    else:
        paths += ["Tools/run_gpt_gt.py"]
    return {name: {"sha256": digest(PROJECT_ROOT / name), "bytes": (PROJECT_ROOT / name).stat().st_size}
            for name in paths if (PROJECT_ROOT / name).is_file()}


def enumeration_diagnostics(rows, predictions, taxonomy):
    counts, large, full = [], [], []
    candidates = len(taxonomy["actions"])
    for row in rows:
        answer = predictions[row["sample_id"]]["response"]
        if answer is None or answer["action_ids"] is None:
            continue
        count = len(answer["action_ids"])
        counts.append(count)
        if count >= LARGE_ENUMERATION_THRESHOLD:
            large.append({"sample_id": row["sample_id"], "predicted_action_count": count,
                          "candidate_fraction": count / candidates,
                          "tp": row["true_positives"], "fp": row["false_positives"],
                          "fn": row["false_negatives"], "action_f1": row["action_f1"]})
        if count == candidates:
            full.append(row["sample_id"])
    return {"candidate_action_count": candidates,
            "action_count_distribution_nonabstained_complete_outputs": distribution(counts),
            "maximum_predicted_actions": max(counts) if counts else None,
            "large_enumeration_threshold": LARGE_ENUMERATION_THRESHOLD,
            "large_enumeration_samples": large,
            "full_candidate_enumeration_sample_ids": full,
            "large_sample_tp": sum(row["tp"] for row in large),
            "large_sample_fp": sum(row["fp"] for row in large),
            "large_sample_fn": sum(row["fn"] for row in large),
            "note": "描述性诊断；不改变固定32分母，不去除大枚举答案，不按此阈值重新计主分。"}


def compare(runs, data):
    require(isinstance(runs, dict) and 2 <= len(runs) <= 4, "Provide two to four distinct labelled runs")
    require(len({Path(path).resolve() for path in runs.values()}) == len(runs), "Duplicate run directory")
    data = Path(data).resolve()
    manifest = benchmark.validate_manifest(read(data / "infer-manifest.json"), frame_count=8)
    taxonomy = benchmark.validate_taxonomy(read(data / "taxonomy.json"))
    truth = read(data / "gt.json")
    benchmark._validate_gt(truth, manifest, taxonomy)
    require(len(manifest["samples"]) == SAMPLE_COUNT, "Exactly 32 frozen samples are required")
    for sample in manifest["samples"]:
        for frame in sample["frames"]:
            require(digest((data / frame["image_path"]).resolve()) == frame["sha256"],
                    f"Source image bytes changed: {sample['sample_id']} {frame['frame_id']}")
    models, predictions, traces, metadata = {}, {}, {}, {}
    same_fields = ("benchmark_id", "protocol", "source_manifest_sha256", "source_taxonomy_sha256",
                   "prompt_sha256", "frame_count", "generation_parameters")
    reference = None
    for label, path in runs.items():
        require(isinstance(label, str) and 1 <= len(label) <= 64 and label.strip() == label,
                "Invalid model label")
        model, prediction, trace, meta = audit_run(path, data, manifest, taxonomy, truth)
        if reference is None:
            reference = meta
        for field in same_fields:
            require(meta.get(field) == reference.get(field), f"Cross-run protocol mismatch: {field}")
        model["source_data_sha256"] = {name: digest(data / name) for name in
                                        ("infer-manifest.json", "taxonomy.json", "gt.json")}
        model["snapshot_sha256"] = {name: digest(Path(path) / name) for name in ("manifest.json", "taxonomy.json")}
        for name in ("scores.json", "native-protocol-binding.json"):
            if (Path(path) / name).exists():
                model["artifact_sha256"][name] = digest(Path(path) / name)
        model["current_source_files"] = current_source_files(meta["model"])
        model["current_source_files_scope"] = "Current local bytes only; not a claim of historical inference-code identity"
        binding_path = Path(path) / "native-protocol-binding.json"
        binding = read(binding_path) if binding_path.exists() else None
        if binding is not None:
            require(binding.get("model") == meta["model"] and binding.get("frame_count") == 8
                    and binding.get("automatic_retry") is False and binding.get("gt_read") is False
                    and binding.get("timeout_seconds") == meta["timeout_seconds"],
                    "Native protocol binding disagrees with its run")
        model["native_protocol_binding"] = binding
        model["client_latency_context"] = (binding.get("client_location") if binding else
                                           "Historical Windows HTTP client through SSH relay to Mage"
                                           if meta["model"] == "Mage-VL" else "Historical Windows HTTP client to public third-party GPT API")
        model["enumeration_diagnostics"] = enumeration_diagnostics(model["rows"], prediction, taxonomy)
        models[label], predictions[label], traces[label], metadata[label] = model, prediction, trace, meta

    indexed_rows = {label: {row["sample_id"]: row for row in model["rows"]} for label, model in models.items()}
    indexed_traces = {label: {row["sample_id"]: row for row in trace} for label, trace in traces.items()}
    paired = []
    for gt in truth["labels"]:
        sid = gt["sample_id"]
        pair = {"sample_id": sid, "source_video_id": gt["source_video_id"],
                "gt_scene_id": gt["scene_id"], "gt_action_ids": sorted(set(gt["action_ids"])), "models": {}}
        for label in models:
            row, trace, prediction = indexed_rows[label][sid], indexed_traces[label][sid], predictions[label][sid]
            answer = prediction["response"]
            pair["models"][label] = {
                "status": row["status"], "scene_correct": row["scene_correct"],
                "scene_id": answer["scene_id"] if answer else None,
                "action_ids": answer["action_ids"] if answer else None,
                "scene_abstained": bool(answer is not None and answer["scene_id"] is None),
                "action_abstained": bool(answer is not None and answer["action_ids"] is None),
                "predicted_action_count": len(answer["action_ids"]) if answer and answer["action_ids"] is not None else None,
                "tp": row["true_positives"], "fp": row["false_positives"], "fn": row["false_negatives"],
                "action_f1": row["action_f1"], "jaccard": row["jaccard"], "latency_ms": row["latency_ms"],
                "error_code": trace.get("error_code"), "http_status": trace.get("http_status"),
                "finish_reason": trace.get("finish_reason", trace.get("server_finish_reason")),
                "returned_model": trace.get("returned_model"), "response_issues": trace.get("response_issues", []),
                "answer": row["answer"],
            }
        paired.append(pair)
    pairwise = []
    for left, right in itertools.combinations(models, 2):
        subset = [(row["models"][left], row["models"][right]) for row in paired]
        pairwise.append({"left": left, "right": right,
                         "left_only_scene_correct": sum(a["scene_correct"] and not b["scene_correct"] for a, b in subset),
                         "right_only_scene_correct": sum(b["scene_correct"] and not a["scene_correct"] for a, b in subset),
                         "left_action_sample_f1_better": sum(a["action_f1"] > b["action_f1"] + 1e-12 for a, b in subset),
                         "right_action_sample_f1_better": sum(b["action_f1"] > a["action_f1"] + 1e-12 for a, b in subset),
                         "action_sample_f1_tied": sum(abs(a["action_f1"] - b["action_f1"]) <= 1e-12 for a, b in subset),
                         "both_completed_samples": sum(a["status"] == b["status"] == "completed" for a, b in subset),
                         "right_minus_left_action_micro_f1": models[right]["metrics"]["action_micro_f1"] - models[left]["metrics"]["action_micro_f1"]})
    return {"success": True, "audit_passed": True, "created_utc": datetime.now(timezone.utc).isoformat(),
            "data_directory": str(data), "fixed_denominator": SAMPLE_COUNT,
            "source_data_sha256": {name: digest(data / name) for name in ("infer-manifest.json", "taxonomy.json", "gt.json")},
            "protocol": {field: reference[field] for field in same_fields},
            "snapshot_policy": "Each snapshot byte hash must match its own run.json; contents and original source hashes must match across runs; LF/CRLF snapshot byte equality is not required",
            "historical_inference_code_freeze_verified_by_this_script": False,
            "native_service_private_audit": "Checked separately by task owner; this script never reads service tokens or private raw audit logs",
            "latency_comparison": {"fair_ranking": False,
                                   "timeout_seconds_by_model": {label: metadata[label]["timeout_seconds"] for label in models},
                                   "explanation": "旧模型180秒、新模型240秒；客户端公网/SSH与服务器CPU回环路径、硬件和预处理不同。耗时只描述各接口，不作公平速度排名。"},
            "models": models, "paired_samples": paired, "pairwise": pairwise,
            "limitations": ["32段已观察过的开发诊断集，不能推断完整官方测试集或Satori步骤完成率。",
                            "8张稀疏帧与整片动作GT范围不同；GT也可能漏标。当前离散标签指标不是官方mAP。",
                            "VideoChat3时序压缩与Dispider16槽重复等原生预处理不同；8张独立观测与文本相同不等于计算量相同。",
                            "GPT使用第三方请求/响应别名，底层权重身份未独立验证；tokenizer不同，token数不直接等价。",
                            "失败、弃答及大枚举答案均保留32个样本分母，不修复输出、不补跑或挑选样本。"]}


def metrics_rows(report):
    return [{"label": label, "model": model["model"], **model["metrics"],
             "completed_samples": model["completed_samples"], "failed_samples": model["failed_samples"],
             "success_rate": model["success_rate"], "scene_abstained_samples": model["scene_abstained_samples"],
             "action_abstained_samples": model["action_abstained_samples"],
             "large_enumeration_samples": len(model["enumeration_diagnostics"]["large_enumeration_samples"]),
             "timeout_seconds": model["timeout_seconds"],
             "latency_p50_seconds": model["latency_ms"]["all_attempts"]["p50"] / 1000,
             "latency_p95_seconds": model["latency_ms"]["all_attempts"]["p95"] / 1000}
            for label, model in report["models"].items()]


def readable_table(report):
    lines = ["| 模型 | 场景准确率 | 动作P | 动作R | 动作F1 | TP/FP/FN | 有效返回 | 大枚举≥32 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in metrics_rows(report):
        percentage = lambda key: f"{100 * row[key]:.2f}%"
        lines.append(f"| {row['label']} | {percentage('scene_top1')} | {percentage('action_micro_precision')} | "
                     f"{percentage('action_micro_recall')} | {percentage('action_micro_f1')} | "
                     f"{row['action_tp']}/{row['action_fp']}/{row['action_fn']} | {row['completed_samples']}/32 | "
                     f"{row['large_enumeration_samples']} |")
    return "\n".join(lines)


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def save(report, output):
    output = Path(output).resolve()
    files = [output, output.with_suffix(".paired.csv"), output.with_suffix(".metrics.csv")]
    require(len(set(files)) == 3 and output.suffix.lower() == ".json", "Choose a new JSON output path")
    require(not any(path.exists() for path in files), "Comparison output already exists; refusing replacement")
    protected = [Path(report["data_directory"]).resolve()] + [Path(x["run_directory"]).resolve() for x in report["models"].values()]
    require(not any(path.is_relative_to(root) for path in files for root in protected), "Output cannot modify source data or original run directories")
    output.parent.mkdir(parents=True, exist_ok=True)
    paired_rows = []
    for pair in report["paired_samples"]:
        row = {key: value for key, value in pair.items() if key != "models"}
        for label, prediction in pair["models"].items():
            row.update({f"{label}.{key}": value for key, value in prediction.items()})
        paired_rows.append(row)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    write_csv(files[1], paired_rows)
    write_csv(files[2], metrics_rows(report))
    return [str(path) for path in files]


def parse_run_specs(specifications):
    runs = {}
    for specification in specifications:
        label, separator, path = specification.partition("=")
        require(separator and label and path and label not in runs, "--run must be a unique label=run-directory")
        runs[label] = Path(path).resolve()
    return runs


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="label=run-directory; repeat two to four times")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = compare(parse_run_specs(args.run), args.data)
        files = save(report, args.output)
        print(readable_table(report))
        print(json.dumps({"success": True, "audit_passed": True, "fixed_denominator": SAMPLE_COUNT,
                          "fair_latency_ranking": False, "output_files": files}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"success": False, "audit_passed": False,
                          "error_type": type(error).__name__, "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

