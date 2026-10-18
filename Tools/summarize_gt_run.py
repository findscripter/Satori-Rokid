"""Independent count-based audit and fixed-training-prior comparison."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def evaluate(labels, predictions, scene_names, action_names):
    tp = fp = fn = correct = exact = 0
    per_scene = Counter()
    rows = []
    for gt in labels:
        item = predictions.get(gt["sample_id"])
        valid = bool(item and item["status"] == "completed")
        answer = item["response"] if valid else None
        scene = answer["scene_id"] if answer else None
        action_valid = bool(answer and answer["action_ids"] is not None)
        predicted = set(answer["action_ids"]) if action_valid else set()
        actual = set(gt["action_ids"])
        n_tp, n_fp, n_fn = len(predicted & actual), len(predicted - actual), len(actual - predicted)
        tp += n_tp
        fp += n_fp
        fn += n_fn
        hit = bool(valid and scene == gt["scene_id"])
        correct += hit
        per_scene[(gt["scene_id"], "total")] += 1
        per_scene[(gt["scene_id"], "correct")] += hit
        exact += bool(action_valid and predicted == actual)
        denominator = 2 * n_tp + n_fp + n_fn
        f1 = (2 * n_tp / denominator if denominator else 1.0) if action_valid else 0.0
        union = actual | predicted
        jaccard = (n_tp / len(union) if union else 1.0) if action_valid else 0.0
        rows.append({
            "sample_id": gt["sample_id"], "source_video_id": gt["source_video_id"],
            "status": item["status"] if item else "not_run",
            "gt_scene": scene_names[gt["scene_id"]],
            "predicted_scene": scene_names.get(scene) if scene is not None else None,
            "scene_correct": hit,
            "gt_actions": [action_names[key] for key in sorted(actual)],
            "predicted_actions": [action_names[key] for key in sorted(predicted)],
            "missed_actions": [action_names[key] for key in sorted(actual - predicted)],
            "extra_actions": [action_names[key] for key in sorted(predicted - actual)],
            "true_positives": n_tp, "false_positives": n_fp, "false_negatives": n_fn,
            "action_f1": f1, "jaccard": jaccard,
            "answer": answer["answer"] if answer else None,
            "latency_ms": item["latency_ms"] if item else None,
        })
    n = len(labels)
    return {
        "samples": n, "scene_correct": correct, "scene_top1": correct / n,
        "action_tp": tp, "action_fp": fp, "action_fn": fn,
        "action_micro_precision": tp / (tp + fp) if tp + fp else 0,
        "action_micro_recall": tp / (tp + fn) if tp + fn else 0,
        "action_micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        "action_exact_match": exact / n,
        "action_sample_mean_f1": sum(row["action_f1"] for row in rows) / n,
        "action_mean_jaccard": sum(row["jaccard"] for row in rows) / n,
        "scene_by_gt_class": [{"scene": scene_names[sid], "correct": per_scene[(sid, "correct")],
                               "total": per_scene[(sid, "total")]} for sid in sorted({gt["scene_id"] for gt in labels})],
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("Research/Data/charades-ego-gt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    taxonomy, truth = read(args.data / "taxonomy.json"), read(args.data / "gt.json")
    official_scores, prior = read(args.run / "scores.json"), read(args.data / "baseline-protocol.json")
    scene_names = {item["id"]: item["label"] for item in taxonomy["scenes"]}
    action_names = {item["id"]: item["label"] for item in taxonomy["actions"]}
    scene_ids = {name: sid for sid, name in scene_names.items()}
    predictions = {}
    for line in (args.run / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        assert item["sample_id"] not in predictions
        predictions[item["sample_id"]] = item
    labels = truth["labels"]
    result = evaluate(labels, predictions, scene_names, action_names)
    checks = {"scene_top1": official_scores["scene"]["top1"],
              "action_micro_precision": official_scores["actions"]["micro_precision"],
              "action_micro_recall": official_scores["actions"]["micro_recall"],
              "action_micro_f1": official_scores["actions"]["micro_f1"],
              "action_exact_match": official_scores["actions"]["exact_match"],
              "action_sample_mean_f1": official_scores["actions"]["sample_mean_f1"],
              "action_mean_jaccard": official_scores["actions"]["mean_jaccard"]}
    for name, value in checks.items():
        assert abs(result[name] - value) < 1e-12, f"Independent metric mismatch: {name}"
    baselines = {gt["sample_id"]: {"status": "completed", "latency_ms": None,
        "response": {"scene_id": scene_ids[prior["scene_label"]], "action_ids": prior["action_ids"],
                     "answer": "Fixed training-frequency prior; no image model or API call."}} for gt in labels}
    baseline = evaluate(labels, baselines, scene_names, action_names)
    baseline.pop("rows")
    report = {"success": True, "independent_metrics_match": checks,
              "model": result, "fixed_training_prior_baseline": {**prior, **baseline},
              "scores_sha256": hashlib.sha256((args.run / "scores.json").read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                             for key, value in row.items()})
    print(json.dumps({"success": True, "model": {k: v for k, v in result.items() if k != "rows"},
                      "baseline": baseline}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
