"""Read-only extensible model matrix; audit runs and preserve untested nulls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import compare_paper_model_runs as compare
import satori_paper_benchmark as bench


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("GPT", "Mage-VL", "VideoChat3-4B", "Dispider")
TASKS = {
    "goal28": ("gtea-satori/goal28", 28, "gtea-goal28", "videochat3-gtea-goal28-20261017"),
    "forecast96": ("gtea-satori/forecast96-raw73-v2", 96, "gtea-forecast96", "videochat3-gtea-forecast96-raw73-v2-20261017"),
    "egotaskqa200": ("egotaskqa-satori/frozen-original-indirect-200", 200, "egotaskqa-indirect200", "videochat3-egotaskqa-indirect200-20261017"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def existing_selections(root):
    selected = {}
    for task, (_, _, label, vc3) in TASKS.items():
        selected[task, "VideoChat3-4B"] = (root / "Research/Backend/runs" / vc3, None)
        selected[task, "Mage-VL"] = (
            root / "Research/Backend/runs" / ("mage-" + label + "-20261017-gpt-mage-fixed4"),
            root / "Evidence/satori-gpt-mage-20261017/mage/mage-comparison-protocols-20261017" / (label + ".json"),
        )
    return selected


def summarize(root, selected, protocol, *, gt_witnesses=None, model_slots=None,
              execution_profiles=None, comparison_mode="strict"):
    """Re-audit sources and traces; never infer a missing run has score zero."""
    model_slots = tuple(MODELS if model_slots is None else model_slots)
    if (not model_slots or len(set(model_slots)) != len(model_slots)
            or any(not isinstance(name, str) or not name.strip() for name in model_slots)):
        raise ValueError("Model slots must be distinct nonempty names or actual API model aliases")
    if any(task not in TASKS or model not in model_slots for task, model in selected):
        raise ValueError("Unknown task or model slot")
    gt_witnesses = gt_witnesses or {}
    if any(key not in selected for key in gt_witnesses):
        raise ValueError("GT witness must identify an explicitly selected run slot")
    execution_profiles = execution_profiles or {}
    if any(key not in selected for key in execution_profiles):
        raise ValueError("Execution profile must identify an explicitly selected run slot")
    dataset_rows, matrix, comparisons = [], [], {}
    for task, (data_rel, sample_count, _, _) in TASKS.items():
        data = root / "Research/Data" / data_rel
        manifest = json.loads((data / "infer-manifest.json").read_text(encoding="utf-8-sig"))
        if len(manifest["samples"]) != sample_count:
            raise ValueError("Fixed denominator changed")
        hashes = {name: sha(data / filename) for name, filename in (
            ("manifest_sha256", "infer-manifest.json"), ("taxonomy_sha256", "taxonomy.json"), ("gt_sha256", "gt.json"))}
        dataset_rows.append({"task": task, "data_dir": str(data.resolve()), "samples": sample_count,
                             "benchmark_id": manifest["benchmark_id"], **hashes})
        audited = []
        for model in model_slots:
            selection = selected.get((task, model))
            if selection is None:
                matrix.append({"task": task, "model_slot": model, "status": "not_tested", "planned_samples": sample_count,
                               "run_id": None, "correct": None, "accuracy": None, "strict_exact_correct": None,
                               "strict_exact_accuracy": None, "failed_samples": None, "valid_null_abstentions": None,
                               "valid_nonnull_answers": None, "api_calls": 0,
                               "note": "No formal run selected; untested is not a zero score."})
                continue
            run, protocol_source = selection
            profile, profile_sha = execution_profiles.get((task, model), (None, None))
            result = compare.audit_and_score(run, data, protocol_source=protocol_source,
                                             gt_witness_run=gt_witnesses.get((task, model)),
                                             execution_profile=profile, execution_profile_sha256=profile_sha)
            if model != "GPT" and result["model"] != model:
                raise ValueError("Actual run model differs from matrix slot")
            audited.append(result)
            row = {key: value for key, value in result.items() if key != "paired"}
            row.update({"task": task, "model_slot": model, "status": "audited",
                        "artifact_sha256": {name: sha(run / name) for name in
                                            ("run.json", "manifest.json", "taxonomy.json", "protocol.json", "predictions.jsonl", "api-calls.jsonl")}})
            if model == "Dispider":
                row["input_representation_note"] = "Four unique frozen source JPEGs; sixteen native Reaction slots; each source repeated four times."
            matrix.append(row)
        if audited:
            comparisons[task] = compare.compare_results(audited, comparison_mode=comparison_mode)
    audited_rows = [row for row in matrix if row["status"] == "audited"]
    generation_equal = all(row["generation_parameters"] == audited_rows[0]["generation_parameters"] for row in audited_rows)
    return {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
            "comparison_protocol_path": str(protocol.resolve()), "comparison_protocol_sha256": sha(protocol),
            "prompt_sha256": hashlib.sha256(bench.PROMPT.encode("utf-8")).hexdigest(),
            "inference_tool_sha256": sha(root / "Tools/satori_paper_benchmark.py"),
            "default_generation_parameters": {"max_tokens": 4096, "temperature": 0.0},
            "generation_parameters": audited_rows[0]["generation_parameters"] if generation_equal and audited_rows else None,
            "generation_parameters_equal": generation_equal if audited_rows else None,
            "execution_parameters_equal": all(item["execution_parameters_equal"] for item in comparisons.values()) if comparisons else None,
            "comparison_mode": comparison_mode, "unique_frames_per_sample": 4,
            "formal_requests_per_sample": 1, "automatic_retry": False,
            "datasets": dataset_rows, "matrix": matrix,
            "model_slots": list(model_slots),
            "audited_runs": len(selected), "untested_cells": len(TASKS) * len(model_slots) - len(selected),
            "metrics_files_read": False,
            "gt_binding_exceptions": [{"task": row["task"], "model_slot": row["model_slot"],
                                       "run_id": row["run_id"], "per_run_gt_binding_missing": True,
                                       "gt_witness": row["audit"]["gt_witness"]}
                                      for row in matrix if row.get("audit", {}).get("per_run_gt_binding_missing")],
            "limitations": ["Sparse module diagnosis, not the complete Satori system or original sample-identical protocol.",
                            "Different native/HTTP preprocessing and model-specific internal representations; no fair speed/compute ranking.",
                            "Failures, legal nulls and not-run samples remain in every initiated run's fixed denominator.",
                            "Entirely untested cells retain null scores; no fabricated zero or extrapolated accuracy."]}, comparisons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path, default=ROOT / "Evidence/satori-all-models-20261018/COMPARISON-PROTOCOL.md")
    parser.add_argument("--run", action="append", nargs=4, default=[], metavar=("TASK", "MODEL_SLOT", "RUN", "PROTOCOL_SOURCE"),
                        help="Add one explicit run to a missing slot. '-' protocol uses its dataset source filename.")
    parser.add_argument("--gt-witness", action="append", nargs=3, default=[], metavar=("TASK", "MODEL_SLOT", "WITNESS_RUN"),
                        help="Explicit GT-hash omission exception; choose an earlier strictly bound run for this task.")
    parser.add_argument("--model-slot", action="append", default=[], metavar="ACTUAL_MODEL_ALIAS",
                        help="Append a model slot beyond the initial four. New slot must equal run.json model exactly.")
    parser.add_argument("--execution-profile", action="append", nargs=4, default=[], metavar=("TASK", "MODEL_SLOT", "PATH", "SHA256"),
                        help="Explicit pre-frozen execution profile for a selected model/task slot")
    parser.add_argument("--comparison-mode", choices=("strict", "same_observations"), default="strict")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison-dir", type=Path)
    args = parser.parse_args(argv)
    model_slots = (*MODELS, *args.model_slot)
    if len(set(model_slots)) != len(model_slots):
        parser.error("--model-slot must append distinct model names beyond the initial four")
    selected = existing_selections(args.root)
    for task, model, run, protocol_source in args.run:
        if (task, model) in selected:
            parser.error("A model/task slot already has an explicitly selected run; do not silently replace it")
        selected[task, model] = (Path(run), None if protocol_source == "-" else Path(protocol_source))
    witnesses = {}
    for task, model, path in args.gt_witness:
        if (task, model) not in selected or (task, model) in witnesses:
            parser.error("GT witness requires a unique selected model/task slot")
        witnesses[task, model] = Path(path)
    profiles = {}
    for task, model, path, digest in args.execution_profile:
        if (task, model) not in selected or (task, model) in profiles:
            parser.error("Execution profile requires a unique selected model/task slot")
        profiles[task, model] = (Path(path), digest)
    destinations = [args.output]
    if args.comparison_dir:
        destinations.extend(args.comparison_dir / ("compare-" + task + ".json") for task in TASKS)
    if any(path.exists() for path in destinations):
        raise FileExistsError("Choose new outputs; old audit evidence must not be overwritten")
    result, comparisons = summarize(args.root, selected, args.protocol, gt_witnesses=witnesses, model_slots=model_slots,
                                    execution_profiles=profiles, comparison_mode=args.comparison_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.comparison_dir:
        args.comparison_dir.mkdir(parents=True, exist_ok=True)
        for task, comparison in comparisons.items():
            (args.comparison_dir / ("compare-" + task + ".json")).write_text(
                json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"audited_runs": result["audited_runs"], "untested_cells": result["untested_cells"],
                      "gt_binding_exception_runs": len(result["gt_binding_exceptions"]),
                      "matrix": [{key: row[key] for key in ("task", "model_slot", "status", "correct", "strict_exact_correct", "planned_samples", "failed_samples")}
                                 for row in result["matrix"]]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
