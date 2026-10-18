"""Standard-library recount with an explicit historical-dataset GT witness.

The per-model protocols omitted a GT-hash field. This audit records that defect;
it never edits a frozen run or represents a retrospective witness as preregistered.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import string

ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / "Evidence/satori-all-models-20261018/dispider/independent-recount.json"
JOBS = [
    ("gtea-goal28", "gtea-satori/goal28", "f797d110c683bbb0dba17414586267790cd91bbd7451d3936ec4b943301ff35c"),
    ("gtea-forecast96-raw73-v2", "gtea-satori/forecast96-raw73-v2", "f0aae814f9ca8876b14eaefe81d904cbf45da5d93f96e4026c357ea99646c7c7"),
    ("egotaskqa-indirect200", "egotaskqa-satori/frozen-original-indirect-200", "d5121a119578db684c12e4dc1ffa88e6f0709449cadd9c136ddad9170aee01ee"),
]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normal(value):
    value = value.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


results = []
for name, relative, expected_gt in JOBS:
    run = ROOT / "Research/Backend/runs" / ("dispider-" + name + "-20261018")
    data = ROOT / "Research/Data" / relative
    meta, metrics, manifest, taxonomy, protocol = [read(run / (filename + ".json"))
                                                 for filename in ("run", "metrics", "manifest", "taxonomy", "protocol")]
    witness = read(data / "protocol.json")
    gt = read(data / "gt.json")
    assert sha(data / "gt.json") == expected_gt == metrics["gt_sha256"] == witness["source_gt_sha256"]
    assert "source_gt_sha256" not in protocol
    assert sha(data / "infer-manifest.json") == meta["source_manifest_sha256"] == protocol["source_manifest_sha256"]
    assert sha(data / "taxonomy.json") == meta["source_taxonomy_sha256"] == protocol["source_taxonomy_sha256"]
    assert manifest == read(data / "infer-manifest.json")
    assert taxonomy == read(data / "taxonomy.json")
    assert gt["benchmark_id"] == manifest["benchmark_id"] == meta["benchmark_id"]
    for filename in ("manifest", "taxonomy", "protocol"):
        assert sha(run / (filename + ".json")) == meta[filename + "_snapshot_sha256"]
    predictions = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
    traces = [json.loads(line) for line in (run / "api-calls.jsonl").read_text().splitlines()]
    pm, tm = ({row["sample_id"]: row for row in rows} for rows in (predictions, traces))
    gm = {row["sample_id"]: row for row in gt["samples"]}
    assert len(pm) == len(predictions) == len(tm) == len(traces) == len(manifest["samples"])
    assert set(pm) == set(tm) == set(gm)
    counts = defaultdict(Counter)
    errors, finishes = Counter(), Counter()
    for sample in manifest["samples"]:
        sid = sample["sample_id"]
        row, trace, gold = pm[sid], tm[sid], gm[sid]
        assert trace["image_sha256"] == {frame["frame_id"]: sha(data / frame["image_path"]) for frame in sample["frames"]}
        assert trace["generation_parameters"] == meta["generation_parameters"] == {"max_tokens": 4096, "temperature": 0.0}
        assert trace["system_prompt_sha256"] == meta["prompt_sha256"] == "681d054bd716f6a69f6a3bd863eeb2efa0afa93d7885afc51fcac5991ba9ccb6"
        visible = {"task_kind": sample["task_kind"], "question": sample["question"], "cutoff_ms": sample["cutoff_ms"],
                   "candidates": taxonomy[sample["candidate_group"]] if sample["candidate_group"] else None,
                   "frames": [{"frame_id": f["frame_id"], "timestamp_ms": f["timestamp_ms"]} for f in sample["frames"]]}
        assert trace["payload_sha256"] == hashlib.sha256(compact(visible).encode()).hexdigest()
        assert row["request_attempted"] and row["latency_ms"] == trace["latency_ms"]
        answer = row["response"]["answer"] if row["status"] == "completed" else None
        if row["status"] == "completed":
            assert row["response"] == trace["raw_json"] and trace["status"] == "success"
        else:
            assert row["response"] is None and trace["status"] == "error"
        strict = answer is not None and answer in gold["answers"]
        hit = answer is not None and (normal(answer) in {normal(a) for a in gold["answers"]}
                                      if sample["task_kind"] == "video_qa" else strict)
        for group in ("all", "horizon=" + str(gold["metadata"].get("look_ahead_ms"))):
            counts[group].update(samples=1, correct=hit, strict_correct=strict, failed=row["status"] == "failed")
        errors[trace.get("error_code", "success")] += 1
        finishes[trace.get("finish_reason", "missing")] += 1
    assert counts["all"]["correct"] == metrics["correct"]
    assert counts["all"]["strict_correct"] == metrics["strict_exact_correct"]
    assert counts["all"]["failed"] == metrics["failed_samples"] == meta["failed_samples"]
    assert meta["api_calls"] == meta["planned_samples"] == len(traces)
    assert metrics["not_run_samples"] == meta["not_run_samples"] == 0
    results.append({"dataset": name, "passed": True, "counts": dict(counts), "errors": dict(errors),
                    "finish_reasons": dict(finishes), "gt_sha256": expected_gt,
                    "historical_dataset_protocol_path": str(data / "protocol.json"),
                    "historical_dataset_protocol_sha256": sha(data / "protocol.json"),
                    "protocol_gap": "Dispider protocol omitted source_gt_sha256; actual labels match historical frozen dataset protocol and fixed pre-existing GT bytes. This is a post-run verification, not a repaired pre-registration."})
if DEST.exists():
    raise FileExistsError("Refusing to replace independent result")
DEST.write_text(json.dumps({"passed": True, "runs": results}, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"passed": True, "runs": results}, indent=2))
