"""Freeze EgoTaskQA original indirect questions without leaking labels to inference.

Uses the data archive publicly linked by the ICML 2024 MFAS authors. The
archive's ``qa_ori`` contains the original 40,323-question version; ``qa`` is a
different, larger question version and must not silently replace it. No API
calls, video access application, or label-dependent sample replacement.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import tarfile


BENCHMARK = "egotaskqa-original-indirect-test-200-seed20261017"
FIELDS = (
    "question", "type", "category", "semantic", "structural", "step",
    "program", "answer", "interval", "reference", "reasoning_type",
    "video_id", "question_id",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    path.write_text(data, encoding="utf-8")


def read_questions(archive: Path, variant: str = "qa_ori") -> tuple[list[dict], dict]:
    if variant not in ("qa_ori", "qa"):
        raise ValueError("Unknown archive variant")
    member = f"Data/{variant}/indirect/test_qas.json"
    with tarfile.open(archive, "r:gz") as stream:
        record = stream.getmember(member)
        if not record.isfile() or record.size > 30 * 1024 * 1024:
            raise ValueError("Unexpected QA member type or size")
        data = stream.extractfile(record).read()
    questions = json.loads(data)
    ids = set()
    for item in questions:
        required = {"question_id", "question", "answer", "interval", "reference"}
        if not isinstance(item, dict) or not required <= item.keys():
            raise ValueError("Unexpected EgoTaskQA question schema")
        if type(item["question_id"]) is not int or item["question_id"] in ids:
            raise ValueError("Question IDs must be unique integers")
        ids.add(item["question_id"])
        if item["reference"] != "indirect":
            raise ValueError("Direct-reference item present in indirect test")
        if not all(isinstance(item[key], str) and item[key].strip()
                   for key in ("question", "answer", "interval")):
            raise ValueError("Question, answer, and interval must be nonempty text")
        if not re.fullmatch(r"[\w-]+\|P\d+\|\d+\|\d+", item["interval"]):
            raise ValueError("Unexpected interval identifier")
    metadata = {"member": member, "member_sha256": sha256_bytes(data),
                "rows": len(questions), "variant": variant}
    return questions, metadata


def select_questions(questions: list[dict], count: int = 200,
                     seed: int = 20261017) -> list[dict]:
    if not 0 < count <= len(questions):
        raise ValueError("Requested count outside source population")
    # Sort IDs before sampling; source JSON order and answers cannot affect it.
    ordered = sorted(questions, key=lambda item: item["question_id"])
    selected = random.Random(seed).sample(ordered, count)
    return sorted(selected, key=lambda item: item["question_id"])


def safe_video_name(interval: str) -> str:
    # Original Linux filenames contain '|', which Windows cannot create.
    return "clip-" + hashlib.sha256(interval.encode("utf-8")).hexdigest()[:24] + ".mp4"


def prepare(archive: Path, output: Path, count: int = 200,
            seed: int = 20261017) -> dict:
    if (output / "infer-manifest.json").exists():
        raise FileExistsError("Inference assets already frozen; refusing to reset preparation state")
    questions, source = read_questions(archive)
    selected = select_questions(questions, count, seed)
    benchmark = f"egotaskqa-original-indirect-test-{count}-seed{seed}"
    frozen_ids = [item["question_id"] for item in selected]
    selection = {
        "schema_version": 1, "benchmark_id": benchmark,
        "population": source, "selection_seed": seed,
        "selection_method": "Sort question_id ascending, random.Random(seed).sample, sort selected IDs",
        "selection_depends_on_answers_or_media": False,
        "question_ids": frozen_ids,
        "ids_sha256": sha256_bytes(json.dumps(frozen_ids, separators=(",", ":")).encode()),
        "satori_original_200_ids_available": False,
        "fixed_denominator": count,
    }
    existing = output / "selection.json"
    if existing.exists() and json.loads(existing.read_text(encoding="utf-8")) != selection:
        raise ValueError("Refusing to overwrite a different frozen selection")
    inference = {"schema_version": 1, "benchmark_id": benchmark,
                 "status": "pending_media_not_executable", "samples": []}
    gt = {"schema_version": 1, "benchmark_id": benchmark,
          "provenance": {"archive_member": source["member"],
                         "member_sha256": source["member_sha256"],
                         "source_split": "original_indirect_test",
                         "satori_original_selection": False}, "samples": []}
    media = []
    for index, item in enumerate(selected):
        sample_id = f"s{index:03d}"
        inference["samples"].append({
            "sample_id": sample_id, "task_kind": "video_qa",
            "question": item["question"], "cutoff_ms": 0, "frames": [],
            "candidate_group": None, "preparation_status": "missing_video",
        })
        gt["samples"].append({
            "sample_id": sample_id, "answers": [item["answer"]],
            "metadata": {"question_id": item["question_id"],
                         **{key: item.get(key) for key in ("reference", "category", "type", "semantic", "structural")}},
        })
        media.append({"sample_id": sample_id, "source_interval": item["interval"],
                      "source_filename": item["interval"] + ".mp4",
                      "windows_filename": safe_video_name(item["interval"]),
                      "status": "missing_video"})
    save_json(output / "selection.json", selection)
    save_json(output / "pending-media-manifest.json", inference)
    save_json(output / "gt.json", gt)
    save_json(output / "media-requirements.json", media)
    summary = {
        "benchmark_id": benchmark, "selected_questions": count,
        "unique_required_clips": len({item["source_interval"] for item in media}),
        "media_ready_questions": 0, "status": "not_run_missing_video",
        "fixed_denominator": count,
        "category_counts": dict(Counter("$".join(item["category"]) for item in selected)),
        "type_counts": dict(Counter("$".join(item["type"]) for item in selected)),
        "no_model_calls": True,
    }
    save_json(output / "preparation-summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("Research/Data/egotaskqa-satori/sources/mfas-Data.tar.gz"))
    parser.add_argument("--output", type=Path, default=Path("Research/Data/egotaskqa-satori/frozen-original-indirect-200"))
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.output, args.count, args.seed), indent=2))


if __name__ == "__main__":
    main()
