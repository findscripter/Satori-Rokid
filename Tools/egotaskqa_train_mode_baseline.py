"""Train-only constant-answer prior, frozen before independent test scoring."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from prepare_egotaskqa_frames import file_sha
from prepare_egotaskqa_satori import save_json
from satori_paper_benchmark import normalize_answer


def freeze(train_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Training prior is already frozen")
    train = json.loads(train_path.read_text(encoding="utf-8"))
    if len(train) != 20303:
        raise ValueError("Expected original indirect train split with 20,303 questions")
    if len({row["question_id"] for row in train}) != len(train):
        raise ValueError("Training question IDs are not unique")
    counts = Counter(row["answer"] for row in train)
    if any(not isinstance(answer, str) or not answer.strip() for answer in counts):
        raise ValueError("Training answers must be nonempty strings")
    selected, frequency = sorted(counts.items(), key=lambda row: (-row[1], row[0]))[0]
    result = {
        "schema_version": 1, "baseline": "constant_original_indirect_training_mode",
        "train_sha256": file_sha(train_path), "train_question_count": len(train),
        "selection_rule": "Count unmodified training answer strings; select maximal frequency; resolve ties by ascending Python string lexicographic order.",
        "selected_answer": selected, "selected_answer_train_frequency": frequency,
        "test_answers_read_for_selection": False,
        "model_calls": 0,
    }
    save_json(output, result)
    return result


def score(prior_path: Path, gt_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Baseline test score is already saved")
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    answer = prior["selected_answer"]
    labels = gt["samples"]
    strict = sum(answer in row["answers"] for row in labels)
    normalized = sum(normalize_answer(answer) in {normalize_answer(value) for value in row["answers"]}
                     for row in labels)
    result = {
        "schema_version": 1, "benchmark_id": gt["benchmark_id"],
        "baseline": prior["baseline"], "frozen_prediction_sha256": file_sha(prior_path),
        "gt_sha256": file_sha(gt_path), "selected_answer": answer,
        "fixed_denominator": len(labels), "strict_correct": strict,
        "strict_exact_match": strict / len(labels),
        "normalized_correct": normalized, "normalized_exact_match": normalized / len(labels),
        "normalization": "Same satori_paper_benchmark.normalize_answer: lowercase, remove ASCII punctuation, remove English articles a/an/the, collapse whitespace; no synonyms or LLM judge.",
        "model_calls": 0,
    }
    save_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "score"))
    parser.add_argument("--train", type=Path, default=Path("Research/Data/egotaskqa-satori/annotations/mfas-qa_ori-indirect/train_qas.json"))
    parser.add_argument("--prior", type=Path, default=Path("Research/Data/egotaskqa-satori/frozen-original-indirect-200/train-mode-prior-20261017.json"))
    parser.add_argument("--gt", type=Path, default=Path("Research/Data/egotaskqa-satori/frozen-original-indirect-200/gt.json"))
    parser.add_argument("--output", type=Path, default=Path("Evidence/egotaskqa-train-mode-baseline-20261017.json"))
    args = parser.parse_args()
    result = freeze(args.train, args.prior) if args.mode == "freeze" else score(args.prior, args.gt, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
