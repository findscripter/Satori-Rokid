import json

import pytest

from egotaskqa_train_mode_baseline import freeze, score


def test_training_tie_is_lexical_and_does_not_consult_test_answers(tmp_path):
    rows = [{"question_id": index, "answer": "Beta" if index < 10151 else "Alpha"}
            for index in range(20302)] + [{"question_id": 20302, "answer": "Other"}]
    train = tmp_path / "train.json"
    train.write_text(json.dumps(rows))
    output = tmp_path / "prior.json"
    prior = freeze(train, output)
    assert prior["selected_answer"] == "Alpha"
    assert prior["selected_answer_train_frequency"] == 10151
    assert prior["test_answers_read_for_selection"] is False
    with pytest.raises(FileExistsError, match="already frozen"):
        freeze(train, output)


def test_same_declared_lexical_normalization_is_separate_from_strict_em(tmp_path):
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"selected_answer": "Alpha", "baseline": "test_prior"}))
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps({"benchmark_id": "unit", "samples": [
        {"sample_id": f"s{index:03d}", "answers": ["The Alpha."]} for index in range(200)]}))
    result = score(prior, gt, tmp_path / "score.json")
    assert result["fixed_denominator"] == 200
    assert result["strict_exact_match"] == 0
    assert result["normalized_exact_match"] == 1
    assert result["model_calls"] == 0
