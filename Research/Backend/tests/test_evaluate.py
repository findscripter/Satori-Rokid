import json

import pytest

from satori_repro.evaluate import evaluate


def files(tmp_path, rows, labels):
    timeline, truth = tmp_path / "timeline.jsonl", tmp_path / "labels.json"
    timeline.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    truth.write_text(json.dumps({"labels": labels}), encoding="utf-8")
    return timeline, truth


def row(fid, step, kind):
    return {"frame_id": fid, "synthetic": True, "state": {"current_step_id": step}, "decision": {"kind": kind}}


def label(fid, step, kind):
    return {"frame_id": fid, "current_step_id": step, "decision_kind": kind}


def test_scores_match_by_id_and_report_coverage(tmp_path):
    args = files(tmp_path, [row("b", "s2", "observe"), row("a", "s1", "guide"), row("extra", None, "observe")],
                 [label("a", "s1", "observe"), label("b", "s1", "observe"), label("missing", None, "observe")])
    scores = evaluate(*args)
    assert scores["matched_frames"] == 2
    assert scores["missing_predictions"] == ["missing"]
    assert scores["unlabeled_predictions"] == ["extra"]
    assert scores["state_step_accuracy"] == 0.5
    assert "step_accuracy" not in scores
    assert scores["raw_observation_step_accuracy"] is None
    assert scores["decision_accuracy"] == 0.5
    assert scores["label_coverage"] == pytest.approx(2 / 3)
    assert scores["prediction_coverage"] == pytest.approx(2 / 3)
    assert scores["data_kind"] == "synthetic"
    assert scores["intervention_precision"] == 0
    assert scores["intervention_recall"] is None


@pytest.mark.parametrize("duplicate", ["predictions", "labels"])
def test_duplicate_ids_are_rejected(tmp_path, duplicate):
    rows, labels = [row("a", None, "observe")], [label("a", None, "observe")]
    if duplicate == "predictions":
        rows *= 2
    else:
        labels *= 2
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(*files(tmp_path, rows, labels))


def test_empty_overlap_is_not_a_perfect_score(tmp_path):
    with pytest.raises(ValueError, match="matching"):
        evaluate(*files(tmp_path, [row("a", None, "observe")], [label("b", None, "observe")]))


def test_unknown_label_decision_rejected(tmp_path):
    with pytest.raises(ValueError):
        evaluate(*files(tmp_path, [row("a", None, "observe")], [label("a", None, "bad")]))


def test_raw_predictions_are_scored_only_against_separate_raw_labels(tmp_path):
    corrected = row("a", "s2", "observe")
    corrected["observation"] = {"current_step_id": "s1"}
    truth = label("a", "s2", "observe")
    scores = evaluate(*files(tmp_path, [corrected], [truth]))
    assert scores["state_step_accuracy"] == 1
    assert scores["raw_observation_step_accuracy"] is None
    assert scores["raw_observation_label_count"] == 0
    assert scores["raw_observation_label_coverage"] is None
    truth["observation_step_id"] = "s2"
    scores = evaluate(*files(tmp_path, [corrected], [truth]))
    assert scores["state_step_accuracy"] == 1
    assert scores["raw_observation_step_accuracy"] == 0
    assert scores["raw_observation_matched_frames"] == 1
    assert scores["raw_observation_label_coverage"] == 1


def test_raw_metric_reports_missing_predictions_and_partial_label_coverage(tmp_path):
    a, b, c = row("a", "s1", "observe"), row("b", "s2", "observe"), row("c", None, "complete")
    a["observation"] = {"current_step_id": "s1"}
    c["observation"] = {"current_step_id": None}
    labels = [label("a", "s1", "observe"), label("b", "s2", "observe"),
              label("c", None, "complete"), label("missing", "s1", "observe")]
    for item in labels:
        item["observation_step_id"] = item["current_step_id"]
    scores = evaluate(*files(tmp_path, [a, b, c], labels))
    assert scores["raw_observation_step_accuracy"] == 1
    assert scores["raw_observation_label_count"] == 4
    assert scores["raw_observation_prediction_count"] == 2
    assert scores["raw_observation_matched_frames"] == 2
    assert scores["raw_observation_missing_predictions"] == ["b", "missing"]
    assert scores["raw_observation_label_coverage"] == 0.5


def test_unlabeled_synthetic_record_prevents_model_only_claim(tmp_path):
    model = row("a", "s1", "observe")
    model["synthetic"] = False
    scores = evaluate(*files(tmp_path, [model, row("unlabeled", "s2", "observe")], [label("a", "s1", "observe")]))
    assert scores["data_kind"] == "mixed"
    assert scores["prediction_coverage"] == 0.5
    assert scores["label_coverage"] == 1
    assert "mixed" in scores["interpretation"].lower()


@pytest.mark.parametrize("malformation", ["label_step_missing", "label_step_typo", "prediction_state_missing",
    "prediction_step_missing", "prediction_state_not_object", "prediction_decision_not_object",
    "synthetic_missing", "synthetic_number", "synthetic_object", "label_extra", "label_empty_step",
    "label_decision_list", "observation_step_wrong_type", "raw_label_wrong_type"])
def test_malformed_records_raise_value_error_instead_of_producing_scores(tmp_path, malformation):
    prediction, truth = row("a", None, "observe"), label("a", None, "observe")
    if malformation == "label_step_missing":
        del truth["current_step_id"]
    elif malformation == "label_step_typo":
        truth["current_step_typo"] = truth.pop("current_step_id")
    elif malformation == "prediction_state_missing":
        del prediction["state"]
    elif malformation == "prediction_step_missing":
        prediction["state"] = {}
    elif malformation == "prediction_state_not_object":
        prediction["state"] = []
    elif malformation == "prediction_decision_not_object":
        prediction["decision"] = []
    elif malformation == "synthetic_missing":
        del prediction["synthetic"]
    elif malformation == "synthetic_number":
        prediction["synthetic"] = 0
    elif malformation == "synthetic_object":
        prediction["synthetic"] = {}
    elif malformation == "label_extra":
        truth["current_step_typo"] = "s1"
    elif malformation == "label_empty_step":
        truth["current_step_id"] = ""
    elif malformation == "label_decision_list":
        truth["decision_kind"] = []
    elif malformation == "observation_step_wrong_type":
        prediction["observation"] = {"current_step_id": []}
    elif malformation == "raw_label_wrong_type":
        truth["observation_step_id"] = 2
    with pytest.raises(ValueError):
        evaluate(*files(tmp_path, [prediction], [truth]))


@pytest.mark.parametrize("rows, labels", [([], []), ([], [label("a", None, "observe")]),
    ([row("a", None, "observe")], [])])
def test_empty_data_never_returns_a_score(tmp_path, rows, labels):
    with pytest.raises(ValueError):
        evaluate(*files(tmp_path, rows, labels))


def test_missing_and_invalid_json_files_raise_value_error(tmp_path):
    with pytest.raises(ValueError):
        evaluate(tmp_path / "missing.jsonl", tmp_path / "labels.json")
    timeline, labels = files(tmp_path, [row("a", None, "observe")], [label("a", None, "observe")])
    timeline.write_text("not JSON", encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate(timeline, labels)
