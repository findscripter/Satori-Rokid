"""Independent fixture labels test scoring mechanics, not model effectiveness."""
from copy import deepcopy
import json

import pytest

from satori_repro.evaluate import evaluate_method


def row(fid="a", timestamp=0):
    return {"frame_id": fid, "timestamp_ms": timestamp, "active_step_id": "s",
            "phase": "active_step", "confirmed_completed_step_ids": [],
            "pending_confirmation": None, "guidance": None, "stage_calls": [], "events": []}


def label(fid="a"):
    return {"frame_id": fid, "active_step_id": "s", "phase": "active_step",
            "confirmed_completed_step_ids": []}


def files(tmp_path, rows, labels, synthetic=True):
    timeline, truth, meta = [tmp_path / name for name in ("method.jsonl", "labels.json", "run.json")]
    timeline.write_text("\n".join(json.dumps(item) for item in rows), encoding="utf-8")
    truth.write_text(json.dumps({"labels": labels}), encoding="utf-8")
    meta.write_text(json.dumps({"synthetic": synthetic}), encoding="utf-8")
    return timeline, truth, meta


def detection(status="IN_PROGRESS", step="s"):
    return {"step_id": step, "overall_status": "COMPLETED" if status == "COMPLETED" else "PROCESSING",
            "checkpoints": [{"checkpoint_id": "c", "status": status}]}


def call(status="IN_PROGRESS", step="s"):
    return {"stage": "checkpoints", "input_frame_ids": ["a"],
            "response": {**detection(status, step), "evidence_frame_ids": ["a"]}}


def test_method_state_scores_and_raw_checkpoint_labels_are_separate(tmp_path):
    actual, truth = row(), label()
    actual["confirmed_completed_step_ids"] = ["previous"]
    truth["confirmed_completed_step_ids"] = ["previous"]
    actual["stage_calls"] = [call("COMPLETED")]
    actual["checkpoint_states"] = {"c": {"status": "COMPLETED"}}
    truth["checkpoint_detection"] = detection("IN_PROGRESS")
    result = evaluate_method(*files(tmp_path, [actual], [truth]))
    assert result["state_active_step_accuracy"] == 1
    assert result["state_completed_steps_exact_match"] == 1
    assert result["raw_checkpoint_status_accuracy"] == 0
    assert result["raw_checkpoint_overall_accuracy"] == 0
    assert result["raw_checkpoint_exact_match"] == 0
    assert result["data_kind"] == "synthetic"


def test_latched_state_is_never_used_as_current_visual_prediction(tmp_path):
    actual, truth = row(), label()
    actual["checkpoint_states"] = {"c": {"status": "COMPLETED"}}
    actual["latest_checkpoint_detection"] = {**detection("COMPLETED"), "evidence_frame_ids": ["old"]}
    truth["checkpoint_detection"] = detection("COMPLETED")
    result = evaluate_method(*files(tmp_path, [actual], [truth]))
    assert result["raw_checkpoint_status_accuracy"] is None
    assert result["raw_checkpoint_label_coverage"] == 0
    assert result["raw_checkpoint_missing_predictions"] == [{"frame_id": "a", "step_id": "s", "checkpoint_id": "c"}]
    assert result["raw_checkpoint_overall_missing_predictions"] == ["a"]


def test_absent_raw_labels_and_prompt_labels_stay_unscored(tmp_path):
    actual = row()
    actual["stage_calls"] = [call()]
    result = evaluate_method(*files(tmp_path, [actual], [label()], synthetic=False))
    assert result["raw_checkpoint_status_accuracy"] is None
    assert result["raw_checkpoint_label_count"] == 0
    assert result["raw_checkpoint_prediction_count"] == 1
    assert result["pending_confirmation_accuracy"] is None
    assert result["guidance_change_accuracy"] is None
    assert result["data_kind"] == "model"


def test_confirmation_prompts_and_guidance_changes_do_not_count_prefetch_calls(tmp_path):
    a, b = row(), row("b", 1)
    a["pending_confirmation"] = {"kind": "accept_step", "step_id": "s"}
    guidance = {"step_id": "s", "text": "Place cup.", "modalities": ["text"], "evidence_frame_ids": ["a"]}
    a["stage_calls"] = [{"stage": "guidance", "input_frame_ids": ["a"], "response": guidance}]
    b["guidance"] = guidance
    c = {**deepcopy(b), "frame_id": "c", "timestamp_ms": 2}
    labels = [dict(label(), pending_confirmation=a["pending_confirmation"], guidance_changed=False),
              dict(label("b"), pending_confirmation=None, guidance_changed=True),
              dict(label("c"), pending_confirmation=None, guidance_changed=False)]
    result = evaluate_method(*files(tmp_path, [a, b, c], labels))
    assert result["pending_confirmation_accuracy"] == 1
    assert result["guidance_change_accuracy"] == 1
    assert result["guidance_change_count"] == 1
    assert "sampled" in result["metric_notes"]


def test_coverage_includes_missing_and_unlabeled_records(tmp_path):
    result = evaluate_method(*files(tmp_path, [row(), row("extra", 1)], [label(), label("missing")]))
    assert result["state_active_step_accuracy"] == 1
    assert result["label_coverage"] == .5
    assert result["prediction_coverage"] == .5
    assert result["missing_predictions"] == ["missing"]
    assert result["unlabeled_predictions"] == ["extra"]


def test_wrong_step_detection_is_wrong_and_not_disguised_as_matching_checkpoints(tmp_path):
    actual, truth = row(), label()
    actual["stage_calls"] = [call("COMPLETED", "wrong_step")]
    truth["checkpoint_detection"] = detection("COMPLETED")
    result = evaluate_method(*files(tmp_path, [actual], [truth]))
    assert result["raw_checkpoint_overall_accuracy"] == 0
    assert result["raw_checkpoint_exact_match"] == 0
    assert result["raw_checkpoint_status_accuracy"] is None
    assert result["raw_checkpoint_label_coverage"] == 0
    assert result["raw_checkpoint_prediction_coverage"] == 0


def test_partial_raw_coverage_and_order_independent_checkpoint_matching(tmp_path):
    actual, truth = row(), label()
    truth["checkpoint_detection"] = detection()
    truth["checkpoint_detection"]["checkpoints"].append({"checkpoint_id": "second", "status": "UNKNOWN"})
    actual["stage_calls"] = [call()]
    result = evaluate_method(*files(tmp_path, [actual], [truth]))
    assert result["raw_checkpoint_status_accuracy"] == 1
    assert result["raw_checkpoint_label_coverage"] == .5
    assert result["raw_checkpoint_exact_match"] == 0
    actual["stage_calls"][0]["response"]["checkpoints"].insert(0, {"checkpoint_id": "second", "status": "UNKNOWN"})
    result = evaluate_method(*files(tmp_path, [actual], [truth]))
    assert result["raw_checkpoint_exact_match"] == 1


def test_operator_events_are_reported_as_submitted_and_never_become_visual_labels(tmp_path):
    actual = row()
    actual["events"] = [{"type": "confirm_step", "timestamp_ms": 0, "step_id": "previous"}]
    result = evaluate_method(*files(tmp_path, [actual], [label()]))
    assert result["submitted_operator_event_counts"] == {"confirm_step": 1}
    assert result["raw_checkpoint_label_count"] == 0
    assert result["raw_checkpoint_status_accuracy"] is None


@pytest.mark.parametrize("malformation", ["state_missing", "label_missing", "extra_label", "wrong_phase", "bad_completed",
    "duplicate_completed", "bad_synthetic", "missing_synthetic", "bad_pending", "bad_guidance", "bad_stage", "bad_checkpoint",
    "duplicate_checkpoint", "duplicate_stage", "duplicate_frame", "backward_time", "nonfinite", "empty_overlap"])
def test_method_scoring_rejects_missing_or_malformed_contracts(tmp_path, malformation):
    actual, truth = row(), label()
    rows, labels = [actual], [truth]
    if malformation == "state_missing": del actual["active_step_id"]
    elif malformation == "label_missing": del truth["active_step_id"]
    elif malformation == "extra_label": truth["typo"] = None
    elif malformation == "wrong_phase": actual["phase"] = "bogus"
    elif malformation == "bad_completed": actual["confirmed_completed_step_ids"] = [None]
    elif malformation == "duplicate_completed": truth["confirmed_completed_step_ids"] = ["s", "s"]
    elif malformation == "bad_pending": truth["pending_confirmation"] = {"kind": "accept_step"}
    elif malformation == "bad_guidance": actual["guidance"] = {}
    elif malformation == "bad_stage": actual["stage_calls"] = None
    elif malformation == "bad_checkpoint": truth["checkpoint_detection"] = {"overall_status": "COMPLETED"}
    elif malformation == "duplicate_checkpoint":
        truth["checkpoint_detection"] = detection()
        truth["checkpoint_detection"]["checkpoints"] *= 2
    elif malformation == "duplicate_stage": actual["stage_calls"] = [call(), call()]
    elif malformation == "duplicate_frame": rows *= 2
    elif malformation == "backward_time": rows.append(row("b", 0))
    elif malformation == "nonfinite": actual["timestamp_ms"] = float("nan")
    elif malformation == "empty_overlap": truth["frame_id"] = "other"
    paths = files(tmp_path, rows, labels)
    if malformation == "bad_synthetic": paths[2].write_text('{"synthetic": 1}', encoding="utf-8")
    elif malformation == "missing_synthetic": paths[2].write_text('{}', encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate_method(*paths)


@pytest.mark.parametrize("mutation", ["no_checkpoint_evidence", "foreign_checkpoint_evidence", "inconsistent_complete",
    "bad_status", "bad_event", "future_event", "bad_guidance_bool", "row_synthetic_conflict", "duplicate_json_field"])
def test_checkpoint_prompt_and_event_contracts_are_strict(tmp_path, mutation):
    actual, truth = row(), label()
    actual["stage_calls"] = [call()]
    if mutation == "no_checkpoint_evidence": actual["stage_calls"][0]["response"]["evidence_frame_ids"] = []
    elif mutation == "foreign_checkpoint_evidence": actual["stage_calls"][0]["response"]["evidence_frame_ids"] = ["future"]
    elif mutation == "inconsistent_complete": actual["stage_calls"][0]["response"]["overall_status"] = "COMPLETED"
    elif mutation == "bad_status": actual["stage_calls"][0]["response"]["checkpoints"][0]["status"] = []
    elif mutation == "bad_event": actual["events"] = [{"type": "confirm_step", "timestamp_ms": 0}]
    elif mutation == "future_event": actual["events"] = [{"type": "confirm_step", "timestamp_ms": 1, "step_id": "s"}]
    elif mutation == "bad_guidance_bool": truth["guidance_changed"] = 0
    elif mutation == "row_synthetic_conflict": actual["synthetic"] = False
    paths = files(tmp_path, [actual], [truth])
    if mutation == "duplicate_json_field": paths[2].write_text('{"synthetic": true, "synthetic": false}', encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate_method(*paths)
