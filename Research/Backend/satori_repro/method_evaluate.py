"""Score staged-method logs against separate frame labels, without field renaming.

State includes operator input. Visual metrics use only a checkpoint stage actually
called on that record, never accumulated checkpoint states or retained detections.
Prompt metrics describe sampled snapshots, not task success or wall-clock latency.
"""
from collections import Counter
import json
from pathlib import Path


PHASES = {"observing", "awaiting_goal_confirmation", "awaiting_step_acceptance",
          "awaiting_feasible_forecast", "active_step", "awaiting_completion_confirmation", "complete"}
STAGES = {"belief", "desire", "forecast", "checkpoints", "guidance"}
STATUSES = {"NOT_STARTED", "IN_PROGRESS", "COMPLETED", "UNKNOWN"}
OVERALL = {"NOT_STARTED", "PROCESSING", "COMPLETED", "UNRELATED", "UNKNOWN"}
STATE_FIELDS = {"active_step_id", "phase", "confirmed_completed_step_ids"}


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("IDs and text must be nonempty strings")
    return value


def _step(value):
    return None if value is None else _text(value)


def _strings(value):
    if not isinstance(value, list):
        raise ValueError("Expected an explicit string array")
    items = [_text(item) for item in value]
    if len(set(items)) != len(items):
        raise ValueError("Duplicate IDs in array")
    return items


def _fields(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise ValueError("Missing or undefined method evaluation fields")


def _state(value):
    _step(value["active_step_id"])
    if not isinstance(value["phase"], str) or value["phase"] not in PHASES:
        raise ValueError("Invalid method phase")
    _strings(value["confirmed_completed_step_ids"])


def _pending(value):
    if value is None:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("kind"), str):
        raise ValueError("Invalid pending confirmation")
    kind = value["kind"]
    if kind not in {"confirm_task", "accept_step", "confirm_step"}:
        raise ValueError("Invalid pending confirmation kind")
    key = "task_id" if kind == "confirm_task" else "step_id"
    _fields(value, {"kind", key})
    _text(value[key])
    return value


def _detection(value, *, prediction=False, input_ids=None):
    required = {"step_id", "overall_status", "checkpoints"} | ({"evidence_frame_ids"} if prediction else set())
    _fields(value, required)
    _text(value["step_id"])
    if not isinstance(value["overall_status"], str) or value["overall_status"] not in OVERALL:
        raise ValueError("Invalid checkpoint overall status")
    if not isinstance(value["checkpoints"], list) or not value["checkpoints"]:
        raise ValueError("Checkpoint detections need nonempty checkpoint arrays")
    items = {}
    for item in value["checkpoints"]:
        _fields(item, {"checkpoint_id", "status"})
        cid = _text(item["checkpoint_id"])
        if cid in items or not isinstance(item["status"], str) or item["status"] not in STATUSES:
            raise ValueError("Duplicate checkpoint ID or invalid status")
        items[cid] = item["status"]
    if value["overall_status"] == "COMPLETED" and any(status != "COMPLETED" for status in items.values()):
        raise ValueError("Overall completion contradicts checkpoint statuses")
    if prediction:
        evidence = _strings(value["evidence_frame_ids"])
        if not evidence or not set(evidence) <= set(input_ids):
            raise ValueError("Raw checkpoint evidence must use actual stage input IDs")
    return {"step_id": value["step_id"], "overall_status": value["overall_status"], "checkpoints": items}


def _events(events, previous_time, stamp):
    if not isinstance(events, list):
        raise ValueError("Operator events must be an explicit array")
    id_fields = {"confirm_task": "task_id", "switch_task": "task_id", "accept_step": "step_id",
                 "confirm_step": "step_id", "reopen_step": "step_id", "reject_completion": "step_id",
                 "reject_task": None, "reject_step": None}
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("type"), str) or event["type"] not in id_fields:
            raise ValueError("Invalid operator event")
        key = id_fields[event["type"]]
        _fields(event, {"timestamp_ms", "type"} | ({key} if key else set()))
        if type(event["timestamp_ms"]) is not int or not previous_time < event["timestamp_ms"] <= stamp:
            raise ValueError("Operator event must be new, nonnegative and causal")
        if key:
            _text(event[key])


def _json(text):
    def invalid(_):
        raise ValueError("Non-finite JSON is invalid evaluation input")

    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("Duplicate JSON object field")
            obj[key] = value
        return obj

    return json.loads(text, parse_constant=invalid, object_pairs_hook=unique)


def _predictions(rows, synthetic):
    result, previous_guidance, previous_time = {}, None, -1
    for row in rows:
        required = {"frame_id", "timestamp_ms", "pending_confirmation", "guidance", "stage_calls", "events"} | STATE_FIELDS
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError("Prediction missing required method log fields")
        fid = _text(row["frame_id"])
        if fid in result:
            raise ValueError("Duplicate prediction frame_id")
        stamp = row["timestamp_ms"]
        if type(stamp) is not int or stamp <= previous_time:
            raise ValueError("Method record timestamps must be nonnegative and strictly increasing")
        if "synthetic" in row and (type(row["synthetic"]) is not bool or row["synthetic"] != synthetic):
            raise ValueError("Prediction synthetic flag contradicts run metadata")
        _state(row)
        _pending(row["pending_confirmation"])
        guidance = row["guidance"]
        if guidance is not None:
            _fields(guidance, {"step_id", "text", "modalities", "evidence_frame_ids"})
            _text(guidance["step_id"])
            _text(guidance["text"])
            if guidance["modalities"] != ["text"] or not _strings(guidance["evidence_frame_ids"]):
                raise ValueError("Invalid displayed guidance")
        if not isinstance(row["stage_calls"], list):
            raise ValueError("Stage calls must be an array")
        _events(row["events"], previous_time, stamp)
        raw = None
        for call in row["stage_calls"]:
            _fields(call, {"stage", "input_frame_ids", "response"})
            if not isinstance(call["stage"], str) or call["stage"] not in STAGES or not isinstance(call["response"], dict):
                raise ValueError("Invalid stage call")
            input_ids = _strings(call["input_frame_ids"])
            if not input_ids:
                raise ValueError("Stage call needs input frames")
            if call["stage"] == "checkpoints":
                if raw is not None:
                    raise ValueError("Duplicate raw checkpoint stage on one frame")
                raw = _detection(call["response"], prediction=True, input_ids=input_ids)
        result[fid] = {**row, "raw_detection": raw,
                       "guidance_changed": guidance is not None and guidance != previous_guidance}
        previous_guidance = guidance
        previous_time = stamp
    return result


def evaluate_method(timeline_path, labels_path, run_metadata_path):
    """Evaluate native method logs. Missing optional labels produce null metrics.

    Labels require frame_id, active_step_id (explicit null allowed), phase and
    confirmed_completed_step_ids. Optional independent labels: checkpoint_detection,
    pending_confirmation (explicit null allowed), guidance_changed (boolean).
    ``run_metadata_path`` must explicitly state a boolean synthetic flag.
    """
    try:
        lines = Path(timeline_path).read_text(encoding="utf-8-sig").splitlines()
        rows = [_json(line) for line in lines if line.strip()]
        labels = _json(Path(labels_path).read_text(encoding="utf-8-sig"))
        metadata = _json(Path(run_metadata_path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, TypeError, ValueError):
        raise ValueError("Cannot read method evaluation inputs as valid JSON") from None
    if not isinstance(metadata, dict) or type(metadata.get("synthetic")) is not bool:
        raise ValueError("Run metadata needs an explicit boolean synthetic flag")
    predictions = _predictions(rows, metadata["synthetic"])
    _fields(labels, {"labels"})
    if not isinstance(labels["labels"], list):
        raise ValueError("Labels must be an array")
    truth = {}
    for item in labels["labels"]:
        _fields(item, {"frame_id"} | STATE_FIELDS, {"checkpoint_detection", "pending_confirmation", "guidance_changed"})
        fid = _text(item["frame_id"])
        if fid in truth:
            raise ValueError("Duplicate label frame_id")
        _state(item)
        validated = dict(item)
        if "checkpoint_detection" in item:
            validated["checkpoint_detection"] = _detection(item["checkpoint_detection"])
        if "pending_confirmation" in item:
            _pending(item["pending_confirmation"])
        if "guidance_changed" in item and type(item["guidance_changed"]) is not bool:
            raise ValueError("guidance_changed must be an explicit boolean")
        truth[fid] = validated
    matched = sorted(predictions.keys() & truth.keys())
    if not matched:
        raise ValueError("No matching prediction and label frames")
    result = {
        "schema": "satori-method-evaluation-v1", "matched_frames": len(matched),
        "label_count": len(truth), "prediction_count": len(predictions),
        "missing_predictions": sorted(truth.keys() - predictions.keys()),
        "unlabeled_predictions": sorted(predictions.keys() - truth.keys()),
        "label_coverage": len(matched) / len(truth), "prediction_coverage": len(matched) / len(predictions),
        "state_active_step_accuracy": sum(predictions[f]["active_step_id"] == truth[f]["active_step_id"] for f in matched) / len(matched),
        "state_phase_accuracy": sum(predictions[f]["phase"] == truth[f]["phase"] for f in matched) / len(matched),
        "state_completed_steps_exact_match": sum(set(predictions[f]["confirmed_completed_step_ids"]) == set(truth[f]["confirmed_completed_step_ids"]) for f in matched) / len(matched),
        "data_kind": "synthetic" if metadata["synthetic"] else "model",
        "interpretation": "Synthetic fixture regression only; not evidence of visual understanding." if metadata["synthetic"] else "Scores require separately authored labels; run metadata does not establish annotation independence or task effectiveness.",
        "metric_notes": "State includes operator input. Raw checkpoint scores use only current checkpoint stage calls, never latched state. Optional labels are scored only when provided; inspect all coverage values. Guidance changes and pending confirmations are sampled snapshot events, not repeated delivery or wall-clock timing metrics.",
    }
    for label_key, metric in (("pending_confirmation", "pending_confirmation_accuracy"), ("guidance_changed", "guidance_change_accuracy")):
        labeled = {fid for fid, item in truth.items() if label_key in item}
        paired = labeled & predictions.keys()
        result.update({metric: sum(predictions[f][label_key] == truth[f][label_key] for f in paired) / len(paired) if paired else None,
                       label_key + "_label_count": len(labeled), label_key + "_matched_frames": len(paired),
                       label_key + "_missing_predictions": sorted(labeled - predictions.keys()),
                       label_key + "_label_coverage": len(paired) / len(labeled) if labeled else None})
    result["guidance_change_count"] = sum(item["guidance_changed"] for item in predictions.values())
    result["submitted_operator_event_counts"] = dict(sorted(Counter(
        event["type"] for item in predictions.values() for event in item["events"]).items()))
    raw_truth = {fid: item["checkpoint_detection"] for fid, item in truth.items() if "checkpoint_detection" in item}
    raw_predictions = {fid: item["raw_detection"] for fid, item in predictions.items() if item["raw_detection"] is not None}
    raw_frames = raw_truth.keys() & raw_predictions.keys()
    expected = {(fid, item["step_id"], cid): status for fid, item in raw_truth.items() for cid, status in item["checkpoints"].items()}
    actual = {(fid, item["step_id"], cid): status for fid, item in raw_predictions.items() for cid, status in item["checkpoints"].items()}
    paired = expected.keys() & actual.keys()
    def ids(keys):
        return [dict(zip(("frame_id", "step_id", "checkpoint_id"), key)) for key in sorted(keys)]
    result.update({
        "raw_checkpoint_status_accuracy": sum(actual[key] == expected[key] for key in paired) / len(paired) if paired else None,
        "raw_checkpoint_label_count": len(expected), "raw_checkpoint_prediction_count": len(actual),
        "raw_checkpoint_matched_count": len(paired),
        "raw_checkpoint_label_coverage": len(paired) / len(expected) if expected else None,
        "raw_checkpoint_prediction_coverage": len(paired) / len(actual) if actual else None,
        "raw_checkpoint_missing_predictions": ids(expected.keys() - actual.keys()),
        "raw_checkpoint_unlabeled_predictions": ids(actual.keys() - expected.keys()),
        "raw_checkpoint_overall_accuracy": sum((raw_predictions[f]["step_id"], raw_predictions[f]["overall_status"]) == (raw_truth[f]["step_id"], raw_truth[f]["overall_status"]) for f in raw_frames) / len(raw_frames) if raw_frames else None,
        "raw_checkpoint_exact_match": sum(raw_predictions[f] == raw_truth[f] for f in raw_frames) / len(raw_frames) if raw_frames else None,
        "raw_checkpoint_overall_matched_frames": len(raw_frames),
        "raw_checkpoint_overall_label_coverage": len(raw_frames) / len(raw_truth) if raw_truth else None,
        "raw_checkpoint_overall_prediction_coverage": len(raw_frames) / len(raw_predictions) if raw_predictions else None,
        "raw_checkpoint_overall_missing_predictions": sorted(raw_truth.keys() - raw_predictions.keys()),
        "raw_checkpoint_overall_unlabeled_predictions": sorted(raw_predictions.keys() - raw_truth.keys()),
    })
    return result
