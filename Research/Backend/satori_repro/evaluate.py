"""Score system states and raw observations against separately authored labels.

State scores may include the effect of human confirmation. Raw observations are
scored only where the label explicitly includes observation_step_id. Missing
values never silently become null labels, and no labels enter model inference.
"""

from collections import Counter
import json
from pathlib import Path

from .method_evaluate import evaluate_method


KINDS = {"observe", "clarify", "guide", "complete"}


def _step(value):
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError("Step IDs must be nonempty strings or explicit null")
    return value


def _index(rows, *, labels=False):
    result = {}
    for row in rows:
        fid = row.get("frame_id") if isinstance(row, dict) else None
        if not isinstance(fid, str) or not fid.strip():
            raise ValueError("Every record needs a non-empty frame_id")
        if fid in result:
            raise ValueError(f"duplicate frame_id: {fid}")
        if labels:
            required = {"frame_id", "current_step_id", "decision_kind"}
            if not required <= row.keys() or row.keys() - required - {"observation_step_id"}:
                raise ValueError("Label has missing or undefined fields")
            step = _step(row["current_step_id"])
            kind = row["decision_kind"]
            synthetic = None
            raw_present = "observation_step_id" in row
            raw_step = _step(row["observation_step_id"]) if raw_present else None
        else:
            state, decision = row.get("state"), row.get("decision")
            if not isinstance(state, dict) or "current_step_id" not in state:
                raise ValueError("Prediction requires state.current_step_id, including explicit null")
            if not isinstance(decision, dict) or "kind" not in decision:
                raise ValueError("Prediction requires decision.kind")
            if type(row.get("synthetic")) is not bool:
                raise ValueError("Every prediction requires an explicit boolean synthetic flag")
            step, kind, synthetic = _step(state["current_step_id"]), decision["kind"], row["synthetic"]
            raw_present = "observation" in row
            if raw_present:
                observation = row["observation"]
                if not isinstance(observation, dict) or "current_step_id" not in observation:
                    raise ValueError("Present observations require current_step_id, including explicit null")
                raw_step = _step(observation["current_step_id"])
            else:
                raw_step = None
        if not isinstance(kind, str) or kind not in KINDS:
            raise ValueError("Invalid decision kind")
        result[fid] = {"step": step, "kind": kind, "synthetic": synthetic,
                       "raw_present": raw_present, "raw_step": raw_step}
    return result


def _reject_constant(value):
    raise ValueError("Non-finite JSON values are not valid evaluation input")


def evaluate(timeline_path, labels_path):
    try:
        lines = Path(timeline_path).read_text(encoding="utf-8-sig").splitlines()
        rows = [json.loads(line, parse_constant=_reject_constant) for line in lines if line.strip()]
        labels = json.loads(Path(labels_path).read_text(encoding="utf-8-sig"), parse_constant=_reject_constant)
    except (OSError, UnicodeError, TypeError, ValueError):
        raise ValueError("Cannot read evaluation inputs as valid JSON") from None
    predictions = _index(rows)
    if not isinstance(labels, dict) or set(labels) != {"labels"} or not isinstance(labels["labels"], list):
        raise ValueError("Labels file must contain only a labels array")
    truth = _index(labels["labels"], labels=True)
    matched = sorted(predictions.keys() & truth.keys())
    if not matched:
        raise ValueError("No matching prediction and label frames")
    step_correct = decision_correct = tp = fp = fn = 0
    confusion = Counter()
    intervention = {"clarify", "guide"}
    for fid in matched:
        actual, expected = predictions[fid], truth[fid]
        step_correct += actual["step"] == expected["step"]
        decision_correct += actual["kind"] == expected["kind"]
        confusion[(expected["kind"], actual["kind"])] += 1
        p, t = actual["kind"] in intervention, expected["kind"] in intervention
        tp += p and t
        fp += p and not t
        fn += not p and t

    # Classify the whole timeline, including predictions without matching labels.
    flags = {prediction["synthetic"] for prediction in predictions.values()}
    data_kind = "synthetic" if flags == {True} else "model" if flags == {False} else "mixed"
    interpretations = {
        "synthetic": "Synthetic fixture regression only; not evidence of visual understanding.",
        "model": "Scores cover matched independent labels only; inspect both coverage measures.",
        "mixed": "Mixed synthetic and model predictions; these aggregate scores are not model-only evidence.",
    }
    raw_labels = {fid for fid, item in truth.items() if item["raw_present"]}
    raw_predictions = {fid for fid, item in predictions.items() if item["raw_present"]}
    raw_matched = raw_labels & raw_predictions
    raw_correct = sum(predictions[fid]["raw_step"] == truth[fid]["raw_step"] for fid in raw_matched)
    return {
        "matched_frames": len(matched), "label_count": len(truth), "prediction_count": len(predictions),
        "missing_predictions": sorted(truth.keys() - predictions.keys()),
        "unlabeled_predictions": sorted(predictions.keys() - truth.keys()),
        "label_coverage": len(matched) / len(truth),
        "prediction_coverage": len(matched) / len(predictions),
        "state_step_accuracy": step_correct / len(matched),
        "raw_observation_step_accuracy": raw_correct / len(raw_matched) if raw_matched else None,
        "raw_observation_label_count": len(raw_labels),
        "raw_observation_prediction_count": len(raw_predictions),
        "raw_observation_matched_frames": len(raw_matched),
        "raw_observation_missing_predictions": sorted(raw_labels - raw_predictions),
        "raw_observation_unlabeled_predictions": sorted(raw_predictions - raw_labels),
        "raw_observation_label_coverage": len(raw_matched) / len(raw_labels) if raw_labels else None,
        "raw_observation_prediction_coverage": len(raw_matched) / len(raw_predictions) if raw_predictions else None,
        "decision_accuracy": decision_correct / len(matched),
        "intervention_precision": tp / (tp + fp) if tp + fp else None,
        "intervention_recall": tp / (tp + fn) if tp + fn else None,
        "decision_confusion": [{"expected": e, "predicted": p, "count": n} for (e, p), n in sorted(confusion.items())],
        "data_kind": data_kind,
        "interpretation": interpretations[data_kind],
        "metric_notes": "state_step_accuracy includes state updates and human confirmations; raw_observation_step_accuracy uses only explicitly provided observation_step_id labels.",
    }
