"""Causal task state with separate model hypotheses and human confirmations."""

from copy import deepcopy

from .models import _timestamp, _validate_event, validate_observation, validate_task


# Engineering baseline; this is a confidence heuristic, not calibrated probability.
CONFIDENCE_THRESHOLD = 0.6


def initial_state(task):
    task = validate_task(task)
    return {
        "task_id": task["task_id"], "task_title": task["title"],
        "task_step_ids": [step["step_id"] for step in task["steps"]],
        "task_steps": task["steps"], "task_confirmed": False,
        "timestamp_ms": None, "current_step_id": None,
        "confirmed_completed_step_ids": [], "observed_completed_step_ids": [],
        "scene": "", "objects": [], "action": "", "confidence": 0.0,
        "uncertain": True, "evidence_frame_ids": [], "needs_help": False,
        "help_text": "",
    }


def update_state(previous, observation, events, *, task, timestamp_ms, frame_ids):
    """Apply this frame and its new events without modifying any input.

    Reliable observations may revise model-only progress. Uncertain frames retain
    the last progress estimate; neither path retracts explicit user completion.
    Events are applied after observations, in chronological/stable order.
    """
    timestamp_ms = _timestamp(timestamp_ms)
    task = validate_task(task)
    if not isinstance(previous, dict) or previous.get("task_id") != task["task_id"]:
        raise ValueError("previous state belongs to a different task")
    previous_timestamp = previous.get("timestamp_ms")
    if previous_timestamp is not None and timestamp_ms <= previous_timestamp:
        raise ValueError("state timestamps must be strictly increasing")
    observation = validate_observation(observation, task, frame_ids)
    if not isinstance(events, list):
        raise ValueError("events must be an array")
    events = [_validate_event(event) for event in events]
    step_ids = [step["step_id"] for step in task["steps"]]
    for event in events:
        if event["timestamp_ms"] > timestamp_ms:
            raise ValueError("future events cannot update current state")
        if previous_timestamp is not None and event["timestamp_ms"] <= previous_timestamp:
            raise ValueError("events must belong to the current unprocessed batch")
        if "task_id" in event and event["task_id"] != task["task_id"]:
            raise ValueError("event refers to a different task")
        if "step_id" in event and event["step_id"] not in step_ids:
            raise ValueError("event refers to an undefined step")

    result = deepcopy(previous)
    result["timestamp_ms"] = timestamp_ms
    for key in ("scene", "objects", "action", "confidence", "evidence_frame_ids",
                "needs_help", "help_text"):
        result[key] = deepcopy(observation[key])
    reliable = not observation["uncertain"] and observation["confidence"] >= CONFIDENCE_THRESHOLD
    result["uncertain"] = not reliable
    if reliable:
        result["observed_completed_step_ids"] = list(observation["completed_step_ids"])
        result["current_step_id"] = observation["current_step_id"]

    confirmed = set(previous["confirmed_completed_step_ids"])
    step_event_changed_current = False
    for event in sorted(events, key=lambda item: item["timestamp_ms"]):
        step_before_event = result["current_step_id"]
        if event["type"] == "confirm_task":
            result["task_confirmed"] = True
        elif event["type"] == "confirm_step":
            confirmed.add(event["step_id"])
            result["current_step_id"] = next((sid for sid in step_ids if sid not in confirmed), None)
        elif event["type"] == "reopen_step":
            confirmed.discard(event["step_id"])
            result["current_step_id"] = event["step_id"]
            result["observed_completed_step_ids"] = [
                sid for sid in result["observed_completed_step_ids"] if sid != event["step_id"]
            ]
        step_event_changed_current |= result["current_step_id"] != step_before_event

    if result["current_step_id"] in confirmed or len(confirmed) == len(step_ids):
        result["current_step_id"] = next((sid for sid in step_ids if sid not in confirmed), None)
    result["confirmed_completed_step_ids"] = [sid for sid in step_ids if sid in confirmed]
    # Guidance belongs to the observed step. Human progress changes must not
    # silently attach that text to another step; wait for a fresh observation.
    if step_event_changed_current or observation["current_step_id"] in confirmed:
        result["needs_help"] = False
        result["help_text"] = ""
    return result
