"""Strict JSON contracts for inspectable, non-executable model observations."""

from copy import deepcopy
import math


def _fields(value, required, optional=(), *, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    missing = set(required) - value.keys()
    extra = value.keys() - set(required) - set(optional)
    if missing or extra:
        raise ValueError(f"{label} has missing or undefined fields")


def _text(value, *, label, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{label} must be {'a' if allow_empty else 'a nonempty'} string")
    return value


def _timestamp(value, *, label="timestamp_ms"):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _list(value, *, label):
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _ids(value, *, label):
    items = _list(value, label=label)
    for item in items:
        _text(item, label=label)
    if len(set(items)) != len(items):
        raise ValueError(f"{label} must contain unique IDs")
    return list(items)


def validate_task(data):
    """Validate a task and return an independent copy with no extra fields."""
    _fields(data, ("task_id", "title", "steps"), label="task")
    _text(data["task_id"], label="task_id")
    _text(data["title"], label="title")
    steps = _list(data["steps"], label="steps")
    if not steps:
        raise ValueError("steps must not be empty")
    step_ids = []
    for step in steps:
        _fields(step, ("step_id", "instruction", "completion_criteria"), label="step")
        for key in ("step_id", "instruction", "completion_criteria"):
            _text(step[key], label=key)
        step_ids.append(step["step_id"])
    if len(set(step_ids)) != len(step_ids):
        raise ValueError("step IDs must be unique")
    return deepcopy(data)


def _validate_event(data):
    _fields(data, ("timestamp_ms", "type"), ("task_id", "step_id"), label="event")
    _timestamp(data["timestamp_ms"], label="event timestamp_ms")
    event_type = data["type"]
    if not isinstance(event_type, str) or event_type not in {
        "confirm_task", "confirm_step", "reopen_step"
    }:
        raise ValueError("undefined event type")
    required_id = "task_id" if event_type == "confirm_task" else "step_id"
    if required_id not in data:
        raise ValueError(f"{event_type} requires {required_id}")
    for key in ("task_id", "step_id"):
        if key in data:
            _text(data[key], label=f"event {key}")
    if event_type == "confirm_task" and "step_id" in data:
        raise ValueError("confirm_task does not accept step_id")
    return deepcopy(data)


def validate_session(data):
    """Validate frames; order events chronologically, stably within a timestamp."""
    _fields(data, ("session_id", "frames", "events"), label="session")
    _text(data["session_id"], label="session_id")
    frames = _list(data["frames"], label="frames")
    if not frames:
        raise ValueError("frames must not be empty")
    frame_ids = set()
    previous_timestamp = -1
    for frame in frames:
        _fields(frame, ("frame_id", "timestamp_ms"), ("image_path",), label="frame")
        frame_id = _text(frame["frame_id"], label="frame_id")
        timestamp = _timestamp(frame["timestamp_ms"], label="frame timestamp_ms")
        if frame_id in frame_ids:
            raise ValueError("frame IDs must be unique")
        if timestamp <= previous_timestamp:
            raise ValueError("frame timestamps must be strictly increasing")
        frame_ids.add(frame_id)
        previous_timestamp = timestamp
        if "image_path" in frame:
            _text(frame["image_path"], label="image_path")
    events = [_validate_event(event) for event in _list(data["events"], label="events")]
    return {"session_id": data["session_id"], "frames": deepcopy(frames),
            "events": sorted(events, key=lambda event: event["timestamp_ms"])}


def validate_observation(data, task, allowed_frame_ids):
    """Require known steps and evidence from exactly the supplied input window."""
    task = validate_task(task)
    fields = ("scene", "objects", "action", "current_step_id", "completed_step_ids",
              "confidence", "uncertain", "evidence_frame_ids", "needs_help", "help_text")
    _fields(data, fields, label="observation")
    for key in ("scene", "action", "help_text"):
        _text(data[key], label=key, allow_empty=True)
    for obj in _list(data["objects"], label="objects"):
        _fields(obj, ("name", "state"), label="object")
        _text(obj["name"], label="object name")
        _text(obj["state"], label="object state", allow_empty=True)

    step_ids = {step["step_id"] for step in task["steps"]}
    current = data["current_step_id"]
    if current is not None:
        _text(current, label="current_step_id")
        if current not in step_ids:
            raise ValueError("current_step_id is not a defined task step")
    completed = _ids(data["completed_step_ids"], label="completed_step_ids")
    if not set(completed) <= step_ids:
        raise ValueError("completed_step_ids contains an undefined step")

    confidence = data["confidence"]
    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1 or not math.isfinite(confidence)):
        raise ValueError("confidence must be a finite number between 0 and 1")
    for key in ("uncertain", "needs_help"):
        if not isinstance(data[key], bool):
            raise ValueError(f"{key} must be a boolean")

    if not isinstance(allowed_frame_ids, (list, tuple, set, frozenset)):
        raise ValueError("allowed_frame_ids must be an ID collection")
    for frame_id in allowed_frame_ids:
        _text(frame_id, label="allowed frame ID")
    evidence = _ids(data["evidence_frame_ids"], label="evidence_frame_ids")
    if not set(evidence) <= set(allowed_frame_ids):
        raise ValueError("evidence_frame_ids contains a frame outside the input window")
    has_conclusion = any((data["scene"].strip(), data["objects"], data["action"].strip(),
                          current, completed, data["needs_help"], data["help_text"].strip()))
    if has_conclusion and not evidence:
        raise ValueError("observation conclusions require evidence_frame_ids")
    return deepcopy(data)
