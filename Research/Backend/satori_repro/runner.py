"""Causal replay with an inspectable Unity projection."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from . import __version__
from .decision import decide
from .models import validate_observation, validate_session, validate_task
from .state import initial_state, update_state


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _unity(session_id, task, records, synthetic):
    steps = {s["step_id"]: s["instruction"] for s in task["steps"]}
    result = []
    for row in records:
        state, decision = row["state"], row["decision"]
        item = {key: deepcopy(state[key]) for key in (
            "scene", "action", "current_step_id", "confidence", "uncertain",
            "confirmed_completed_step_ids", "observed_completed_step_ids", "evidence_frame_ids")}
        item.update(frame_id=row["frame_id"], timestamp_ms=row["timestamp_ms"], task_title=task["title"],
                    current_step_instruction=steps.get(state["current_step_id"], ""),
                    decision_kind=decision["kind"], decision_message=decision["message"], image_path=row["image_path"])
        result.append(item)
    return {"session_id": session_id, "synthetic": synthetic, "records": result}


def run_session(session_path, task_path, provider, output_dir, *, context_mode="stateful",
                window_size=4, history_limit=8, cooldown_ms=15000):
    if context_mode not in {"single", "window", "stateful"}:
        raise ValueError("Unknown context mode")
    for name, value, minimum in (("window_size", window_size, 1), ("history_limit", history_limit, 0), ("cooldown_ms", cooldown_ms, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"Invalid {name}")
    session_path, task_path, output_dir = Path(session_path), Path(task_path), Path(output_dir)
    task, session = validate_task(read_json(task_path)), validate_session(read_json(session_path))
    if not session["frames"]:
        raise ValueError("Session requires at least one frame")
    frames = deepcopy(session["frames"])
    for frame in frames:
        if frame.get("image_path"):
            frame["image_path"] = str((session_path.parent / frame["image_path"]).resolve())
    output_dir.mkdir(parents=True, exist_ok=False)
    synthetic = bool(provider.synthetic)
    metadata = {"version": __version__, "status": "running", "session_id": session["session_id"],
                "started_utc": datetime.now(timezone.utc).isoformat(), "synthetic": synthetic,
                "provider": provider.name, "model": getattr(provider, "model", None),
                "inputs": {"session_sha256": _sha(session_path), "task_sha256": _sha(task_path)},
                "configuration": {"context_mode": context_mode, "window_size": window_size,
                                  "history_limit": history_limit, "cooldown_ms": cooldown_ms}}
    replay = getattr(provider, "path", None)
    if replay and Path(replay).is_file():
        metadata["inputs"]["responses_sha256"] = _sha(replay)
    write_json(output_dir / "run.json", metadata)
    records, state, last_hint = [], initial_state(task), None
    events = sorted(session["events"], key=lambda event: event["timestamp_ms"])
    event_index, stage = 0, "initialization"
    try:
        with (output_dir / "timeline.jsonl").open("w", encoding="utf-8") as timeline:
            for i, frame in enumerate(frames):
                now = frame["timestamp_ms"]
                count = 1 if context_mode == "single" else window_size
                selected = frames[max(0, i + 1 - count):i + 1]
                history = []
                if context_mode == "stateful" and history_limit:
                    history = [{"frame_id": r["frame_id"], "timestamp_ms": r["timestamp_ms"],
                                "state": deepcopy(r["state"]), "decision": deepcopy(r["decision"])} for r in records[-history_limit:]]
                stage = "input hashing"
                assets = [{"frame_id": f["frame_id"], "sha256": _sha(f["image_path"]) if f.get("image_path") else None} for f in selected]
                stage = "inference"
                start = time.perf_counter()
                raw = provider.infer(deepcopy(selected), task=deepcopy(task), history=history,
                                     state=deepcopy(state) if context_mode == "stateful" else None)
                inference_ms = (time.perf_counter() - start) * 1000
                ids = [f["frame_id"] for f in selected]
                stage = "observation validation"
                observation = validate_observation(raw, task, ids)
                batch = []
                while event_index < len(events) and events[event_index]["timestamp_ms"] <= now:
                    batch.append(events[event_index])
                    event_index += 1
                stage = "state update"
                state = update_state(state, observation, batch, task=task, timestamp_ms=now, frame_ids=ids)
                decision = decide(state, last_hint, timestamp_ms=now, cooldown_ms=cooldown_ms)
                if decision["kind"] != "observe":
                    last_hint = deepcopy(decision)
                row = {"frame_id": frame["frame_id"], "timestamp_ms": now, "input_frame_ids": ids,
                       "input_assets": assets, "image_path": frame.get("image_path", ""), "synthetic": synthetic,
                       "inference_ms": round(inference_ms, 3), "observation": observation,
                       "state": deepcopy(state), "decision": decision}
                stage = "record writing"
                timeline.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                timeline.flush()
                records.append(row)
        metadata["status"] = "completed"
    except Exception as error:
        metadata.update(status="failed", error_type=type(error).__name__, error_stage=stage)
        raise RuntimeError(f"Pipeline failed during {stage}; see run.json. Partial records retained.") from None
    finally:
        metadata.update(finished_utc=datetime.now(timezone.utc).isoformat(), processed_frames=len(records),
                        unconsumed_events=len(events) - event_index)
        write_json(output_dir / "run.json", metadata)
        summary = {"status": metadata["status"], "processed_frames": len(records), "synthetic": synthetic,
                   "decision_counts": dict(Counter(r["decision"]["kind"] for r in records)),
                   "uncertain_frames": sum(r["state"]["uncertain"] for r in records),
                   "interpretation": "Synthetic regression only; not model performance." if synthetic else "Predictions require independent labels."}
        write_json(output_dir / "summary.json", summary)
        write_json(output_dir / "unity-report.json", _unity(session["session_id"], task, records, synthetic))
    return summary
