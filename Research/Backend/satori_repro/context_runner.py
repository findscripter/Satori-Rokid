"""One-shot environment/action questions over causal images and optional real IMU.

This path does not build a task catalog, infer task checkpoints, or insert user
confirmations. Original source IDs remain only in local audit records.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from . import __version__
from .models import validate_session
from .providers import _image_data_url, _json_loads
from .runner import write_json
from .stage_provider import _json_text


_SENSOR_LIMIT = 256


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read(path):
    raw = Path(path).read_bytes()
    return _json_loads(raw.decode("utf-8-sig")), _sha(raw)


def _save_trace(path, traces):
    with Path(path).open("w", encoding="utf-8") as output:
        for trace in traces:
            output.write(_json_text(trace) + "\n")


def run_context_question(session_path, question, provider, output_dir, *,
                         window_size=8, at_ms=None, sensors_path=None):
    """Ask once at the chosen last image, retaining failed runs for inspection."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A nonempty context question is required")
    if type(window_size) is not int or window_size < 1:
        raise ValueError("window_size must be a positive integer")
    if at_ms is not None and (type(at_ms) is not int or at_ms < 0):
        raise ValueError("at_ms must be a nonnegative integer")
    if getattr(provider, "synthetic", False) is not False:
        raise ValueError("ask-context requires a model provider, not synthetic replay")
    session_path = Path(session_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    initial_trace_count = len(getattr(provider, "traces", []))
    meta = {
        "version": __version__, "status": "running", "synthetic": False,
        "mode": "context_question", "source_kind": "unspecified_session_images",
        "source_authenticity": "Not independently verified by this runner",
        "synthetic_scope": "No prerecorded response fixture; this flag does not authenticate the original images",
        "provider": getattr(provider, "name", type(provider).__name__),
        "model": getattr(provider, "model", None), "endpoint": getattr(provider, "base_url", None),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {"window_size": window_size, "requested_at_ms": at_ms,
                          "sensor_sample_limit": _SENSOR_LIMIT},
        "missing_sensors": ["imu"] if sensors_path is None else [],
        "sensor_status": "missing" if sensors_path is None else "not_yet_validated",
        "inputs": {},
    }
    context = {"question": question, "frames": [], "frame_id_map": {},
               "sensor_samples": None, "sensor_id_map": {}, "inputs": {},
               "assistant_context": None}
    answer, assistant = None, None
    stage = "input loading"
    write_json(output_dir / "run.json", meta)
    (output_dir / "api-calls.jsonl").touch(exist_ok=False)
    try:
        from .context_assistant import ContextAssistant, validate_sensor_samples

        raw_session, session_hash = _read(session_path)
        session = validate_session(raw_session)
        meta["inputs"]["session_sha256"] = session_hash
        context["inputs"]["session_sha256"] = session_hash
        eligible = [(index, frame) for index, frame in enumerate(session["frames"])
                    if at_ms is None or frame["timestamp_ms"] <= at_ms]
        if not eligible:
            raise ValueError("No image frame exists at or before the requested time")
        selected = eligible[-window_size:]
        frames = []
        stage = "image preparation"
        for index, source in selected:
            if not source.get("image_path"):
                raise ValueError("Context questions require an image for every selected frame")
            image_path = (session_path.parent / source["image_path"]).resolve()
            _image_data_url(image_path)  # Validate actual image bytes before any model call.
            frame = {"frame_id": f"v{index:03d}", "timestamp_ms": source["timestamp_ms"],
                     "image_path": str(image_path)}
            frames.append(frame)
            context["frame_id_map"][frame["frame_id"]] = source["frame_id"]
            context["frames"].append({**frame, "sha256": _sha(image_path.read_bytes())})
        cutoff = frames[-1]["timestamp_ms"]
        meta["configuration"]["actual_at_ms"] = cutoff
        meta["configuration"]["first_frame_ms"] = frames[0]["timestamp_ms"]

        stage = "sensor preparation"
        samples = None
        if sensors_path is not None:
            sensor_path = Path(sensors_path)
            if not sensor_path.is_absolute():
                sensor_path = session_path.parent / sensor_path
            sensor_data, sensor_hash = _read(sensor_path.resolve())
            if not isinstance(sensor_data, dict) or set(sensor_data) != {"samples"}:
                raise ValueError("Sensor file must contain exactly the samples field")
            all_samples = validate_sensor_samples(sensor_data["samples"])
            in_window = [(index, sample) for index, sample in enumerate(all_samples)
                         if frames[0]["timestamp_ms"] <= sample["timestamp_ms"] <= cutoff]
            selected_samples = in_window[-_SENSOR_LIMIT:]
            samples = []
            for index, original in selected_samples:
                normalized = deepcopy(original)
                normalized["sample_id"] = f"m{index:03d}"
                samples.append(normalized)
                context["sensor_id_map"][normalized["sample_id"]] = original["sample_id"]
            meta["inputs"]["sensors_sha256"] = sensor_hash
            context["inputs"]["sensors_sha256"] = sensor_hash
            context["inputs"]["selected_sensors_sha256"] = _sha(_json_text(samples).encode("utf-8"))
            context["sensor_selection"] = {
                "file_samples": len(all_samples), "within_image_window": len(in_window),
                "selected": len(samples), "truncated": max(0, len(in_window) - _SENSOR_LIMIT),
                "policy": "Only samples within selected image times; retain newest 256 if necessary",
            }
            context["sensor_samples"] = deepcopy(samples)
        write_json(output_dir / "input-context.json", context)
        stage = "context inference"
        assistant = ContextAssistant(provider)
        answer = assistant.ask(deepcopy(frames), question, history=[], sensor_samples=deepcopy(samples))
        stage = "answer writing"
        write_json(output_dir / "answer.json", answer)
        meta["status"] = "completed"
    except KeyboardInterrupt:
        meta.update(status="aborted", error_type="KeyboardInterrupt", error_stage=stage)
        raise
    except Exception as error:
        meta.update(status="failed", error_type=type(error).__name__, error_stage=stage)
        raise RuntimeError(f"Context question failed during {stage}; see run.json. No automatic retry.") from None
    finally:
        if assistant is not None:
            context["assistant_context"] = deepcopy(assistant.last_context)
            if isinstance(context["assistant_context"], dict):
                meta["missing_sensors"] = deepcopy(context["assistant_context"].get("missing_sensors", ["imu"]))
                meta["sensor_status"] = context["assistant_context"].get("sensor_context", {}).get("status", "unknown")
        traces = deepcopy(getattr(provider, "traces", [])[initial_trace_count:])
        meta.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                    selected_frames=len(context["frames"]), api_calls=len(traces),
                    usage=[trace.get("usage") for trace in traces],
                    returned_models=[trace.get("returned_model") for trace in traces])
        _save_trace(output_dir / "api-calls.jsonl", traces)
        write_json(output_dir / "input-context.json", context)
        write_json(output_dir / "run.json", meta)
    return {"status": meta["status"], "answer": answer["answer"], "answer_path": str(output_dir / "answer.json"),
            "model": meta["model"], "api_calls": meta["api_calls"], "missing_sensors": meta["missing_sensors"],
            "source_kind": meta["source_kind"]}
