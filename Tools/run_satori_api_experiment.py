"""Run a small, real-image staged method test; operator events are explicitly scripted."""
import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from satori_repro.satori_method import SatoriMethodSession
from satori_repro.stage_provider import SatoriStageProvider
from satori_repro.stage_resume import ResumableSatoriStageProvider


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_task(path, task_id):
    """Keep source semantics without inventing per-step object requirements.

    Source ``objects`` describes the whole task. It is useful scene vocabulary,
    but is not evidence that every listed object is required by every step.
    Checkpoint instruction and detector prompt are deliberately kept together:
    a broad detector question alone cannot establish quantities such as 11g.
    """
    assignments = [node for node in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(node, ast.Assign)]
    if not assignments:
        raise ValueError("Source plan needs a literal task assignment")
    original = ast.literal_eval(assignments[0].value)
    if not isinstance(original, dict) or not isinstance(original.get("steps"), list) or not original["steps"]:
        raise ValueError("Source plan needs a nonempty steps list")

    def text(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Source plan text must be nonempty")
        return value

    def texts(value):
        if not isinstance(value, list):
            raise ValueError("Source plan text lists must be arrays")
        return [text(item) for item in value]

    text(task_id)
    title, objects = text(original.get("desired_task")), texts(original.get("objects"))
    steps = []
    for i, step in enumerate(original["steps"]):
        if not isinstance(step, dict) or type(step.get("is_start")) is not bool:
            raise ValueError("Source steps need an explicit boolean is_start")
        checkpoints = step.get("checkpoints")
        if not isinstance(checkpoints, list) or not checkpoints or any(not isinstance(c, dict) for c in checkpoints):
            raise ValueError("Source steps need nonempty checkpoint objects")
        steps.append({"step_id": f"{task_id}_{i+1}", "instruction": text(step.get("content")),
                      "is_start": step["is_start"], "step_check_prompts": texts(step.get("step_check_prompts")),
                      "required_objects": [], "required_objects_status": "unknown",
                      "checkpoints": [{"checkpoint_id": f"{task_id}_{i+1}_c{j+1}", "question":
                          "Required result: " + text(c.get("instruction")) + ". Visual check: " + text(c.get("blip_prompt"))
                          + " Judge the complete required result, including quantities and timing. The broad visual check alone does not prove those constraints; use UNKNOWN if they cannot be observed."}
                                      for j, c in enumerate(checkpoints)]})
    return {"task_id": task_id, "title": title, "objects": objects, "steps": steps,
            "adaptation": "Original public task-plan semantics with stable IDs; per-step required objects are unknown, not known empty. Task objects are scene vocabulary only.",
            "source_path": str(path), "source_sha256": file_hash(path)}


def catalog(root):
    cleaning = {"task_id": "egodex_clean_surface_adapted", "title": "擦拭局部桌面",
                "adaptation": "Adapted subtask for a 10-second EgoDex clip; not Satori's full clean-room task.", "steps": [
        {"step_id": "wipe_surface", "instruction": "用清洁布擦拭目标桌面，必要时调整清洁用品的位置后继续擦拭。", "required_objects": ["cleaning cloth", "table"],
         "checkpoints": [{"checkpoint_id": "cloth_moving", "question": "Is the cloth visibly being moved across the target table surface?"},
                         {"checkpoint_id": "wiping_finished", "question": "Is there clear evidence that the intended wiping has finished and ceased, rather than a momentary pause? If not visible, say UNKNOWN or IN_PROGRESS."}]},
        {"step_id": "finish_cleanup", "instruction": "确认擦拭完成后，整理或放下清洁用品。", "required_objects": ["cleaning cloth"],
         "checkpoints": [{"checkpoint_id": "tools_put_down", "question": "Have the cleaning cloth and any held cleaning supplies visibly been put down after wiping ended?"}]}]}
    plans = root / "Research/Reference/satori-assistance/pipelines/task/task_plans"
    return [cleaning, source_task(plans / "coffee.py", "source_coffee"), source_task(plans / "ns.py", "source_switch")]


def model_catalog(plans):
    """Send task semantics with uniform IDs; retain research provenance locally."""
    return [{"task_id": f"t{i:03d}", "title": task["title"],
        **({"objects": deepcopy(task["objects"])} if "objects" in task else {}), "steps": [
        {"step_id": f"t{i:03d}_s{j:03d}", "instruction": step["instruction"],
         "required_objects": list(step["required_objects"]),
         **{key: deepcopy(step[key]) for key in ("is_start", "step_check_prompts", "required_objects_status") if key in step},
         "checkpoints": [
             {"checkpoint_id": f"t{i:03d}_s{j:03d}_c{k:03d}", "question": checkpoint["question"]}
             for k, checkpoint in enumerate(step["checkpoints"])]}
        for j, step in enumerate(task["steps"])]}
        for i, task in enumerate(plans)]


def flatten(record, frame, tasks, previous):
    """Project displayed content with its own evidence, including cached sources.

    evidence_timestamp_ms is the source result's generation/event time, not the
    capture time of every cited image. Unknown source times remain null.
    """
    def stage_time(stage, response):
        for snapshot in (record, previous):
            if snapshot and any(call.get("stage") == stage and call.get("response") == response
                                for call in snapshot.get("stage_calls", [])):
                return snapshot.get("timestamp_ms")
        return None

    def event_time(event_type, id_key=None, id_value=None):
        stamps = [event["timestamp_ms"] for snapshot in (record, previous) if snapshot
                  for event in snapshot.get("events", []) if event.get("type") == event_type
                  and (id_key is None or event.get(id_key) == id_value)]
        return max(stamps) if stamps else None

    task_id = record["confirmed_task_id"] or (record.get("desire_candidate") or {}).get("task_id")
    task = tasks.get(task_id)
    belief = record.get("belief") or {"scene": "", "action": "", "confidence": 0, "uncertain": True, "evidence_frame_ids": []}
    active = record["active_step_id"]
    steps = {s["step_id"]: s for s in task["steps"]} if task else {}
    pending = record.get("pending_confirmation")
    kind, message = "observe", "继续观察当前动作；片段本身不足以证明任务已经完成。"
    evidence_source = "belief"
    evidence_ids = list(belief["evidence_frame_ids"])
    evidence_time = stage_time("belief", belief)
    if record["phase"] == "complete":
        kind, message = "complete", "任务步骤已得到操作者确认。"
        evidence_source, evidence_ids = "user_confirmation", []
        evidence_time = event_time("confirm_step")
    elif pending:
        kind = "clarify"
        if pending["kind"] == "confirm_task":
            message = "推测目标为“" + task["title"] + "”，请确认。"
            proposal = record.get("desire_candidate") or {}
            evidence_source, evidence_ids = "desire", list(proposal.get("evidence_frame_ids", []))
            evidence_time = stage_time("desire", proposal)
            if proposal.get("source") == "user_requested":
                evidence_source = "user_request"
                evidence_time = event_time("switch_task", "task_id", pending["task_id"])
        elif pending["kind"] == "accept_step":
            message = "建议步骤：" + steps[pending["step_id"]]["instruction"] + " 请确认是否接受。"
            cache = record.get("cached_forecast") or {}
            forecast = cache.get("forecast", {}) if cache.get("target_step_id") == pending["step_id"] else {}
            evidence_source, evidence_ids = "forecast", list(forecast.get("evidence_frame_ids", []))
            evidence_time = cache.get("generated_at_ms") if forecast else None
        else:
            message = "视觉检查点提示步骤可能已完成，请操作者核实后确认。"
            checkpoints = [checkpoint for checkpoint in record.get("checkpoint_statuses", [])
                           if checkpoint.get("status") == "COMPLETED"]
            evidence_source = "checkpoints"
            if checkpoints:
                evidence_ids = list(dict.fromkeys(fid for checkpoint in checkpoints
                                                 for fid in checkpoint.get("evidence_frame_ids", [])))
                stamps = [checkpoint["timestamp_ms"] for checkpoint in checkpoints if "timestamp_ms" in checkpoint]
                evidence_time = max(stamps) if stamps else None
            else:
                detection = record.get("latest_checkpoint_detection") or {}
                evidence_ids = list(detection.get("evidence_frame_ids", []))
                evidence_time = stage_time("checkpoints", detection)
    elif record.get("guidance") and (not previous or previous.get("guidance") != record["guidance"]):
        kind, message = "guide", record["guidance"]["text"]
        guidance = record["guidance"]
        evidence_source, evidence_ids = "guidance", list(guidance.get("evidence_frame_ids", []))
        evidence_time = stage_time("guidance", guidance)
        for snapshot in (previous, record):
            cache = snapshot.get("cached_forecast") if snapshot else None
            if cache and cache.get("guidance") == guidance:
                evidence_time = cache.get("generated_at_ms")
                break
    observed = [active] if active and pending and pending["kind"] == "confirm_step" else []
    return {"frame_id": frame["frame_id"], "timestamp_ms": frame["timestamp_ms"], "image_path": frame["image_path"],
            "task_title": task["title"] if task else "目标尚未确认", "scene": belief["scene"], "action": belief["action"],
            "current_step_id": active, "current_step_instruction": steps.get(active, {}).get("instruction", ""),
            "confidence": belief["confidence"], "uncertain": belief["uncertain"],
            "confirmed_completed_step_ids": list(record["confirmed_completed_step_ids"]), "observed_completed_step_ids": observed,
            "decision_kind": kind, "decision_message": message, "evidence_frame_ids": evidence_ids,
            "evidence_source": evidence_source, "evidence_timestamp_ms": evidence_time}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, help="Failed real API run whose exact successful call prefix can be reused")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sample = root / "Research/Data/real-task/clean_surface_0"
    session_path = sample / "session.json"
    source = json.loads(session_path.read_text(encoding="utf-8"))
    frames = source["frames"]
    frame_id_map = {}
    for index, frame in enumerate(frames):
        opaque_id = f"v{index:03d}"
        frame_id_map[opaque_id] = frame["frame_id"]
        frame["frame_id"] = opaque_id
        frame["image_path"] = str((sample / frame["image_path"]).resolve())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Keep provenance local: only opaque IDs and pixels go to the model.
    save(out / "frame-id-map.json", frame_id_map)
    provenance_plans = catalog(root)
    save(out / "task-provenance.json", provenance_plans)
    plans = model_catalog(provenance_plans)
    save(out / "task-catalog.json", plans)
    tasks = {task["task_id"]: task for task in plans}
    live_provider = SatoriStageProvider(model=args.model, base_url=args.base_url, api_key_env="SATORI_API_KEY", timeout=90)
    provider = ResumableSatoriStageProvider(live_provider, resume_from=args.resume_from, trace_path=out / "api-calls.jsonl")
    algorithm = SatoriMethodSession(plans, provider)
    meta = {"status": "running", "synthetic": False, "session_id": source["session_id"] + "_satori_staged",
            "provider": "satori-staged-api", "model": args.model, "endpoint": args.base_url,
            "started_utc": datetime.now(timezone.utc).isoformat(), "source_kind": "public_dataset_video",
            "source_metadata": str(root / "Research/Data/real-task/source.json"),
            "operator_policy": "Script accepts a model-proposed goal and initial step on the following frame. It NEVER confirms any step completed. Not a human study.",
            "method_scope": "Satori structure reimplementation with unified VLM, adapted cleaning subtask, text only and synchronous prefetch; not exact original implementation or latency replication.",
            "inputs": {"session_sha256": file_hash(session_path), "task_catalog_sha256": file_hash(out / "task-catalog.json"),
                       "frame_id_policy": "Opaque sequential IDs; original dataset names are never sent as frame metadata.",
                       "catalog_policy": "Uniform task/step/checkpoint IDs and semantic fields only; no source paths, hashes or adaptation notes in model payload."}}
    provider.validate_run_inputs(meta["inputs"])
    meta["resume"] = provider.resume_metadata
    meta["call_count_policy"] = "api_calls and real_api_calls count this run network attempts; cached_calls are exact historical API-success reuses; stage_calls is their sum."
    save(out / "run.json", meta)
    records, flat = [], []
    try:
        with (out / "method-timeline.jsonl").open("w", encoding="utf-8") as log:
            for i, frame in enumerate(frames):
                events = []
                if records:
                    pending = records[-1].get("pending_confirmation")
                    if pending and pending["kind"] == "confirm_task":
                        events = [{"timestamp_ms": frame["timestamp_ms"], "type": "confirm_task", "task_id": pending["task_id"]}]
                    elif pending and pending["kind"] == "accept_step" and records[-1]["active_step_id"] is None:
                        events = [{"timestamp_ms": frame["timestamp_ms"], "type": "accept_step", "step_id": pending["step_id"]}]
                record = algorithm.process_frame(frames[max(0, i-3):i+1], frame["timestamp_ms"], events)
                item = flatten(record, frame, tasks, records[-1] if records else None)
                records.append(record);flat.append(item)
                log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n");log.flush()
                print(json.dumps({"frame": i, "phase": record["phase"], "active_step": record["active_step_id"],
                                  "stages": [call["stage"] for call in record["stage_calls"]]}, ensure_ascii=True), flush=True)
        provider.assert_cache_consumed()
        meta["status"] = "completed"
    except KeyboardInterrupt:
        meta.update(status="aborted", error_type="KeyboardInterrupt", error="Experiment interrupted by operator")
        raise
    except Exception as error:
        meta.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        meta.update(finished_utc=datetime.now(timezone.utc).isoformat(), processed_frames=len(records), api_calls=provider.real_api_calls,
                    real_api_calls=provider.real_api_calls, cached_calls=provider.cached_calls, stage_calls=len(provider.traces))
        save(out / "run.json", meta)
        save(out / "unity-report.json", {"session_id": meta["session_id"], "synthetic": False,
              "source_kind": "public_dataset_video", "source_label": "公开数据集视频（非用户实拍）", "records": flat})
        save(out / "summary.json", {"status": meta["status"], "frames": len(records), "api_calls": provider.real_api_calls,
              "real_api_calls": provider.real_api_calls, "cached_calls": provider.cached_calls, "stage_calls": len(provider.traces), "resume": provider.resume_metadata,
              "final_phase": records[-1]["phase"] if records else None,
              "operator_confirmed_completed_steps": records[-1]["confirmed_completed_step_ids"] if records else [],
              "interpretation": "Real-image API integration test with scripted operator acceptance; no accuracy or user-study claim."})


if __name__ == "__main__":
    main()
