from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "Tools" / "run_satori_api_experiment.py"
SPEC = importlib.util.spec_from_file_location("satori_experiment_projection", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def inputs():
    frame = {"frame_id": "now", "timestamp_ms": 3000, "image_path": "frame.jpg"}
    task = {"task_id": "t", "title": "Task", "steps": [{"step_id": "s", "instruction": "Do step"}]}
    belief = {"scene": "desk", "action": "act", "confidence": .9, "uncertain": False,
              "evidence_frame_ids": ["now"]}
    record = {"timestamp_ms": 3000, "confirmed_task_id": "t", "desire_candidate": None,
              "belief": belief, "active_step_id": "s", "pending_confirmation": None,
              "phase": "active_step", "guidance": None, "confirmed_completed_step_ids": [],
              "stage_calls": [{"stage": "belief", "response": deepcopy(belief)}]}
    return record, frame, {"t": task}


def test_cached_guidance_keeps_its_original_evidence_and_generation_time():
    record, frame, tasks = inputs()
    guidance = {"step_id": "s", "text": "Cached instruction", "modalities": ["text"], "evidence_frame_ids": ["old"]}
    record["guidance"] = guidance
    previous = {"timestamp_ms": 2000, "guidance": None, "cached_forecast": {
        "target_step_id": "s", "generated_at_ms": 1000, "guidance": deepcopy(guidance)}}
    original = deepcopy((record, frame, tasks, previous))
    flat = MODULE.flatten(record, frame, tasks, previous)
    assert flat["decision_kind"] == "guide"
    assert flat["evidence_frame_ids"] == ["old"]
    assert flat["evidence_source"] == "guidance"
    assert flat["evidence_timestamp_ms"] == 1000
    assert (record, frame, tasks, previous) == original
    flat["evidence_frame_ids"].append("mutated")
    flat["confirmed_completed_step_ids"].append("s")
    assert (record, frame, tasks, previous) == original


@pytest.mark.parametrize("pending_kind, source, expected_ids, expected_time", [
    ("confirm_task", "desire", ["goal_frame"], 3000),
    ("accept_step", "forecast", ["forecast_frame"], 1000),
    ("confirm_step", "checkpoints", ["first_checkpoint", "last_checkpoint"], 2000),
])
def test_pending_confirmation_uses_its_own_inference_evidence(pending_kind, source, expected_ids, expected_time):
    record, frame, tasks = inputs()
    if pending_kind == "confirm_task":
        proposal = {"task_id": "t", "confidence": .9, "evidence_frame_ids": ["goal_frame"]}
        record.update(confirmed_task_id=None, desire_candidate=proposal,
                      pending_confirmation={"kind": pending_kind, "task_id": "t"})
        record["stage_calls"].append({"stage": "desire", "response": deepcopy(proposal)})
    elif pending_kind == "accept_step":
        record.update(active_step_id=None, pending_confirmation={"kind": pending_kind, "step_id": "s"},
                      cached_forecast={"target_step_id": "s", "generated_at_ms": 1000,
                                       "forecast": {"evidence_frame_ids": ["forecast_frame"]}})
    else:
        record["pending_confirmation"] = {"kind": pending_kind, "step_id": "s"}
        record["checkpoint_statuses"] = [
            {"checkpoint_id": "c1", "status": "COMPLETED", "timestamp_ms": 1000, "evidence_frame_ids": ["first_checkpoint"]},
            {"checkpoint_id": "c2", "status": "COMPLETED", "timestamp_ms": 2000, "evidence_frame_ids": ["last_checkpoint"]},
        ]
        record["latest_checkpoint_detection"] = {"evidence_frame_ids": ["now"], "overall_status": "UNKNOWN"}
    original = deepcopy(record)
    flat = MODULE.flatten(record, frame, tasks, None)
    assert flat["evidence_source"] == source
    assert flat["evidence_frame_ids"] == expected_ids
    assert flat["evidence_timestamp_ms"] == expected_time
    assert record == original


def test_observe_uses_belief_but_human_completion_does_not_invent_visual_evidence():
    record, frame, tasks = inputs()
    observed = MODULE.flatten(record, frame, tasks, None)
    assert observed["evidence_source"] == "belief"
    assert observed["evidence_frame_ids"] == ["now"]
    record.update(phase="complete", events=[{"timestamp_ms": 2900, "type": "confirm_step", "step_id": "s"}])
    completed = MODULE.flatten(record, frame, tasks, None)
    assert completed["evidence_source"] == "user_confirmation"
    assert completed["evidence_frame_ids"] == []
    assert completed["evidence_timestamp_ms"] == 2900


def test_unknown_guidance_generation_time_stays_null_instead_of_using_current_frame():
    record, frame, tasks = inputs()
    record["guidance"] = {"step_id": "s", "text": "Cached instruction", "evidence_frame_ids": ["old"]}
    flat = MODULE.flatten(record, frame, tasks, None)
    assert flat["evidence_frame_ids"] == ["old"]
    assert flat["evidence_timestamp_ms"] is None


def test_source_task_keeps_quantities_start_checks_and_task_object_semantics():
    root = SCRIPT.parents[1]
    plan = root / "Research/Reference/satori-assistance/pipelines/task/task_plans/coffee.py"
    task = MODULE.source_task(plan, "source_coffee")
    first = task["steps"][0]
    assert "11g" in first["checkpoints"][0]["question"]
    assert "Do you see coffee beans on a scale?" in first["checkpoints"][0]["question"]
    assert first["is_start"] is True
    assert first["step_check_prompts"] == ["Are there coffee beans visible on a scale?"]
    assert task["steps"][1]["is_start"] is False
    assert "Scale" in task["objects"]
    assert first["required_objects"] == []
    assert first["required_objects_status"] == "unknown"
    assert "unknown" in task["adaptation"].lower()
    assert set(first["checkpoints"][0]) == {"checkpoint_id", "question"}


def test_anonymized_model_catalog_keeps_semantics_without_local_provenance():
    root = SCRIPT.parents[1]
    plans = MODULE.catalog(root)
    original = deepcopy(plans)
    model = MODULE.model_catalog(plans)
    first = model[1]["steps"][0]
    assert "11g" in first["checkpoints"][0]["question"]
    assert first["is_start"] is True
    assert first["step_check_prompts"] == plans[1]["steps"][0]["step_check_prompts"]
    assert first["required_objects_status"] == "unknown"
    assert model[1]["objects"] == plans[1]["objects"]
    assert first["step_id"] == "t001_s000"
    assert first["checkpoints"][0]["checkpoint_id"] == "t001_s000_c000"
    import json
    encoded = json.dumps(model)
    for private in ("source_path", "source_sha256", "adaptation", "source_coffee", "egodex"):
        assert private not in encoded
    assert plans == original
    model[1]["objects"].append("mutated")
    first["step_check_prompts"].append("mutated")
    assert plans == original


@pytest.mark.parametrize("mutation", ["missing_start", "numeric_start", "missing_instruction", "missing_check", "bad_objects", "empty_checkpoints"])
def test_source_plan_missing_semantics_is_rejected_instead_of_silently_dropped(tmp_path, mutation):
    task = {"desired_task": "Task", "objects": ["Cup"], "steps": [{"is_start": True,
        "content": "Measure beans", "step_check_prompts": ["Is a scale visible?"],
        "checkpoints": [{"instruction": "Measure 11g", "blip_prompt": "Are beans on scale?"}]}]}
    if mutation == "missing_start": del task["steps"][0]["is_start"]
    elif mutation == "numeric_start": task["steps"][0]["is_start"] = 1
    elif mutation == "missing_instruction": del task["steps"][0]["checkpoints"][0]["instruction"]
    elif mutation == "missing_check": del task["steps"][0]["step_check_prompts"]
    elif mutation == "bad_objects": task["objects"] = "Cup"
    elif mutation == "empty_checkpoints": task["steps"][0]["checkpoints"] = []
    path = tmp_path / "plan.py"
    path.write_text("TASK = " + repr(task), encoding="utf-8")
    with pytest.raises(ValueError):
        MODULE.source_task(path, "t")
