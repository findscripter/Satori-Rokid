"""One-shot context Q&A runner tests; all providers are in-process test doubles."""

import copy
import hashlib
import json

import cv2
import numpy as np
import pytest

from satori_repro.context_runner import run_context_question


ANSWER = {
    "answer": "你正在桌边擦拭桌面。",
    "environment": {"place_type": "室内桌边", "visible_objects": ["桌子", "清洁布"]},
    "activity": {"action": "擦拭桌面", "phase": "in_progress"},
    "uncertainties": ["无法确认清洁是否完成。"],
    "evidence_frame_ids": ["v002"],
    "sensor_evidence_ids": [],
}


class Provider:
    name = "local-test"
    model = "explicit-context-model"
    base_url = "http://127.0.0.1:9999/v1"
    synthetic = False
    trace_path = None
    def __init__(self, fail=False):
        self.calls = []
        self.traces = []
        self.fail = fail
    def complete_json(self, stage, frames, payload):
        self.calls.append((stage, copy.deepcopy(frames), copy.deepcopy(payload)))
        trace={"stage":stage,"status":"error" if self.fail else "success",
               "requested_model":self.model,"returned_model":self.model,
               "usage":{"total_tokens":17},"image_ids":[f["frame_id"] for f in frames]}
        self.traces.append(trace)
        if self.fail:
            raise RuntimeError("DO_NOT_LOG_PRIVATE_PROVIDER_ERROR")
        result=copy.deepcopy(ANSWER)
        result["evidence_frame_ids"]=[frames[-1]["frame_id"]]
        return result


@pytest.fixture
def session(tmp_path):
    directory=tmp_path/"source"
    directory.mkdir()
    frames=[]
    for i in range(4):
        path=directory/f"label_cleaning_{i}.png"
        ok,encoded=cv2.imencode(".png",np.full((6,6,3),i*40,dtype=np.uint8))
        assert ok
        path.write_bytes(encoded.tobytes())
        frames.append({"frame_id":f"label_cleaning_{i}","timestamp_ms":i*1000,"image_path":path.name})
    path=directory/"session.json"
    path.write_text(json.dumps({"session_id":"leaking_dataset_task_name","frames":frames,
        "events":[{"timestamp_ms":0,"type":"confirm_task","task_id":"should_not_be_sent"}]}),encoding="utf-8")
    return path


def sample(sample_id,timestamp):
    return {"sample_id":sample_id,"timestamp_ms":timestamp,
            "accelerometer_m_s2":[0.0,0.0,9.81],"gyroscope_rad_s":[0.1,0.0,0.0],
            "mount":"head","source":"recorded"}


def test_selects_causal_window_anonymizes_ids_and_writes_audit_outputs(session,tmp_path):
    out=tmp_path/"out"
    provider=Provider()
    result=run_context_question(session,"我现在在哪、在做什么？",provider,out,window_size=2,at_ms=2500)
    assert result["status"]=="completed" and len(provider.calls)==1
    stage,frames,payload=provider.calls[0]
    assert stage=="context_question"
    assert [f["frame_id"] for f in frames]==["v001","v002"]
    assert [f["timestamp_ms"] for f in frames]==[1000,2000]
    assert all(str(session.parent) in f["image_path"] for f in frames)
    assert "task_catalog" not in json.dumps(payload) and "should_not_be_sent" not in json.dumps(payload)
    assert "leaking_dataset_task_name" not in json.dumps(payload)
    run=json.loads((out/"run.json").read_text(encoding="utf-8"))
    assert run["synthetic"] is False and run["source_kind"]=="unspecified_session_images"
    assert run["status"]=="completed" and run["api_calls"]==1
    assert run["model"]==provider.model and run["usage"][0]["total_tokens"]==17
    assert run["missing_sensors"]==["imu"]
    answer=json.loads((out/"answer.json").read_text(encoding="utf-8"))
    assert answer["activity"]["phase"]=="in_progress"
    context=json.loads((out/"input-context.json").read_text(encoding="utf-8"))
    assert context["inputs"]["session_sha256"]==hashlib.sha256(session.read_bytes()).hexdigest()
    assert len(context["frames"])==2
    for frame in context["frames"]:
        assert frame["sha256"]==hashlib.sha256(__import__("pathlib").Path(frame["image_path"]).read_bytes()).hexdigest()
    assert "base64" not in (out/"input-context.json").read_text(encoding="utf-8")
    assert (out/"api-calls.jsonl").is_file()
    assert not (out/"unity-report.json").exists()


def test_sensor_input_is_hashed_anonymized_and_filtered_to_image_window(session,tmp_path):
    sensors=session.parent/"imu.json"
    sensors.write_text(json.dumps({"samples":[sample("activity_label_old",0),
        sample("activity_label_one",1000),sample("activity_label_two",2000),
        sample("future_activity_label",3000)]}),encoding="utf-8")
    out=tmp_path/"out"
    result=run_context_question(session,"正在做什么？",Provider(),out,window_size=2,at_ms=2000,sensors_path="imu.json")
    assert result["status"]=="completed"
    context=json.loads((out/"input-context.json").read_text(encoding="utf-8"))
    assert context["inputs"]["sensors_sha256"]==hashlib.sha256(sensors.read_bytes()).hexdigest()
    assert [s["sample_id"] for s in context["sensor_samples"]]==["m001","m002"]
    assert [s["timestamp_ms"] for s in context["sensor_samples"]]==[1000,2000]
    assert context["assistant_context"]["sensor_context"]["status"]=="provided"
    run=json.loads((out/"run.json").read_text(encoding="utf-8"))
    assert run["missing_sensors"]==[]


def test_defaults_to_latest_frame_without_inventing_sensor_data(session,tmp_path):
    provider=Provider()
    run_context_question(session,"环境是什么？",provider,tmp_path/"out",window_size=1)
    assert provider.calls[0][1][0]["frame_id"]=="v003"
    context=json.loads((tmp_path/"out"/"input-context.json").read_text(encoding="utf-8"))
    assert context["sensor_samples"] is None
    assert context["assistant_context"]["missing_sensors"]==["imu"]


def test_failure_retains_sanitized_failed_run_and_api_trace(session,tmp_path):
    out=tmp_path/"failed"
    provider=Provider(fail=True)
    with pytest.raises(RuntimeError,match="Context question failed"):
        run_context_question(session,"当前活动是什么？",provider,out)
    run=json.loads((out/"run.json").read_text(encoding="utf-8"))
    assert run["status"]=="failed" and run["api_calls"]==1
    assert not (out/"answer.json").exists()
    assert len((out/"api-calls.jsonl").read_text().splitlines())==1
    assert "DO_NOT_LOG_PRIVATE_PROVIDER_ERROR" not in (out/"run.json").read_text()
    assert len(provider.calls)==1


def test_existing_output_is_never_overwritten(session,tmp_path):
    out=tmp_path/"out"
    out.mkdir()
    marker=out/"run.json"
    marker.write_text("preserve")
    provider=Provider()
    with pytest.raises(FileExistsError):
        run_context_question(session,"活动？",provider,out)
    assert marker.read_text()=="preserve" and provider.calls==[]


@pytest.mark.parametrize("kwargs",[{"window_size":0},{"window_size":True},{"at_ms":-1},{"at_ms":True}])
def test_invalid_options_fail_without_network(session,tmp_path,kwargs):
    provider=Provider()
    with pytest.raises(ValueError):
        run_context_question(session,"活动？",provider,tmp_path/"out",**kwargs)
    assert provider.calls==[]


def test_missing_image_and_empty_eligible_window_leave_failed_metadata(session,tmp_path):
    data=json.loads(session.read_text())
    data["frames"][0]["timestamp_ms"]=500
    session.write_text(json.dumps(data))
    provider=Provider()
    with pytest.raises(RuntimeError):
        run_context_question(session,"活动？",provider,tmp_path/"before",at_ms=100)
    assert json.loads((tmp_path/"before"/"run.json").read_text())["status"]=="failed"
    (session.parent/"label_cleaning_3.png").unlink()
    with pytest.raises(RuntimeError):
        run_context_question(session,"活动？",provider,tmp_path/"missing",window_size=1)
    assert json.loads((tmp_path/"missing"/"run.json").read_text())["status"]=="failed"
    assert provider.calls==[]


def test_invalid_sensors_file_is_not_silently_treated_as_missing(session,tmp_path):
    sensors=session.parent/"imu.json"
    sensors.write_text(json.dumps({"samples":[{"sample_id":"bad","timestamp_ms":2000}]}))
    with pytest.raises(RuntimeError):
        run_context_question(session,"活动？",Provider(),tmp_path/"out",sensors_path=sensors)
    assert json.loads((tmp_path/"out"/"run.json").read_text())["status"]=="failed"
