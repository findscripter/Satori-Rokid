"""Unity bridge tests use generated fixtures and in-process/mock transports only."""

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import uuid

import cv2
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("unity_mage_bridge", Path(__file__).resolve().parents[3] / "Tools/unity_mage_request.py")
bridge = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bridge
SPEC.loader.exec_module(bridge)


TEST_TOKEN = "test-only-unity-token-01234567890123456789"


@pytest.fixture
def project(tmp_path):
    (tmp_path/"UserSettings/SatoriMage").mkdir(parents=True)
    source=tmp_path/"input"
    source.mkdir()
    frames=[]
    for i in range(4):
        image=source/f"frame_{i}.png"
        ok,encoded=cv2.imencode(".png",np.full((8,8,3),i*40,dtype=np.uint8))
        assert ok
        image.write_bytes(encoded.tobytes())
        frames.append({"frame_id":f"original-cleaning-{i}","timestamp_ms":i*1000,"image_path":image.name})
    (source/"session.json").write_text(json.dumps({"session_id":"test-private-name","frames":frames,"events":[]}),encoding="utf-8")
    return tmp_path


def request_files(project, **patch):
    request_id=str(uuid.uuid4())
    request={"request_id":request_id,"question":"我的双手在做什么？","session_path":"input/session.json","at_ms":2000,"window":2}
    request.update(patch)
    folder=project/"UserSettings/SatoriMage"
    source=folder/f"{request_id}.request.json"
    target=folder/f"{request_id}.response.json"
    source.write_text(json.dumps(request,ensure_ascii=False),encoding="utf-8")
    return source,target,request


class Provider:
    name="mock-stage"
    model="Mage-VL"
    base_url="http://127.0.0.1:18866/v1"
    synthetic=False
    def __init__(self, *, echo_key=False, **kwargs):
        self.traces=[]
        self.echo_key=echo_key
        self.key_env=kwargs["api_key_env"]
    def complete_json(self,stage,frames,payload):
        answer={"answer":TEST_TOKEN if self.echo_key else "你的手正在擦拭桌面。",
                "environment":{"place_type":"室内桌边","visible_objects":["桌面","布"]},
                "activity":{"action":"擦拭桌面","phase":"in_progress"},
                "uncertainties":["无法确定完整清洁任务是否结束。"],
                "evidence_frame_ids":[frames[-1]["frame_id"]],"sensor_evidence_ids":[]}
        self.traces.append({"stage":stage,"status":"success","raw_json":json.loads(json.dumps(answer)),
                            "image_ids":[frame["frame_id"] for frame in frames],
                            "image_sha256":{frame["frame_id"]:hashlib.sha256(Path(frame["image_path"]).read_bytes()).hexdigest() for frame in frames},
                            "requested_model":self.model,"returned_model":self.model,"usage":{"total_tokens":19}})
        return answer


def mock_remote(monkeypatch):
    calls=[]
    monkeypatch.setattr(bridge,"ensure_service",lambda root:calls.append("health"))
    monkeypatch.setattr(bridge,"fetch_service_token",lambda: calls.append("token") or TEST_TOKEN)
    monkeypatch.setattr(bridge,"SatoriStageProvider",Provider)
    return calls


def test_success_is_new_run_with_raw_answer_matching_evidence_and_no_key(project,monkeypatch):
    calls=mock_remote(monkeypatch)
    source,target,request=request_files(project)
    response=bridge.handle_request(source,target,project_root=project)
    assert response["status"]=="completed" and calls==["health","token"]
    assert response["request_id"]==request["request_id"] and response["question"]==request["question"]
    assert response["model"]=="Mage-VL" and response["latency_ms"]>=0
    assert [f["frame_id"] for f in response["input_frames"]]==["v001","v002"]
    assert [f["timestamp_ms"] for f in response["input_frames"]]==[1000,2000]
    assert response["answer_payload"]["evidence_frame_ids"]==["v002"]
    assert response["missing_sensors"]==["imu"] and response["usage"]==[{"total_tokens":19}]
    run=Path(response["run_directory"])
    assert run.is_relative_to(project/"Research/Backend/runs")
    assert response["raw_answer_json"]==(run/"answer.json").read_bytes().decode("utf-8")
    assert json.loads(response["raw_answer_json"])==response["answer_payload"]
    assert json.loads(target.read_text(encoding="utf-8"))==response
    assert TEST_TOKEN not in target.read_text(encoding="utf-8")
    assert not any(name.startswith("SATORI_UNITY_MAGE_") for name in bridge.os.environ)


@pytest.mark.parametrize("patch",[
    {"request_id":"../../unsafe"},{"window":0},{"window":33},{"window":True},
    {"at_ms":-1},{"at_ms":False},{"question":""},{"question":"x"*2001},{"extra":True},
])
def test_bad_request_is_rejected_before_any_remote_access(project,monkeypatch,patch):
    calls=mock_remote(monkeypatch)
    source,target,_=request_files(project,**patch)
    response=bridge.handle_request(source,target,project_root=project)
    assert response["status"]=="failed" and calls==[]
    assert response["answer_payload"] is None and response["raw_answer_json"] is None
    assert response["input_frames"]==[] and response["run_directory"] is None
    assert target.is_file()


def test_missing_selected_image_fails_before_health_or_ssh(project,monkeypatch):
    calls=mock_remote(monkeypatch)
    (project/"input/frame_2.png").unlink()
    source,target,_=request_files(project)
    response=bridge.handle_request(source,target,project_root=project)
    assert response["status"]=="failed" and response["error_code"]=="invalid_session"
    assert calls==[]


def test_io_paths_cannot_escape_or_overwrite_request(project,monkeypatch):
    calls=mock_remote(monkeypatch)
    source,target,_=request_files(project)
    original=source.read_bytes()
    with pytest.raises(bridge.BridgeError):
        bridge.handle_request(source,project/"outside.json",project_root=project)
    with pytest.raises(bridge.BridgeError):
        bridge.handle_request(source,source,project_root=project)
    assert source.read_bytes()==original and calls==[]


def test_existing_response_is_preserved_and_no_model_request_occurs(project,monkeypatch):
    calls=mock_remote(monkeypatch)
    source,target,_=request_files(project)
    target.write_text("previous response")
    with pytest.raises(bridge.BridgeError):
        bridge.handle_request(source,target,project_root=project)
    assert target.read_text()=="previous response" and calls==[]


def test_service_failure_is_explicit_and_does_not_reuse_answer(project,monkeypatch):
    calls=mock_remote(monkeypatch)
    def failed_health(root):
        raise bridge.BridgeError("service_unavailable","服务当前不可用。")
    monkeypatch.setattr(bridge,"ensure_service",failed_health)
    source,target,_=request_files(project)
    response=bridge.handle_request(source,target,project_root=project)
    assert response["status"]=="failed" and response["error_code"]=="service_unavailable"
    assert response["answer_payload"] is None and calls==[]


def test_credential_echo_fails_before_answer_is_written_and_traces_are_redacted(project,monkeypatch):
    mock_remote(monkeypatch)
    monkeypatch.setattr(bridge,"SatoriStageProvider",lambda **kwargs:Provider(echo_key=True,**kwargs))
    source,target,_=request_files(project)
    response=bridge.handle_request(source,target,project_root=project)
    assert response["status"]=="failed" and response["answer_payload"] is None
    run=Path(response["run_directory"])
    assert not (run/"answer.json").exists()
    for file in [target,*run.glob("*.json"),*run.glob("*.jsonl")]:
        assert TEST_TOKEN not in file.read_text(encoding="utf-8")
    assert not any(name.startswith("SATORI_UNITY_MAGE_") for name in bridge.os.environ)


def test_subprocess_helpers_hide_windows_and_do_not_echo_captured_token(monkeypatch,tmp_path,capsys):
    class Result:
        returncode=0
        stdout=TEST_TOKEN.encode("ascii")+b"\n"
        stderr=b""
    commands=[]
    monkeypatch.setattr(bridge.shutil,"which",lambda name:"C:/test/"+name)
    monkeypatch.setattr(bridge.subprocess,"run",lambda command,**kwargs:commands.append((command,kwargs)) or Result())
    assert bridge.fetch_service_token()==TEST_TOKEN
    command,kwargs=commands[0]
    assert command[-2:]==["gpu-host","cat /home/gpu-user/work/mage-vl-satori/.secrets/api.token"]
    assert kwargs["creationflags"]==getattr(bridge.subprocess,"CREATE_NO_WINDOW",0)
    assert kwargs["stdout"]==bridge.subprocess.PIPE and kwargs["stderr"]==bridge.subprocess.PIPE
    assert capsys.readouterr().out==""


def test_cli_stdout_is_only_status_not_answer_question_or_token(project,monkeypatch,capsys):
    mock_remote(monkeypatch)
    source,target,request=request_files(project)
    monkeypatch.setattr(bridge,"PROJECT_ROOT",project)
    assert bridge.main(["--request",str(source),"--response",str(target)])==0
    output=capsys.readouterr().out
    assert json.loads(output)["status"]=="completed"
    assert request["question"] not in output and TEST_TOKEN not in output and "擦拭" not in output


@pytest.mark.parametrize("mismatch",[
    "wrong_returned_model","wrong_image_hash","missing_image_hash","wrong_image_order",
    "different_raw_answer","extra_api_call","unsuccessful_call","no_trace",
])
def test_success_requires_exact_single_real_trace_for_this_answer(project,monkeypatch,mismatch):
    mock_remote(monkeypatch)
    class InconsistentProvider(Provider):
        def complete_json(self,stage,frames,payload):
            answer=super().complete_json(stage,frames,payload)
            trace=self.traces[0]
            if mismatch=="wrong_returned_model":
                trace["returned_model"]="another-model"
            elif mismatch=="wrong_image_hash":
                trace["image_sha256"][frames[0]["frame_id"]]="0"*64
            elif mismatch=="missing_image_hash":
                trace.pop("image_sha256")
            elif mismatch=="wrong_image_order":
                trace["image_ids"].reverse()
            elif mismatch=="different_raw_answer":
                trace["raw_json"]["answer"]="A different answer from the API trace."
            elif mismatch=="extra_api_call":
                self.traces.append(dict(trace))
            elif mismatch=="unsuccessful_call":
                trace["status"]="error"
            elif mismatch=="no_trace":
                self.traces.clear()
            return answer
    monkeypatch.setattr(bridge,"SatoriStageProvider",InconsistentProvider)
    source,target,_=request_files(project)
    response=bridge.handle_request(source,target,project_root=project)
    assert response["status"]=="failed"
    assert response["error_code"]=="invalid_artifacts"
    assert response["answer_payload"] is None and response["raw_answer_json"] is None
    assert response["input_frames"]==[]
    assert json.loads(target.read_text(encoding="utf-8"))["status"]=="failed"
