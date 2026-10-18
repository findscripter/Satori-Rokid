"""Exact eight-frame reconstruction and bounded error diagnostics, without network."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from urllib.error import HTTPError

import cv2
import numpy as np
import pytest

TOOLS=Path(__file__).resolve().parents[3]/"Tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0,str(TOOLS))
SPEC=importlib.util.spec_from_file_location("diagnose_gt_failure",TOOLS/"diagnose_gt_failure.py")
diagnostic=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=diagnostic
SPEC.loader.exec_module(diagnostic)


def dump(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def original(tmp_path):
    data_dir=tmp_path/"data";data_dir.mkdir()
    ok,encoded=cv2.imencode(".png",np.zeros((6,6,3),dtype=np.uint8));assert ok
    image=data_dir/"source-label-not-for-model.png";image.write_bytes(encoded.tobytes())
    sha=hashlib.sha256(image.read_bytes()).hexdigest()
    frames=[{"frame_id":f"v{i:03d}","timestamp_ms":i*1000,"image_path":image.name,"sha256":sha} for i in range(8)]
    manifest={"schema_version":1,"benchmark_id":"frozen-test","samples":[{"sample_id":"e003","frames":frames}]}
    taxonomy={"schema_version":1,"scenes":[{"id":f"s{i:03d}","label":f"Scene {i}"} for i in range(4)],
              "actions":[{"id":f"c{i:03d}","label":f"Action {i}"} for i in range(157)]}
    manifest_path=data_dir/"manifest.json";manifest_sha=dump(manifest_path,manifest)
    source=tmp_path/"original-run";source.mkdir()
    snapshot_sha=dump(source/"manifest.json",manifest)
    taxonomy_sha=dump(source/"taxonomy.json",taxonomy)
    payload={"scene_candidates":taxonomy["scenes"],"action_candidates":taxonomy["actions"],
             "frames":[{"frame_id":f["frame_id"],"timestamp_ms":f["timestamp_ms"]} for f in frames]}
    prompt_sha=hashlib.sha256(diagnostic.benchmark.PROMPT.encode()).hexdigest()
    payload_sha=hashlib.sha256(diagnostic.benchmark._json_text(payload).encode()).hexdigest()
    trace={"sample_id":"e003","stage":"charades_ego_classification","status":"error","http_status":502,
           "requested_model":"Mage-VL","system_prompt_sha256":prompt_sha,"payload_sha256":payload_sha,
           "image_ids":[f["frame_id"] for f in frames],"image_sha256":{f["frame_id"]:sha for f in frames}}
    (source/"api-calls.jsonl").write_text(json.dumps(trace)+"\n",encoding="utf-8")
    dump(source/"run.json",{"synthetic":False,"model":"Mage-VL","endpoint":diagnostic.ENDPOINT,
         "prompt_sha256":prompt_sha,"source_manifest_sha256":manifest_sha,
         "manifest_snapshot_sha256":snapshot_sha,"taxonomy_snapshot_sha256":taxonomy_sha})
    return source,manifest_path


def setup_transport(monkeypatch,body):
    calls=[]
    class Opener:
        def open(self,req,**kwargs):
            calls.append((req.full_url,req.data))
            raise HTTPError(req.full_url,502,"upstream",{},io.BytesIO(body))
    monkeypatch.setattr(diagnostic.benchmark.request,"build_opener",lambda *handlers:Opener())
    monkeypatch.setattr(diagnostic,"ensure_service",lambda root:None)
    monkeypatch.setattr(diagnostic,"fetch_service_token",lambda:"test-only-diagnostic-secret")
    return calls


def test_one_original_request_records_only_safe_error_metadata(original,tmp_path,monkeypatch):
    source,manifest=original
    body=json.dumps({"error":{"code":"incomplete_json","finish_reason":"length",
                             "message":"test-only-diagnostic-secret BASE64_MUST_NOT_PERSIST"},
                     "usage":{"prompt_tokens":4146,"completion_tokens":1024,"total_tokens":5170},
                     "headers":{"Authorization":"test-only-diagnostic-secret"},
                     "base64":"BASE64_MUST_NOT_PERSIST"}).encode()
    calls=setup_transport(monkeypatch,body)
    original_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    out=tmp_path/"diagnostic"
    result=diagnostic.diagnose(source,manifest,out)
    assert len(calls)==1 and result["model_requests"]==1
    assert result["diagnosis"]=="output_token_limit"
    assert result["http_status"]==502 and result["server_finish_reason"]=="length"
    assert result["server_usage"]["completion_tokens"]==1024
    assert result["original_fingerprints_match"] is True and result["wire_request_verified"] is True
    assert original_hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    for path in out.iterdir():
        content=path.read_text(encoding="utf-8")
        assert "test-only-diagnostic-secret" not in content
        assert "BASE64_MUST_NOT_PERSIST" not in content and "Authorization" not in content
    assert not any(key.startswith("SATORI_GT_DIAG_") for key in diagnostic.os.environ)


@pytest.mark.parametrize("change",["prompt","payload","image"])
def test_changed_original_inputs_fail_before_ssh_or_model(original,tmp_path,monkeypatch,change):
    source,manifest=original
    touched=[]
    monkeypatch.setattr(diagnostic,"ensure_service",lambda root:touched.append("health"))
    monkeypatch.setattr(diagnostic,"fetch_service_token",lambda:touched.append("token"))
    path=source/"api-calls.jsonl"
    trace=json.loads(path.read_text())
    if change=="prompt":trace["system_prompt_sha256"]="0"*64
    elif change=="payload":trace["payload_sha256"]="0"*64
    else:trace["image_sha256"]["v000"]="0"*64
    path.write_text(json.dumps(trace)+"\n")
    with pytest.raises(ValueError):
        diagnostic.diagnose(source,manifest,tmp_path/"out")
    assert touched==[]


def test_oversized_error_body_is_discarded_without_retry(original,tmp_path,monkeypatch):
    source,manifest=original
    calls=setup_transport(monkeypatch,b"RAW_BODY_MUST_NOT_PERSIST"*4000)
    result=diagnostic.diagnose(source,manifest,tmp_path/"out")
    assert len(calls)==1 and result["error_body_status"]=="too_large"
    assert result["diagnosis"]=="unclassified_http_failure"
    assert "RAW_BODY_MUST_NOT_PERSIST" not in (tmp_path/"out"/"diagnostic.json").read_text()


def test_old_output_directory_cannot_be_overwritten(original,tmp_path,monkeypatch):
    source,manifest=original
    calls=setup_transport(monkeypatch,b"{}")
    out=tmp_path/"out";out.mkdir()
    with pytest.raises(FileExistsError):
        diagnostic.diagnose(source,manifest,out)
    assert calls==[]
