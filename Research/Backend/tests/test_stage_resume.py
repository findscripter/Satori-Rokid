"""Resume uses exact fingerprints and a real-response prefix; tests use local doubles."""

import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from satori_repro.satori_method import build_stage_prompt
from satori_repro.stage_provider import _CHINESE_FIELDS, _json_text
from satori_repro.stage_resume import ResumableSatoriStageProvider


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def success(stage, frames, payload, model="explicit-model"):
    return {"stage": stage, "status": "success", "requested_model": model,
            "returned_model": model, "usage": {"total_tokens": 7}, "latency_ms": 3.5,
            "image_ids": [f["frame_id"] for f in frames],
            "image_sha256": {f["frame_id"]: digest(Path(f["image_path"]).read_bytes()) for f in frames},
            "system_prompt_sha256": digest((build_stage_prompt(stage, payload) + _CHINESE_FIELDS).encode()),
            "payload_sha256": digest(_json_text(payload).encode()),
            "raw_json": {"value": stage, "evidence_frame_ids": [f["frame_id"] for f in frames]}}


class Live:
    model = "explicit-model"
    base_url = "http://127.0.0.1:9999/v1"
    trace_path = None
    def __init__(self, fail=False):
        self.traces = []
        self.calls = []
        self.fail = fail
    def complete_json(self, stage, frames, payload):
        self.calls.append((stage, copy.deepcopy(frames), copy.deepcopy(payload)))
        trace = success(stage, frames, payload)
        if self.fail:
            trace.pop("raw_json")
            trace.update(status="error", http_status=429, error="http_error")
        self.traces.append(trace)
        if self.fail:
            raise RuntimeError("Satori stage API returned HTTP 429")
        return copy.deepcopy(trace["raw_json"])


@pytest.fixture
def frames(tmp_path):
    result=[]
    for i in range(2):
        image=tmp_path/f"{i}.png"
        ok, encoded=cv2.imencode(".png",np.full((5,5,3),i*100,dtype=np.uint8))
        assert ok
        image.write_bytes(encoded.tobytes())
        result.append({"frame_id":f"v{i:03d}","timestamp_ms":i*1000,"image_path":str(image)})
    return result


def source_run(tmp_path, traces, **overrides):
    source=tmp_path/"source"
    source.mkdir()
    run={"status":"failed","synthetic":False,"provider":"satori-staged-api",
         "model":"explicit-model","endpoint":Live.base_url,
         "inputs":{"session_sha256":"a"*64,"task_catalog_sha256":"b"*64}}
    run.update(overrides)
    (source/"run.json").write_text(json.dumps(run),encoding="utf-8")
    (source/"api-calls.jsonl").write_text("".join(json.dumps(t)+"\n" for t in traces),encoding="utf-8")
    return source


def test_success_prefix_is_reused_once_then_live_calls_continue(tmp_path,frames):
    first=success("belief",frames[:1],{})
    source=source_run(tmp_path,[first,{"status":"error","http_status":429}])
    live=Live()
    path=tmp_path/"new"/"api-calls.jsonl"
    provider=ResumableSatoriStageProvider(live,resume_from=source,trace_path=path)
    returned=provider.complete_json("belief",frames[:1],{})
    assert returned==first["raw_json"] and live.calls==[]
    returned["value"]="mutated"
    assert provider.traces[0]["raw_json"]==first["raw_json"]
    provider.complete_json("desire",frames,{"belief":"current"})
    provider.assert_cache_consumed()
    assert len(live.calls)==1 and provider.cached_calls==1 and provider.real_api_calls==1
    assert [t["call_source"] for t in provider.traces]==["cached","real"]
    assert provider.traces[0]["request_sent_this_run"] is False
    assert provider.traces[1]["request_sent_this_run"] is True
    assert provider.resume_metadata["source_trace_sha256"]==digest((source/"api-calls.jsonl").read_bytes())
    assert provider.resume_metadata["source_run_sha256"]==digest((source/"run.json").read_bytes())
    assert provider.traces[0]["cache_provenance"]["source_line"]==1
    assert len(path.read_text().splitlines())==2


@pytest.mark.parametrize("field,value",[
    ("requested_model","another-model"),("stage","desire"),
    ("image_ids",["v001","v000"]),("image_sha256",{"v000":"0"*64,"v001":"1"*64}),
    ("system_prompt_sha256","c"*64),("payload_sha256","d"*64),
])
def test_any_fingerprint_mismatch_fails_closed_without_live_call(tmp_path,frames,field,value):
    trace=success("belief",frames,{})
    trace[field]=value
    source=source_run(tmp_path,[trace])
    live=Live()
    provider=ResumableSatoriStageProvider(live,resume_from=source)
    with pytest.raises(ValueError,match="match"):
        provider.complete_json("belief",frames,{})
    assert live.calls==[] and provider.traces==[] and provider.cached_calls==0


def test_changed_image_bytes_are_not_reused(tmp_path,frames):
    source=source_run(tmp_path,[success("belief",frames,{})])
    image=Path(frames[0]["image_path"])
    ok,encoded=cv2.imencode(".png",np.full((5,5,3),200,dtype=np.uint8))
    assert ok
    image.write_bytes(encoded.tobytes())
    live=Live()
    provider=ResumableSatoriStageProvider(live,resume_from=source)
    with pytest.raises(ValueError,match="match"):
        provider.complete_json("belief",frames,{})
    assert live.calls==[]


def test_cached_prefix_must_be_fully_consumed_before_success(tmp_path,frames):
    source=source_run(tmp_path,[success("belief",frames,{})])
    provider=ResumableSatoriStageProvider(Live(),resume_from=source)
    with pytest.raises(ValueError,match="unconsumed"):
        provider.assert_cache_consumed()


def test_failure_boundary_prevents_reusing_later_success_and_no_retry(tmp_path,frames):
    source=source_run(tmp_path,[success("belief",frames,{}),
        {"status":"error","http_status":429},success("desire",frames,{})])
    live=Live(fail=True)
    provider=ResumableSatoriStageProvider(live,resume_from=source)
    provider.complete_json("belief",frames,{})
    with pytest.raises(RuntimeError,match="429"):
        provider.complete_json("desire",frames,{})
    assert len(live.calls)==1 and provider.real_api_calls==1 and provider.cached_calls==1
    assert provider.traces[-1]["call_source"]=="real" and provider.traces[-1]["status"]=="error"


@pytest.mark.parametrize("overrides",[
    {"synthetic":True},{"status":"completed"},{"provider":"replay"},
    {"model":"other"},{"endpoint":"https://other.example/v1"},
])
def test_nonreal_or_incompatible_source_run_is_rejected(tmp_path,frames,overrides):
    source=source_run(tmp_path,[success("belief",frames,{})],**overrides)
    with pytest.raises(ValueError):
        ResumableSatoriStageProvider(Live(),resume_from=source)


def test_missing_raw_response_and_malformed_trace_are_rejected(tmp_path,frames):
    trace=success("belief",frames,{})
    trace.pop("raw_json")
    source=source_run(tmp_path,[trace])
    with pytest.raises(ValueError):
        ResumableSatoriStageProvider(Live(),resume_from=source)
    (source/"api-calls.jsonl").write_text("{bad\n",encoding="utf-8")
    with pytest.raises(ValueError):
        ResumableSatoriStageProvider(Live(),resume_from=source)


def test_source_input_hashes_must_match_current_run(tmp_path,frames):
    source=source_run(tmp_path,[success("belief",frames,{})])
    provider=ResumableSatoriStageProvider(Live(),resume_from=source)
    provider.validate_run_inputs({"session_sha256":"a"*64,"task_catalog_sha256":"b"*64})
    with pytest.raises(ValueError,match="input"):
        provider.validate_run_inputs({"session_sha256":"c"*64,"task_catalog_sha256":"b"*64})


def test_fresh_wrapper_marks_real_calls_and_resume_chain_keeps_origin(tmp_path,frames):
    fresh_trace=tmp_path/"fresh"/"api-calls.jsonl"
    fresh=ResumableSatoriStageProvider(Live(),trace_path=fresh_trace)
    fresh.complete_json("belief",frames,{})
    assert fresh.resume_metadata is None and fresh.real_api_calls==1 and fresh.cached_calls==0
    source=source_run(tmp_path,fresh.traces)
    cached=ResumableSatoriStageProvider(Live(),resume_from=source)
    cached.complete_json("belief",frames,{})
    assert cached.traces[0]["response_origin"]=="real_api"
    assert cached.traces[0]["call_source"]=="cached"
