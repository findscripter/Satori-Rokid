"""Leakage, protocol and metric checks; no external model endpoint is used."""

from contextlib import contextmanager
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import sys
import threading
import types

import cv2
import numpy as np
import pytest

SPEC=importlib.util.spec_from_file_location("gt_benchmark",Path(__file__).resolve().parents[3]/"Tools/gt_benchmark.py")
benchmark=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=benchmark
SPEC.loader.exec_module(benchmark)


def taxonomy():
    return {"schema_version":1,"scenes":[{"id":f"s{i:03d}","label":f"Scene {i}"} for i in range(4)],
            "actions":[{"id":f"c{i:03d}","label":f"Action {i}"} for i in range(157)]}


def answer(scene="s000",actions=None):
    return {"scene_id":scene,"action_ids":["c000"] if actions is None else actions,
            "evidence_frame_ids":["v007"],"answer":"图片中的环境和可见动作。"}


@pytest.fixture
def inputs(tmp_path):
    image=tmp_path/"GT_bedroom_c000_DO_NOT_SEND.png"
    ok,encoded=cv2.imencode(".png",np.zeros((6,6,3),dtype=np.uint8))
    assert ok
    image.write_bytes(encoded.tobytes())
    sha=hashlib.sha256(image.read_bytes()).hexdigest()
    manifest={"schema_version":1,"benchmark_id":"opaque-test","samples":[
        {"sample_id":f"e{i:03d}","frames":[{"frame_id":f"v{j:03d}","timestamp_ms":j*100,
            "image_path":image.name,"sha256":sha} for j in range(8)]} for i in range(8)]}
    manifest_path=tmp_path/"manifest.json"
    manifest_path.write_text(json.dumps(manifest),encoding="utf-8")
    taxonomy_path=tmp_path/"taxonomy.json"
    taxonomy_path.write_text(json.dumps(taxonomy()),encoding="utf-8")
    return manifest_path,taxonomy_path,manifest


class Provider:
    model="Mage-VL"
    base_url="http://127.0.0.1:9999/v1"
    def __init__(self,responses=None):
        self.traces=[]
        self.calls=[]
        self.responses=responses or [answer()]*8
    def classify(self,frames,candidates):
        index=len(self.calls)
        self.calls.append(copy.deepcopy((frames,candidates)))
        value=self.responses[index]
        if isinstance(value,Exception):
            self.traces.append({"status":"error","latency_ms":12.0,"http_status":429})
            raise value
        self.traces.append({"status":"success","latency_ms":10.0,"returned_model":self.model,
                            "raw_json":copy.deepcopy(value)})
        return copy.deepcopy(value)


def write_gt(tmp_path,manifest,*,empty_last=False):
    labels=[]
    for i,sample in enumerate(manifest["samples"]):
        labels.append({"sample_id":sample["sample_id"],"scene_id":f"s{i//2:03d}",
                       "action_ids":[] if empty_last and i==7 else [f"c{i//2:03d}"],
                       "source_video_id":f"official-{i}-EGO"})
    path=tmp_path/"gt.json"
    path.write_text(json.dumps({"schema_version":1,"benchmark_id":manifest["benchmark_id"],
                               "labels":labels,"provenance":{"kind":"test-fixture-only"}}),encoding="utf-8")
    return path


def test_infer_has_no_gt_argument_and_records_every_failure(inputs,tmp_path):
    manifest_path,taxonomy_path,_=inputs
    provider=Provider([answer(),RuntimeError("PRIVATE_ERROR"),answer(),answer(),answer(),answer(),answer(),answer()])
    run=tmp_path/"run"
    summary=benchmark.infer(manifest_path,taxonomy_path,run,provider)
    assert summary["planned_samples"]==8 and summary["failed_samples"]==1
    rows=[json.loads(line) for line in (run/"predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows)==8 and rows[1]["status"]=="failed" and rows[1]["response"] is None
    assert len(provider.calls)==8
    assert all(len(frames)==8 and candidates==taxonomy() for frames,candidates in provider.calls)
    assert "PRIVATE_ERROR" not in (run/"predictions.jsonl").read_text(encoding="utf-8")


def test_machine_metrics_include_failed_and_abstained_samples(inputs,tmp_path):
    manifest_path,taxonomy_path,manifest=inputs
    responses=[answer("s000",["c000"]),answer("s000",["c000"]),
               answer("s001",["c001"]),answer("s000",["c000"]),
               answer("s002",["c002"]),RuntimeError("failed"),
               answer("s003",["c003"]),{"scene_id":None,"action_ids":None,
                   "evidence_frame_ids":["v007"],"answer":"无法确定。"}]
    run=tmp_path/"run"
    benchmark.infer(manifest_path,taxonomy_path,run,Provider(responses))
    scores=benchmark.score(run,write_gt(tmp_path,manifest))
    assert scores["sample_count"]==8
    assert scores["scene"]["top1"]==pytest.approx(5/8)
    assert scores["scene"]["macro_f1_gt_present"]==pytest.approx(0.7)
    assert sum(sum(row) for row in scores["scene"]["confusion"]["counts"])==8
    actions=scores["actions"]
    assert actions["micro_precision"]==pytest.approx(5/6)
    assert actions["micro_recall"]==pytest.approx(5/8)
    assert actions["micro_f1"]==pytest.approx(10/14)
    assert actions["macro_f1_gt_present"]==pytest.approx(0.7)
    assert actions["macro_f1_all_157"]==pytest.approx(2.8/157)
    assert actions["exact_match"]==pytest.approx(5/8)
    assert actions["mean_jaccard"]==pytest.approx(5/8)
    assert actions["sample_mean_f1"]==pytest.approx(5/8)
    assert scores["coverage"]["failed_samples"]==1 and actions["abstained_samples"]==1
    assert scores["latency"]["attempted_samples"]==8
    assert scores["mAP"] is None


def test_missing_prediction_stays_in_fixed_denominator(inputs,tmp_path):
    manifest_path,taxonomy_path,manifest=inputs
    run=tmp_path/"run"
    benchmark.infer(manifest_path,taxonomy_path,run,Provider())
    path=run/"predictions.jsonl"
    rows=path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(rows[:-1])+"\n",encoding="utf-8")
    scores=benchmark.score(run,write_gt(tmp_path,manifest))
    assert scores["sample_count"]==8 and scores["coverage"]["not_run_samples"]==1
    assert sum(sum(row) for row in scores["scene"]["confusion"]["counts"])==8


def test_failed_empty_gt_is_not_a_correct_empty_set_prediction(inputs,tmp_path):
    manifest_path,taxonomy_path,manifest=inputs
    run=tmp_path/"run"
    responses=[answer(actions=[])]*7+[RuntimeError("failed")]
    benchmark.infer(manifest_path,taxonomy_path,run,Provider(responses))
    scores=benchmark.score(run,write_gt(tmp_path,manifest,empty_last=True))
    assert scores["actions"]["exact_match"]==0
    assert scores["actions"]["mean_jaccard"]==0


@pytest.mark.parametrize("patch",[
    {"scene_id":"s999"},{"action_ids":["c999"]},{"action_ids":["c000","c000"]},
    {"evidence_frame_ids":[]},{"evidence_frame_ids":["future"]},{"answer":""},{"extra":"no"},
])
def test_prediction_contract_is_strict(patch):
    value=answer()
    value.update(patch)
    with pytest.raises(ValueError):
        benchmark.validate_prediction(value,taxonomy(),[f"v{i:03d}" for i in range(8)])


def test_manifest_rejects_gt_fields_and_nonanonymous_ids(inputs):
    _,_,manifest=inputs
    for change in ("gt","name","count"):
        data=copy.deepcopy(manifest)
        if change=="gt": data["samples"][0]["scene_id"]="s000"
        elif change=="name": data["samples"][0]["sample_id"]="Bedroom_video"
        else: data["samples"][0]["frames"].pop()
        with pytest.raises(ValueError):
            benchmark.validate_manifest(data)


def test_taxonomy_requires_all_157_action_ids():
    value=taxonomy()
    value["actions"].pop()
    with pytest.raises(ValueError):
        benchmark.validate_taxonomy(value)


@pytest.mark.parametrize("change",["missing","duplicate","unknown_class","duplicate_source"])
def test_gt_alignment_errors_are_rejected_instead_of_dropping_samples(inputs,tmp_path,change):
    manifest_path,taxonomy_path,manifest=inputs
    run=tmp_path/"run"
    benchmark.infer(manifest_path,taxonomy_path,run,Provider())
    gt=write_gt(tmp_path,manifest)
    data=json.loads(gt.read_text(encoding="utf-8"))
    if change=="missing": data["labels"].pop()
    elif change=="duplicate": data["labels"].append(data["labels"][0])
    elif change=="unknown_class": data["labels"][0]["action_ids"]=["c999"]
    elif change=="duplicate_source": data["labels"][1]["source_video_id"]=data["labels"][0]["source_video_id"]
    gt.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        benchmark.score(run,gt)


@contextmanager
def local_api(response,status=200):
    captured=[]
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            captured.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(status);self.send_header("Content-Type","application/json");self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        def log_message(self,*args): pass
    server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    try: yield f"http://127.0.0.1:{server.server_port}/v1",captured
    finally: server.shutdown();server.server_close();worker.join(timeout=2)


def test_http_payload_has_full_candidates_no_source_paths_or_gt(inputs,monkeypatch):
    manifest_path,_,manifest=inputs
    frames=copy.deepcopy(manifest["samples"][0]["frames"])
    for frame in frames: frame["image_path"]=str(manifest_path.parent/frame["image_path"])
    envelope={"model":"Mage-VL","usage":{"total_tokens":33},
              "choices":[{"finish_reason":"stop","message":{"content":json.dumps(answer())}}]}
    monkeypatch.setenv("GT_BENCHMARK_TEST_KEY","test-only-gt-key")
    with local_api(envelope) as (url,captured):
        provider=benchmark.BenchmarkProvider(model="Mage-VL",base_url=url,api_key_env="GT_BENCHMARK_TEST_KEY")
        assert provider.classify(frames,taxonomy())==answer()
    request=captured[0]
    wire=json.dumps(request,ensure_ascii=False)
    assert "GT_bedroom" not in wire and str(manifest_path.parent) not in wire and "sample_id" not in wire
    assert request["response_format"]=={"type":"json_object"} and request["stream"] is False
    assert set(request)=={"model","stream","response_format","messages"}
    assert provider.traces[0]["generation_parameters"]=={}
    assert provider.traces[0]["expected_response_model"]=="Mage-VL"
    assert provider.traces[0]["timeout_seconds"]==180
    context=json.loads(request["messages"][1]["content"][0]["text"])
    assert context["scene_candidates"]==taxonomy()["scenes"]
    assert len(context["action_candidates"])==157
    assert provider.traces[0]["raw_json"]==answer()
    assert len(provider.traces[0]["image_sha256"])==8
    assert "test-only-gt-key" not in json.dumps(provider.traces)


@pytest.mark.parametrize("status,envelope",[
    (429,{"error":{"message":"PRIVATE_BODY"}}),
    (200,{"model":"Mage-VL","choices":[{"finish_reason":"length","message":{"content":"{}"}}]}),
    (200,{"model":"Other","choices":[{"finish_reason":"stop","message":{"content":json.dumps(answer())}}]}),
    (200,{"model":"Mage-VL","choices":[{"finish_reason":"stop","message":{"content":chr(96)*3+"json\n{}\n"+chr(96)*3}}]}),
])
def test_http_failure_is_not_repaired_or_retried(inputs,monkeypatch,status,envelope):
    manifest_path,_,manifest=inputs
    frames=copy.deepcopy(manifest["samples"][0]["frames"])
    for frame in frames: frame["image_path"]=str(manifest_path.parent/frame["image_path"])
    monkeypatch.setenv("GT_BENCHMARK_TEST_KEY","test-only-gt-key")
    with local_api(envelope,status) as (url,captured):
        provider=benchmark.BenchmarkProvider(model="Mage-VL",base_url=url,api_key_env="GT_BENCHMARK_TEST_KEY")
        with pytest.raises(RuntimeError):
            provider.classify(frames,taxonomy())
    assert len(captured)==1 and provider.traces[0]["status"]=="error"
    assert "PRIVATE_BODY" not in json.dumps(provider.traces)


def test_interrupted_attempt_is_recorded_before_remaining_samples_are_not_run(inputs,tmp_path):
    manifest_path,taxonomy_path,manifest=inputs
    class InterruptedProvider(Provider):
        def classify(self,frames,candidates):
            if len(self.calls)==2:
                self.calls.append(copy.deepcopy((frames,candidates)))
                self.traces.append({"status":"error","latency_ms":9.0,"error_code":"interrupted"})
                raise KeyboardInterrupt
            return super().classify(frames,candidates)
    run=tmp_path/"run"
    with pytest.raises(KeyboardInterrupt):
        benchmark.infer(manifest_path,taxonomy_path,run,InterruptedProvider())
    rows=[json.loads(line) for line in (run/"predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows)==3 and rows[-1]["status"]=="failed" and rows[-1]["error_code"]=="KeyboardInterrupt"
    metadata=json.loads((run/"run.json").read_text(encoding="utf-8"))
    assert metadata["status"]=="aborted" and metadata["api_calls"]==3
    result=benchmark.score(run,write_gt(tmp_path,manifest))
    assert result["coverage"]["failed_samples"]==1 and result["coverage"]["not_run_samples"]==5
    assert result["latency"]["attempted_samples"]==3


def test_malformed_prediction_status_is_rejected_cleanly(inputs,tmp_path):
    manifest_path,taxonomy_path,manifest=inputs
    run=tmp_path/"run"
    benchmark.infer(manifest_path,taxonomy_path,run,Provider())
    path=run/"predictions.jsonl"
    rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["status"]=[]
    path.write_text("\n".join(json.dumps(row) for row in rows)+"\n",encoding="utf-8")
    with pytest.raises(ValueError):
        benchmark.score(run,write_gt(tmp_path,manifest))


def test_valid_empty_prediction_and_empty_gt_count_as_an_exact_match(inputs,tmp_path):
    manifest_path,taxonomy_path,manifest=inputs
    run=tmp_path/"run"
    benchmark.infer(manifest_path,taxonomy_path,run,Provider([answer(actions=[])]*8))
    result=benchmark.score(run,write_gt(tmp_path,manifest,empty_last=True))
    assert result["actions"]["exact_match"]==pytest.approx(1/8)
    assert result["actions"]["mean_jaccard"]==pytest.approx(1/8)
    assert result["actions"]["sample_mean_f1"]==pytest.approx(1/8)


def test_32_frame_http_uses_every_image_without_changing_default_eight_prompt(inputs,monkeypatch):
    manifest_path,_,manifest=inputs
    original=manifest["samples"][0]["frames"][0]
    frames=[{**original,"frame_id":f"v{i:03d}","timestamp_ms":i*100,
             "image_path":str(manifest_path.parent/original["image_path"])} for i in range(32)]
    value={**answer(),"evidence_frame_ids":["v031"]}
    envelope={"model":"Mage-VL","choices":[{"finish_reason":"stop","message":{"content":json.dumps(value)}}]}
    monkeypatch.setenv("GT_BENCHMARK_TEST_KEY","test-only-gt-key")
    with local_api(envelope) as (url,captured):
        provider=benchmark.BenchmarkProvider(model="Mage-VL",base_url=url,api_key_env="GT_BENCHMARK_TEST_KEY",frame_count=32)
        assert provider.classify(frames,taxonomy())==value
    content=captured[0]["messages"][1]["content"]
    assert len([part for part in content if part["type"]=="image_url"])==32
    assert len(provider.traces[0]["image_sha256"])==32
    assert provider.traces[0]["image_ids"][-1]=="v031"
    assert provider.prompt==benchmark.PROMPT.replace("eight","thirty-two")
    default=benchmark.BenchmarkProvider(model="Mage-VL",base_url="http://127.0.0.1:9999/v1")
    assert hashlib.sha256(default.prompt.encode()).hexdigest()=="20fd957dedc0bd534f2d8014d9f219de345f786dfff1b6dd51cf7ec8d2399b9a"
    with pytest.raises(ValueError):
        benchmark.validate_manifest({"schema_version":1,"benchmark_id":"wrong-count","samples":[{"sample_id":"e000","frames":frames}]})


def test_32_frame_run_scores_v031_evidence_and_keeps_failures_in_denominator(inputs,tmp_path):
    manifest_path,taxonomy_path,manifest=inputs
    for sample in manifest["samples"]:
        original=sample["frames"][0]
        sample["frames"]=[{**original,"frame_id":f"v{i:03d}","timestamp_ms":i*100} for i in range(32)]
    manifest_path.write_text(json.dumps(manifest),encoding="utf-8")
    value={**answer(),"evidence_frame_ids":["v031"]}
    provider=Provider([value,RuntimeError("controlled fixture failure"),value,value,value,value,value,value])
    provider.frame_count=32
    run=tmp_path/"32-frame-run"
    benchmark.infer(manifest_path,taxonomy_path,run,provider,frame_count=32)
    result=benchmark.score(run,write_gt(tmp_path,manifest))
    assert result["sample_count"]==8 and result["coverage"]["failed_samples"]==1
    assert result["scene"]["top1"]==pytest.approx(1/8)
    assert result["actions"]["micro_f1"]==pytest.approx(2/15)


def test_http_502_keeps_safe_reason_and_usage_without_raw_error_or_retry(inputs,monkeypatch):
    manifest_path,_,manifest=inputs
    frames=copy.deepcopy(manifest["samples"][0]["frames"])
    for frame in frames:frame["image_path"]=str(manifest_path.parent/frame["image_path"])
    monkeypatch.setenv("GT_BENCHMARK_TEST_KEY","test-only-gt-key")
    envelope={"error":{"code":"incomplete_json","finish_reason":"length","message":"PRIVATE_ERROR_BODY test-only-gt-key"},
              "usage":{"prompt_tokens":100,"completion_tokens":1024,"total_tokens":1124,"secret":"PRIVATE_USAGE"}}
    with local_api(envelope,502) as (url,captured):
        provider=benchmark.BenchmarkProvider(model="Mage-VL",base_url=url,api_key_env="GT_BENCHMARK_TEST_KEY")
        with pytest.raises(RuntimeError):provider.classify(frames,taxonomy())
    trace=provider.traces[0]
    assert len(captured)==1 and trace["http_status"]==502
    assert trace["server_error_code"]=="incomplete_json" and trace["server_finish_reason"]=="length"
    assert trace["server_usage"]=={"prompt_tokens":100,"completion_tokens":1024,"total_tokens":1124}
    assert "PRIVATE" not in json.dumps(trace) and "test-only-gt-key" not in json.dumps(trace)


@pytest.mark.parametrize("field",["max_tokens","max_completion_tokens"])
def test_explicit_generation_budget_and_model_identity_are_frozen_in_wire_trace_and_run(inputs,tmp_path,monkeypatch,field):
    manifest_path,taxonomy_path,_=inputs
    monkeypatch.setenv("GT_BENCHMARK_TEST_KEY","test-only-gt-key")
    envelope={"model":"fixture-model-2026-10-16","usage":{"completion_tokens":22},
              "choices":[{"finish_reason":"stop","message":{"content":json.dumps(answer())}}]}
    with local_api(envelope) as (url,captured):
        provider=benchmark.BenchmarkProvider(model="fixture-model-alias",base_url=url,
            api_key_env="GT_BENCHMARK_TEST_KEY",max_output_tokens=4096,output_token_field=field,
            temperature=0,expected_response_model="fixture-model-2026-10-16",timeout=25)
        result=benchmark.infer(manifest_path,taxonomy_path,tmp_path/field,provider)
    assert len(captured)==8 and result["successful_samples"]==8
    parameters={field:4096,"temperature":0.0}
    assert result["generation_parameters"]==parameters and result["timeout_seconds"]==25
    assert result["expected_response_model"]=="fixture-model-2026-10-16"
    saved=json.loads((tmp_path/field/"run.json").read_text(encoding="utf-8"))
    assert saved==result
    for body,trace in zip(captured,provider.traces):
        assert body[field]==4096 and body["temperature"]==0
        assert set(body)=={"model","stream","response_format","messages",field,"temperature"}
        assert body["model"]=="fixture-model-alias"
        assert trace["generation_parameters"]==parameters and trace["timeout_seconds"]==25
        assert trace["expected_response_model"]=="fixture-model-2026-10-16"
        assert trace["returned_model"]=="fixture-model-2026-10-16" and trace["response_model_matches"] is True
        assert trace["finish_reason"]=="stop" and trace["response_refused"] is False


@pytest.mark.parametrize("options",[
    {"max_output_tokens":True},{"max_output_tokens":0},{"max_output_tokens":-1},{"max_output_tokens":4.5},
    {"output_token_field":"arbitrary"},{"temperature":True},{"temperature":float("nan")},
    {"temperature":float("inf")},{"temperature":-0.1},{"temperature":2.1},
    {"expected_response_model":""},{"expected_response_model":["model"]},
])
def test_invalid_generation_configuration_is_rejected_before_request(options):
    with pytest.raises(ValueError):
        benchmark.BenchmarkProvider(model="Mage-VL",base_url="http://127.0.0.1:9999/v1",**options)


@pytest.mark.parametrize("returned,reason,refusal,issue",[
    ("fixture-alias","stop",None,"unexpected_model"),
    ("fixture-version-2","stop",None,"unexpected_model"),
    ("fixture-version-1","length",None,"incomplete_finish_reason"),
    ("fixture-version-1","stop","PRIVATE_REFUSAL_TEXT","refusal"),
])
def test_expected_model_is_one_exact_id_and_failure_diagnostics_do_not_enable_retry(inputs,monkeypatch,returned,reason,refusal,issue):
    manifest_path,_,manifest=inputs
    frames=copy.deepcopy(manifest["samples"][0]["frames"])
    for frame in frames:frame["image_path"]=str(manifest_path.parent/frame["image_path"])
    monkeypatch.setenv("GT_BENCHMARK_TEST_KEY","test-only-gt-key")
    envelope={"model":returned,"choices":[{"finish_reason":reason,
              "message":{"content":json.dumps(answer()),"refusal":refusal}}]}
    with local_api(envelope) as (url,captured):
        provider=benchmark.BenchmarkProvider(model="fixture-alias",base_url=url,
            api_key_env="GT_BENCHMARK_TEST_KEY",expected_response_model="fixture-version-1")
        with pytest.raises(RuntimeError):provider.classify(frames,taxonomy())
    trace=provider.traces[0]
    assert len(captured)==1 and trace["error_code"]=="invalid_complete_prediction"
    assert trace["finish_reason"]==reason and trace["response_refused"]==bool(refusal)
    assert trace["response_model_matches"]==(returned=="fixture-version-1")
    assert trace["response_issues"]==[issue]
    assert "PRIVATE_REFUSAL_TEXT" not in json.dumps(trace)


def test_benchmark_cli_passes_explicit_generation_options_without_calling_endpoint(tmp_path,monkeypatch):
    captured={}
    def fake_infer(manifest,taxonomy,output,provider,**options):
        captured["provider"]=provider
        return {"failed_samples":0}
    monkeypatch.setattr(benchmark,"infer",fake_infer)
    result=benchmark.main(["infer","--manifest",str(tmp_path/"manifest.json"),
        "--taxonomy",str(tmp_path/"taxonomy.json"),"--output",str(tmp_path/"run"),
        "--model","fixture-alias","--expected-response-model","fixture-version-1",
        "--max-output-tokens","4096","--output-token-field","max_completion_tokens",
        "--temperature","1","--timeout","60"])
    provider=captured["provider"]
    assert result==0 and provider.generation_parameters=={"max_completion_tokens":4096,"temperature":1.0}
    assert provider.expected_response_model=="fixture-version-1" and provider.timeout==60


def test_mage_cli_passes_generation_options_and_cleans_mock_credential(tmp_path,monkeypatch):
    captured={}
    bridge=types.ModuleType("unity_mage_request")
    bridge.PROJECT_ROOT=tmp_path
    bridge.ensure_service=lambda root: captured.update(service_root=root)
    bridge.fetch_service_token=lambda: "test-only-mage-key"
    monkeypatch.setitem(sys.modules,"unity_mage_request",bridge)
    module_path=Path(__file__).resolve().parents[3]/"Tools/run_mage_gt.py"
    spec=importlib.util.spec_from_file_location("run_mage_gt_configuration_test",module_path)
    runner=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    def fake_infer(manifest,taxonomy,output,provider,**options):
        captured["provider"]=provider
        return {"status":"completed","planned_samples":32,"successful_samples":32,"failed_samples":0,"api_calls":32}
    monkeypatch.setattr(runner,"infer",fake_infer)
    monkeypatch.setattr(sys,"argv",[str(module_path),"--manifest",str(tmp_path/"manifest.json"),
        "--taxonomy",str(tmp_path/"taxonomy.json"),"--output",str(tmp_path/"run"),
        "--max-output-tokens","4096","--output-token-field","max_completion_tokens",
        "--temperature","0","--timeout","75","--expected-response-model","Mage-VL"])
    assert runner.main()==0
    provider=captured["provider"]
    assert provider.generation_parameters=={"max_completion_tokens":4096,"temperature":0.0}
    assert provider.expected_response_model=="Mage-VL" and provider.timeout==75
    assert provider.api_key_env not in runner.os.environ
