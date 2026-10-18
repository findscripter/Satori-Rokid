"""Local HTTP contract tests for the staged Satori transport; no real API calls."""

import base64
from contextlib import contextmanager
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import pytest

from satori_repro.stage_provider import SatoriStageProvider


TEST_KEY_ENV = "SATORI_STAGE_PROVIDER_TEST_KEY"


@contextmanager
def local_api(response=None, *, status=200, delay=0, redirect=None):
    received = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append({"path": self.path, "headers": dict(self.headers),
                             "body": json.loads(self.rfile.read(int(self.headers["Content-Length"])))})
            if delay:
                time.sleep(delay)
            self.send_response(status)
            if redirect:
                self.send_header("Location", redirect)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body = response if response is not None else {
                "model": "returned-model",
                "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
                "choices": [{"finish_reason": "stop", "message": {
                    "content": '{"scene":"桌面","evidence_frame_ids":["f1"]}',
                    "reasoning_content": "PRIVATE_REASONING_DO_NOT_LOG"}}]}
            try:
                self.wfile.write(body if isinstance(body, bytes) else json.dumps(body).encode())
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def frames(tmp_path):
    image = tmp_path / "fixture.png"
    ok, encoded = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    image.write_bytes(encoded.tobytes())
    return [{"frame_id": "f1", "timestamp_ms": 100, "image_path": str(image)}]


def test_protocol_chinese_instruction_and_request_time_key(monkeypatch, frames):
    monkeypatch.delenv(TEST_KEY_ENV, raising=False)
    with local_api() as (url, received):
        provider = SatoriStageProvider(model="explicit-model", base_url=url, api_key_env=TEST_KEY_ENV)
        payload = {"history": [{"action": "past"}]}
        original = copy.deepcopy((frames, payload))
        monkeypatch.setenv(TEST_KEY_ENV, "test-stage-secret-1")
        assert provider.complete_json("belief", frames, payload)["scene"] == "桌面"
        monkeypatch.setenv(TEST_KEY_ENV, "test-stage-secret-2")
        provider.complete_json("belief", frames, payload)
    assert (frames, payload) == original
    assert [r["headers"]["Authorization"] for r in received] == [
        "Bearer test-stage-secret-1", "Bearer test-stage-secret-2"]
    body = received[0]["body"]
    assert received[0]["path"] == "/v1/chat/completions"
    assert body["model"] == "explicit-model"
    assert body["stream"] is False and body["response_format"] == {"type": "json_object"}
    system = body["messages"][0]["content"]
    assert "JSON contract" in system and "中文" in system
    parts = body["messages"][1]["content"]
    context = json.loads(parts[0]["text"])
    assert context["payload"] == payload
    assert context["frames"] == [{"frame_id": "f1", "timestamp_ms": 100}]
    assert "f1" in parts[1]["text"] and "100" in parts[1]["text"]
    data_url = parts[2]["image_url"]["url"]
    assert base64.b64decode(data_url.split(",")[1]) == Path(frames[0]["image_path"]).read_bytes()


def test_trace_records_auditable_facts_without_reasoning_or_key(monkeypatch, frames, tmp_path):
    monkeypatch.setenv(TEST_KEY_ENV, "trace-secret")
    trace_path = tmp_path / "logs" / "stage.jsonl"
    with local_api() as (url, _):
        provider = SatoriStageProvider(model="explicit-model", base_url=url,
                                      api_key_env=TEST_KEY_ENV, trace_path=trace_path)
        result = provider.complete_json("belief", frames, {"history": []})
    trace = provider.traces[0]
    assert trace["stage"] == "belief" and trace["status"] == "success"
    assert trace["requested_model"] == "explicit-model" and trace["returned_model"] == "returned-model"
    assert trace["usage"]["total_tokens"] == 16
    assert trace["latency_ms"] >= 0 and trace["image_ids"] == ["f1"]
    assert trace["image_sha256"] == {
        "f1": hashlib.sha256(Path(frames[0]["image_path"]).read_bytes()).hexdigest()}
    assert len(trace["system_prompt_sha256"]) == 64 and len(trace["payload_sha256"]) == 64
    assert trace["raw_json"] == result
    result["scene"] = "mutated"
    assert trace["raw_json"]["scene"] == "桌面"
    text = trace_path.read_text(encoding="utf-8")
    assert json.loads(text) == trace
    assert "PRIVATE_REASONING" not in text and "reasoning_content" not in text
    assert "trace-secret" not in text and "Authorization" not in text and "base64" not in text


def test_trace_redacts_key_echo_and_omits_reasoning_keys(monkeypatch, frames, tmp_path):
    key = "actual-key-echo-test"
    monkeypatch.setenv(TEST_KEY_ENV, key)
    response = {"model": "returned", "usage": {"debug": key},
                "choices": [{"finish_reason": "stop", "message": {
                    "content": json.dumps({"scene": key, "nested": {"reasoning_content": "hidden"}}),
                    "reasoning_content": "NEVER_LOG"}}]}
    with local_api(response) as (url, _):
        provider = SatoriStageProvider(model="x", base_url=url, api_key_env=TEST_KEY_ENV)
        result = provider.complete_json("belief", frames, {})
    assert result["scene"] == key  # Transport leaves schema/content to MethodSession.
    logged = json.dumps(provider.traces)
    assert key not in logged and "NEVER_LOG" not in logged and "reasoning_content" not in logged


@pytest.mark.parametrize("response", [
    b"not json", {}, {"choices": []},
    {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": "[]"}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": '{"x":NaN}'}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": "{}", "refusal": "no"}}]},
])
def test_rejects_invalid_incomplete_or_refused_json_and_does_not_retry(monkeypatch, frames, response):
    monkeypatch.setenv(TEST_KEY_ENV, "response-test-key")
    with local_api(response) as (url, received):
        provider = SatoriStageProvider(model="x", base_url=url, api_key_env=TEST_KEY_ENV)
        with pytest.raises(RuntimeError, match="complete JSON"):
            provider.complete_json("belief", frames, {})
    assert len(received) == 1
    assert len(provider.traces) == 1 and provider.traces[0]["status"] == "error"


@pytest.mark.parametrize("status", [401, 429, 500])
def test_http_errors_record_status_without_body_or_key(monkeypatch, frames, status):
    key = "http-secret-key"
    monkeypatch.setenv(TEST_KEY_ENV, key)
    with local_api((key + " secret-body").encode(), status=status) as (url, received):
        provider = SatoriStageProvider(model="x", base_url=url, api_key_env=TEST_KEY_ENV)
        with pytest.raises(RuntimeError) as caught:
            provider.complete_json("belief", frames, {})
    assert str(caught.value) == f"Satori stage API returned HTTP {status}"
    assert caught.value.__suppress_context__ is True
    trace = provider.traces[0]
    assert trace["http_status"] == status and trace["status"] == "error"
    assert len(received) == 1
    assert key not in json.dumps(trace) and "secret-body" not in json.dumps(trace)


def test_redirect_not_followed_and_timeout_is_sanitized(monkeypatch, frames):
    monkeypatch.setenv(TEST_KEY_ENV, "redirect-secret")
    with local_api() as (destination, received):
        with local_api(status=307, redirect=destination) as (url, _):
            provider = SatoriStageProvider(model="x", base_url=url, api_key_env=TEST_KEY_ENV)
            with pytest.raises(RuntimeError, match="HTTP 307"):
                provider.complete_json("belief", frames, {})
        assert not received
    with local_api(delay=0.1) as (url, _):
        provider = SatoriStageProvider(model="x", base_url=url, api_key_env=TEST_KEY_ENV, timeout=0.01)
        with pytest.raises(RuntimeError, match="connection"):
            provider.complete_json("belief", frames, {})
        assert provider.traces[0]["status"] == "error"


@pytest.mark.parametrize("kwargs", [
    {"model": ""}, {"model": "x", "timeout": False}, {"model": "x", "timeout": float("nan")},
    {"model": "x", "timeout": 0}, {"model": "x", "api_key_env": "bad name"},
    {"model": "x", "base_url": "http://example.org/v1"},
])
def test_invalid_configuration_rejected(kwargs):
    kwargs.setdefault("base_url", "http://127.0.0.1:1/v1")
    with pytest.raises(ValueError):
        SatoriStageProvider(**kwargs)


def test_missing_key_and_invalid_inputs_do_not_send_requests(monkeypatch, frames):
    monkeypatch.delenv(TEST_KEY_ENV, raising=False)
    with local_api() as (url, received):
        provider = SatoriStageProvider(model="x", base_url=url, api_key_env=TEST_KEY_ENV)
        with pytest.raises(ValueError):
            provider.complete_json("belief", frames, {})
        monkeypatch.setenv(TEST_KEY_ENV, "input-test-key")
        for stage, images, payload in [
            ("not-a-stage", frames, {}), ("belief", [], {}),
            ("belief", frames, {"bad": float("nan")}),
            ("belief", [{**frames[0], "image_path": "missing.png"}], {}),
        ]:
            with pytest.raises(ValueError):
                provider.complete_json(stage, images, payload)
    assert not received and provider.traces == []


def test_json_objects_pass_through_for_method_schema_validation(monkeypatch, frames):
    monkeypatch.setenv(TEST_KEY_ENV, "schema-test-key")
    response = {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}
    with local_api(response) as (url, _):
        provider = SatoriStageProvider(model="x", base_url=url, api_key_env=TEST_KEY_ENV)
        assert provider.complete_json("guidance", frames, {"target_step": {}}) == {}


def test_context_question_dispatches_independent_grounded_prompt(monkeypatch, frames):
    monkeypatch.setenv(TEST_KEY_ENV, "context-only-dummy-key")
    answer = {"answer": "画面中是室内台面，手里拿着布。",
              "environment": {"place_type": "室内台面附近", "visible_objects": ["布"]},
              "activity": {"action": "拿着布", "phase": "unknown"},
              "uncertainties": ["无法确定具体房间"], "evidence_frame_ids": ["f1"],
              "sensor_evidence_ids": []}
    response = {"model": "local-test-model", "choices": [{"finish_reason": "stop",
                 "message": {"content": json.dumps(answer, ensure_ascii=False)}}]}
    payload = {"question": "我在哪里，正在做什么？", "history": []}
    with local_api(response) as (url, received):
        provider = SatoriStageProvider(model="explicit-model", base_url=url, api_key_env=TEST_KEY_ENV)
        result = provider.complete_json("context_question", frames, payload)
    assert result == answer and len(received) == 1
    system = received[0]["body"]["messages"][0]["content"]
    assert "place_type" in system and "sensor_evidence_ids" in system
    sent = json.loads(received[0]["body"]["messages"][1]["content"][0]["text"])
    assert sent["payload"] == payload and "task_catalog" not in sent["payload"]
    assert provider.traces[0]["stage"] == "context_question"
