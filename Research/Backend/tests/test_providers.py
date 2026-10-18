import base64
import copy
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from satori_repro.providers import OpenAICompatibleProvider, ReplayProvider


OBSERVATION = {"scene": "desk", "objects": [], "action": "place cup", "current_step_id": "s1",
               "completed_step_ids": [], "confidence": 0.8, "uncertain": False,
               "evidence_frame_ids": ["f2"], "needs_help": False, "help_text": ""}


@contextmanager
def local_api(*, status=200, response=None, delay=0, redirect=None):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append({"path": self.path, "headers": dict(self.headers), "body": json.loads(body)})
            if delay:
                time.sleep(delay)
            self.send_response(status)
            if redirect:
                self.send_header("Location", redirect)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            payload = response if response is not None else {
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(OBSERVATION)}}]
            }
            try:
                self.wfile.write(payload if isinstance(payload, bytes) else json.dumps(payload).encode())
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def frames(tmp_path):
    image = tmp_path / "not-really-a-jpeg.jpg"
    ok, encoded = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    image.write_bytes(encoded.tobytes())
    return [{"frame_id": "f1", "timestamp_ms": 0, "image_path": str(image)},
            {"frame_id": "f2", "timestamp_ms": 1000, "image_path": str(image)}]


def test_replay_selects_current_frame_and_returns_independent_copy(tmp_path):
    path = tmp_path / "responses.json"
    path.write_text(json.dumps({"synthetic": True, "observations": {"f1": {"scene": "old"}, "f2": OBSERVATION}}))
    provider = ReplayProvider(path)
    assert provider.name == "replay" and provider.synthetic is True
    result = provider.infer([{"frame_id": "f1"}, {"frame_id": "f2"}], task={}, history=[], state=None)
    assert result == OBSERVATION
    result["objects"].append({"name": "changed"})
    assert provider.infer([{"frame_id": "f2"}], task={}, history=[], state=None) == OBSERVATION
    with pytest.raises(ValueError):
        provider.infer([{"frame_id": "missing"}], task={}, history=[], state=None)
    with pytest.raises(ValueError):
        provider.infer([], task={}, history=[], state=None)


@pytest.mark.parametrize("payload", [[], {"synthetic": "true", "observations": {}},
    {"observations": {}}, {"synthetic": False, "observations": [], "extra": 1},
    {"synthetic": False, "observations": {"f1": []}}])
def test_replay_rejects_invalid_fixture(tmp_path, payload):
    path = tmp_path / "responses.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        ReplayProvider(path)


def test_replay_rejects_missing_and_malformed_files(tmp_path):
    with pytest.raises(ValueError):
        ReplayProvider(tmp_path / "missing.json")
    path = tmp_path / "responses.json"
    path.write_text("not json")
    with pytest.raises(ValueError):
        ReplayProvider(path)


def test_vision_request_protocol_and_call_time_key(monkeypatch, frames):
    monkeypatch.delenv("SATORI_TEST_KEY", raising=False)
    with local_api() as (url, requests):
        provider = OpenAICompatibleProvider(model="explicit-test-model", base_url=url, api_key_env="SATORI_TEST_KEY")
        monkeypatch.setenv("SATORI_TEST_KEY", "test-key-1")
        task, history, state = {"title": "cup task"}, [{"scene": "previous"}], {"task_confirmed": True}
        original = copy.deepcopy((frames, task, history, state))
        assert provider.infer(frames, task=task, history=history, state=state) == OBSERVATION
        monkeypatch.setenv("SATORI_TEST_KEY", "test-key-2")
        provider.infer(frames, task=task, history=history, state=state)
    assert provider.name == "openai-compatible" and provider.synthetic is False
    assert (frames, task, history, state) == original
    assert [r["headers"]["Authorization"] for r in requests] == ["Bearer test-key-1", "Bearer test-key-2"]
    assert requests[0]["path"] == "/v1/chat/completions"
    body = requests[0]["body"]
    assert body["model"] == "explicit-test-model"
    assert body["response_format"] == {"type": "json_object"}
    assert body["stream"] is False
    assert "JSON" in body["messages"][0]["content"]
    content = body["messages"][1]["content"]
    context = json.loads(content[0]["text"])
    assert context["task"] == task and context["history"] == history and context["state"] == state
    assert str(frames[0]["image_path"]) not in json.dumps(context)
    images = [part["image_url"]["url"] for part in content if part["type"] == "image_url"]
    assert len(images) == 2
    assert all(value.startswith("data:image/png;base64,") for value in images)
    assert base64.b64decode(images[0].split(",", 1)[1]).startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("kwargs", [{"model": ""}, {"model": " "}, {"model": "x", "timeout": 0},
    {"model": "x", "timeout": float("nan")}, {"model": "x", "timeout": True},
    {"model": "x", "base_url": "http://example.org/v1"}, {"model": "x", "base_url": "file:///tmp"},
    {"model": "x", "base_url": "https://name:secret@example.org"},
    {"model": "x", "base_url": "https://example.org/v1?key=secret"},
    {"model": "x", "api_key_env": "bad name"}])
def test_provider_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        OpenAICompatibleProvider(**kwargs)


@pytest.mark.parametrize("url", ["https://example.org/v1", "http://localhost:8080/v1", "http://[::1]:8080/v1"])
def test_provider_accepts_https_or_loopback_without_loading_key(url, monkeypatch):
    monkeypatch.delenv("SATORI_TEST_KEY", raising=False)
    assert OpenAICompatibleProvider(model="x", base_url=url, api_key_env="SATORI_TEST_KEY").model == "x"


def test_missing_key_fails_before_network(monkeypatch, frames):
    monkeypatch.delenv("SATORI_TEST_KEY", raising=False)
    with local_api() as (url, requests):
        provider = OpenAICompatibleProvider(model="x", base_url=url, api_key_env="SATORI_TEST_KEY")
        with pytest.raises(ValueError):
            provider.infer(frames, task={}, history=[], state=None)
    assert not requests


@pytest.mark.parametrize("data", [b"text masquerading as image", b"\x89PNG\r\n\x1a\nnot-valid"])
def test_invalid_image_bytes_fail_before_network(tmp_path, monkeypatch, data):
    monkeypatch.setenv("SATORI_TEST_KEY", "secret-test-key")
    path = tmp_path / "image.png"
    path.write_bytes(data)
    with local_api() as (url, requests):
        provider = OpenAICompatibleProvider(model="x", base_url=url, api_key_env="SATORI_TEST_KEY")
        with pytest.raises(ValueError):
            provider.infer([{"frame_id": "f1", "timestamp_ms": 0, "image_path": str(path)}], task={}, history=[], state=None)
    assert not requests


@pytest.mark.parametrize("response", [b"secret-response-body", {}, {"choices": []},
    {"choices": [{"finish_reason": "length", "message": {"content": "secret-response-body"}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": "not-json secret-response-body"}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": "[]"}}]}])
def test_bad_response_is_sanitized(monkeypatch, frames, response):
    monkeypatch.setenv("SATORI_TEST_KEY", "secret-test-key")
    with local_api(response=response) as (url, _):
        provider = OpenAICompatibleProvider(model="x", base_url=url, api_key_env="SATORI_TEST_KEY")
        with pytest.raises(RuntimeError) as error:
            provider.infer(frames, task={}, history=[], state=None)
    assert "secret-response-body" not in str(error.value)
    assert "secret-test-key" not in str(error.value)
    assert error.value.__suppress_context__ is True


@pytest.mark.parametrize("status", [401, 429, 500])
def test_http_errors_are_sanitized(monkeypatch, frames, status):
    monkeypatch.setenv("SATORI_TEST_KEY", "secret-test-key")
    with local_api(status=status, response=b"secret-response-body") as (url, _):
        provider = OpenAICompatibleProvider(model="x", base_url=url, api_key_env="SATORI_TEST_KEY")
        with pytest.raises(RuntimeError) as error:
            provider.infer(frames, task={}, history=[], state=None)
    assert str(status) in str(error.value)
    assert "secret" not in str(error.value)


def test_timeout_and_redirect_do_not_leak_key(monkeypatch, frames):
    monkeypatch.setenv("SATORI_TEST_KEY", "secret-test-key")
    with local_api(delay=0.15) as (url, _):
        provider = OpenAICompatibleProvider(model="x", base_url=url, api_key_env="SATORI_TEST_KEY", timeout=0.02)
        with pytest.raises(RuntimeError) as error:
            provider.infer(frames, task={}, history=[], state=None)
        assert "secret" not in str(error.value)
    with local_api() as (destination, received):
        with local_api(status=307, redirect=destination) as (url, _):
            provider = OpenAICompatibleProvider(model="x", base_url=url, api_key_env="SATORI_TEST_KEY")
            with pytest.raises(RuntimeError):
                provider.infer(frames, task={}, history=[], state=None)
        assert not received
