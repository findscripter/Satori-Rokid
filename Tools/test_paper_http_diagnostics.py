"""Offline fake opener checks; no real credentials or HTTP calls."""

from copy import deepcopy
from email.message import Message
import io
import json
from pathlib import Path
from urllib import error

import cv2
import numpy as np
import pytest

import paper_http_diagnostics as diagnostic
import satori_paper_benchmark as bench


@pytest.fixture
def inputs(tmp_path):
    ok, encoded = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    image = tmp_path / "frame.jpg"
    image.write_bytes(encoded.tobytes())
    sample = {"sample_id": "s000", "task_kind": "video_qa", "candidate_group": None,
              "question": "What is visible?", "cutoff_ms": 3000,
              "frames": [{"frame_id": f"v{i:03d}", "timestamp_ms": i * 1000, "image_path": str(image),
                          "sha256": bench._sha(image.read_bytes())} for i in range(4)]}
    return sample, {"schema_version": 1, "goals": [], "actions": []}


def provider(monkeypatch):
    monkeypatch.setenv("OFFLINE_DIAGNOSTIC_KEY", "fake-credential-known-only-to-test")
    return diagnostic.DiagnosticPaperProvider(model="test-model", base_url="https://example.invalid/v1",
        api_key_env="OFFLINE_DIAGNOSTIC_KEY", timeout=180, max_output_tokens=4096, temperature=0)


def test_http_error_once_bounded_and_redacted(monkeypatch, inputs):
    p = provider(monkeypatch)
    raw = json.dumps({"error": {"type": "rate_limit_error", "code": "request_quota",
        "message": "fake-credential-known-only-to-test Bearer another-secret-value user@example.com sk-abcdef123456"}}).encode()
    headers = Message()
    headers["Retry-After"] = "60"
    headers["x-ratelimit-remaining-requests"] = "0"
    headers["Set-Cookie"] = "sensitive-cookie"
    calls, reads = [], []
    class Body(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)
    class Opener:
        def open(self, request, **kwargs):
            calls.append((request, kwargs))
            raise error.HTTPError(request.full_url, 429, "rate limited", headers, Body(raw))
    def original(*args, **kwargs):
        assert len(args) == 1 and isinstance(args[0], bench._NoRedirect)
        return Opener()
    monkeypatch.setattr(bench.request, "build_opener", original)
    with pytest.raises(RuntimeError, match="HTTP error"):
        p.predict(*inputs)
    assert bench.request.build_opener is original
    assert len(calls) == len(p.traces) == 1 and reads == [8192]
    trace = p.traces[0]
    detail = trace["http_error_diagnostics"]
    assert trace["error_code"] == "http_error" and trace["http_status"] == 429
    assert detail["retry_after_seconds"] == 60
    assert detail["captured_body_sha256"] == bench._sha(raw)
    assert detail["headers"] == {"retry-after": "60", "x-ratelimit-remaining-requests": "0"}
    saved = json.dumps(p.traces)
    for private in ("fake-credential-known-only-to-test", "another-secret-value", "user@example.com", "sk-abcdef123456", "sensitive-cookie"):
        assert private not in saved


def test_success_payload_and_prediction_identical_to_base(monkeypatch, inputs):
    captured = []
    answer = {"answer": "cup", "evidence_frame_ids": ["v000"]}
    raw = json.dumps({"model": "test-model", "usage": {"total_tokens": 25},
                      "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer)}}]}).encode()
    class Opener:
        def open(self, request, **kwargs):
            captured.append((request.full_url, request.data, request.headers, kwargs))
            return io.BytesIO(raw)
    original = lambda *args, **kwargs: Opener()
    monkeypatch.setattr(bench.request, "build_opener", original)
    p = provider(monkeypatch)
    base = bench.PaperBenchmarkProvider(model="test-model", base_url="https://example.invalid/v1",
        api_key_env="OFFLINE_DIAGNOSTIC_KEY", timeout=180, max_output_tokens=4096, temperature=0)
    assert p.predict(*inputs) == base.predict(*inputs) == answer
    assert len(captured) == 2 and captured[0] == captured[1]
    first, second = deepcopy(p.traces[0]), deepcopy(base.traces[0])
    first.pop("latency_ms")
    second.pop("latency_ms")
    assert first == second
    assert bench.request.build_opener is original


def test_unexpected_open_exception_restores_global_opener(monkeypatch, inputs):
    p = provider(monkeypatch)
    class Opener:
        def open(self, *args, **kwargs):
            raise RuntimeError("fake opener exception")
    original = lambda *args, **kwargs: Opener()
    monkeypatch.setattr(bench.request, "build_opener", original)
    with pytest.raises(RuntimeError, match="fake opener"):
        p.predict(*inputs)
    assert bench.request.build_opener is original


def test_large_nonjson_error_keeps_only_bounded_hash(monkeypatch, inputs):
    p = provider(monkeypatch)
    body = b"x" * 10000
    class Opener:
        def open(self, request, **kwargs):
            raise error.HTTPError(request.full_url, 503, "unavailable", Message(), io.BytesIO(body))
    original = lambda *args, **kwargs: Opener()
    monkeypatch.setattr(bench.request, "build_opener", original)
    with pytest.raises(RuntimeError):
        p.predict(*inputs)
    detail = p.traces[0]["http_error_diagnostics"]
    assert detail["captured_body_bytes"] == 8192 and detail["body_may_be_truncated"]
    assert detail["captured_body_sha256"] == bench._sha(body[:8192])
    assert "xxxxxxxx" not in json.dumps(detail)


@pytest.mark.parametrize("value,expected", [("60", 60), (" 0 ", 0), ("1.5", None),
                                           ("Wed, 21 Oct 2026 07:28:00 GMT", None), ("-1", None)])
def test_numeric_retry_after_does_not_guess_dates(value, expected):
    assert diagnostic.numeric_retry_after(value) == expected
