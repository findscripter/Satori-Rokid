"""CPU protocol tests: strict input/output, no secret trace, no model execution."""

import base64
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest
from PIL import Image


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("native_eval_server", HERE / "server.py")
server = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = server
SPEC.loader.exec_module(server)


def fixture():
    raw = io.BytesIO()
    Image.new("RGB", (8, 6), "green").save(raw, format="JPEG")
    data = raw.getvalue()
    uri = "data:image/jpeg;base64," + base64.b64encode(data).decode()
    context = {"scene_candidates": [{"id": f"s{i:03d}", "label": "scene"} for i in range(16)],
               "action_candidates": [{"id": f"c{i:03d}", "label": "action"} for i in range(157)],
               "frames": [{"frame_id": f"v{i:03d}", "timestamp_ms": 1000*(i+1)} for i in range(8)]}
    parts = [{"type": "text", "text": server.compact(context)}]
    for i in range(8):
        parts += [{"type": "text", "text": f"Frame ID: v{i:03d}; timestamp_ms: {1000*(i+1)}"},
                  {"type": "image_url", "image_url": {"url": uri}}]
    body = {"model": "mock-native", "stream": False, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": "Return classification JSON."},
                         {"role": "user", "content": parts}], "max_tokens": 4096, "temperature": 0}
    registry = {server.input_fingerprint([server.sha(data)]*8, [1000*(i+1) for i in range(8)]):
                {"fps": 30.0, "total_num_frames": 300, "duration": 10.0, "frames_indices": list(range(30, 270, 30)),
                 "width": 8, "height": 6, "video_backend": "frozen-jpeg"}}
    return body, registry


def result(**patch):
    value = {"text": json.dumps({"scene_id": "s000", "action_ids": ["c000"], "evidence_frame_ids": ["v000"], "answer": "可见动作"}),
             "finish_reason": "stop", "prompt_tokens": 10, "completion_tokens": 20,
             "preprocessing": {"unique_frames": 8}}
    value.update(patch)
    return value


def test_parse_exact_eight_jpegs_and_anonymous_facts():
    body, registry = fixture()
    spec, trace = server.parse_request(body, "mock-native", registry)
    assert len(spec["frames"]) == 8
    assert spec["frame_metadata"] == next(iter(registry.values()))
    assert trace["image_sha256"] == [trace["image_sha256"][0]]*8
    assert "source_video_id" not in server.compact(trace)
    assert spec["max_new_tokens"] == 4096


@pytest.mark.parametrize("patch", [{"model": "other"}, {"stream": True}, {"max_tokens": True},
    {"max_tokens": 8193}, {"temperature": 1}, {"temperature": float("nan")}, {"tools": []},
    {"response_format": {"type": "json_schema"}}])
def test_reject_invalid_request_parameters(patch):
    body, registry = fixture()
    body.update(patch)
    with pytest.raises(server.APIError):
        server.parse_request(body, "mock-native", registry)


def test_reject_frame_text_mismatch_and_unregistered_inputs():
    body, registry = fixture()
    body["messages"][1]["content"][1]["text"] = "Frame ID: v000; timestamp_ms: 0"
    with pytest.raises(server.APIError):
        server.parse_request(body, "mock-native", registry)
    with pytest.raises(server.APIError, match="unregistered"):
        server.parse_request(fixture()[0], "mock-native", {})


@pytest.mark.parametrize("uri", ["https://example.org/image.jpg", "file:///tmp/a.jpg", "data:image/jpeg;base64,YWJj"])
def test_no_url_fetch_or_corrupt_image(uri):
    with pytest.raises(server.APIError):
        server.decode_image(uri)


@pytest.mark.parametrize("text", ['```json\n{}\n```', '{"scene_id":null,"scene_id":"s000"}', 'not json'])
def test_output_never_repaired(text):
    spec, _ = server.parse_request(*[fixture()[0], "mock-native", fixture()[1]])
    with pytest.raises(server.APIError) as caught:
        server.build_completion(result(text=text), spec, "mock-native", "fixed", "secret-token-value")
    assert caught.value.status == 502


def test_truncated_valid_json_still_fails_and_true_usage_survives():
    spec, _ = server.parse_request(fixture()[0], "mock-native", fixture()[1])
    with pytest.raises(server.APIError) as caught:
        server.build_completion(result(finish_reason="length"), spec, "mock-native", "fixed", "secret-token-value")
    assert caught.value.usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    assert caught.value.finish_reason == "length"
    accepted = server.build_completion(result(prompt_tokens=None), spec, "mock-native", "fixed", "secret-token-value")
    assert accepted["usage"] == {"completion_tokens": 20}


def test_live_http_one_call_no_retry_and_no_credential_trace(tmp_path):
    from fastapi.testclient import TestClient
    body, registry = fixture()
    token = "not-a-real-token-123456789"
    (tmp_path / "token").write_text(token)
    (tmp_path / "metadata.json").write_text(json.dumps({"inputs": registry}))
    calls = []
    class Backend:
        model_id = "mock-native"
        def generate(self, **kwargs):
            calls.append(kwargs)
            return result()
    app = server.create_app(model_id="mock-native", revision="fixed", token_file=tmp_path / "token",
        metadata_file=tmp_path / "metadata.json", audit_file=tmp_path / "audit.jsonl", loader=Backend)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 401
        headers = {"Authorization": "Bearer " + token}
        assert client.get("/health", headers=headers).json()["registered_inputs"] == 1
        response = client.post("/v1/chat/completions", headers=headers, json=body)
        assert response.status_code == 200
        assert response.json()["model"] == "mock-native"
        assert len(calls) == 1
    raw = (tmp_path / "audit.jsonl").read_text()
    assert token not in raw and "base64" not in raw
    assert json.loads(raw)["status"] == "completed"


def test_live_http_invalid_output_kept_as_failure(tmp_path):
    from fastapi.testclient import TestClient
    body, registry = fixture()
    token = "not-a-real-token-123456789"
    (tmp_path / "token").write_text(token)
    (tmp_path / "metadata.json").write_text(json.dumps({"inputs": registry}))
    class Backend:
        model_id = "mock-native"
        def generate(self, **kwargs):
            return result(text='```json\n{}\n```')
    app = server.create_app(model_id="mock-native", revision="fixed", token_file=tmp_path / "token",
        metadata_file=tmp_path / "metadata.json", audit_file=tmp_path / "audit.jsonl", loader=Backend)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", headers={"Authorization": "Bearer " + token}, json=body)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_complete_prediction"
    trace = json.loads((tmp_path / "audit.jsonl").read_text())
    assert trace["status"] == "failed" and trace["result"]["text"].startswith("```")


def test_existing_benchmark_provider_wire_roundtrip(tmp_path, monkeypatch):
    """Use the unchanged real benchmark encoder and decoder through the new app."""
    from fastapi.testclient import TestClient
    from urllib import request
    sys.path.insert(0, str(HERE.parents[1]))
    from gt_benchmark import BenchmarkProvider
    body, registry = fixture()
    token = "not-a-real-token-123456789"
    (tmp_path / "token").write_text(token)
    (tmp_path / "metadata.json").write_text(json.dumps({"inputs": registry}))
    calls = []
    class Backend:
        model_id = "mock-native"
        def generate(self, **kwargs):
            calls.append(kwargs)
            return result()
    app = server.create_app(model_id="mock-native", revision="fixed", token_file=tmp_path / "token",
        metadata_file=tmp_path / "metadata.json", audit_file=tmp_path / "audit.jsonl", loader=Backend)
    frames = []
    image_raw = base64.b64decode(body["messages"][1]["content"][2]["image_url"]["url"].split(",", 1)[1])
    for i in range(8):
        path = tmp_path / f"{i}.jpg"
        path.write_bytes(image_raw)
        frames.append({"frame_id": f"v{i:03d}", "timestamp_ms": 1000*(i+1),
                       "sha256": server.sha(image_raw), "image_path": str(path)})
    context = json.loads(body["messages"][1]["content"][0]["text"])
    taxonomy = {"schema_version": 1, "scenes": context["scene_candidates"], "actions": context["action_candidates"]}
    monkeypatch.setenv("NATIVE_TEST_ONLY_KEY", token)
    with TestClient(app) as client:
        class Opener:
            def open(self, req, timeout):
                assert timeout == 180
                response = client.post("/v1/chat/completions", content=req.data, headers=dict(req.header_items()))
                assert response.status_code == 200
                return io.BytesIO(response.content)
        monkeypatch.setattr(request, "build_opener", lambda *_: Opener())
        provider = BenchmarkProvider(model="mock-native", base_url="http://127.0.0.1:18868/v1",
                                     api_key_env="NATIVE_TEST_ONLY_KEY", max_output_tokens=4096, temperature=0)
        prediction = provider.classify(frames, taxonomy)
    assert prediction["action_ids"] == ["c000"]
    assert len(calls) == len(provider.traces) == 1
    assert provider.traces[0]["status"] == "success"
    assert calls[0]["frame_metadata"] == next(iter(registry.values()))
