"""No-network subprocess mocks for the isolated Claude CLI paper provider."""

import base64
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import pytest

from deploy.paper_claude_cli import provider as cli
import satori_paper_benchmark as bench


def stream(*, text=None, stop="end_turn", model=cli.MODEL, tools=None, extra=None, turns=1, models=None):
    text = text or '{"answer":"cup","evidence_frame_ids":["v000"]}'
    events = [
        {"type": "system", "subtype": "init", "model": cli.MODEL, "tools": tools or [], "mcp_servers": [],
         "cwd": "C:/Users/private/.claude", "session_id": "private-session", "apiKeySource": "private-auth"},
        {"type": "assistant", "message": {"id": "msg_1", "role": "assistant", "model": model,
         "content": [{"type": "thinking", "thinking": "Private reasoning is omitted", "signature": "opaque"},
                     {"type": "text", "text": text}], "stop_reason": stop, "usage": {"input_tokens": 10, "output_tokens": 20}},
         "parent_tool_use_id": None},
        {"type": "result", "subtype": "success", "is_error": False, "num_turns": turns,
         "result": text, "stop_reason": stop, "usage": {"input_tokens": 10, "output_tokens": 20},
         "modelUsage": models or {cli.MODEL: {"inputTokens": 10, "outputTokens": 20}}, "session_id": "private-session"},
    ]
    if extra:
        events.insert(2, extra)
    return events


@pytest.fixture
def sample(tmp_path):
    ok, encoded = cv2.imencode(".jpg", np.full((8, 8, 3), 127, dtype=np.uint8))
    assert ok
    image = tmp_path / "frame.jpg"
    image.write_bytes(encoded.tobytes())
    frames = [{"frame_id": f"v{i:03d}", "timestamp_ms": i * 1000, "image_path": str(image),
               "sha256": bench._sha(image.read_bytes())} for i in range(4)]
    return {"sample_id": "s000", "task_kind": "video_qa", "question": "What is visible?",
            "cutoff_ms": 3000, "candidate_group": None, "frames": frames}


@pytest.fixture
def taxonomy():
    return {"schema_version": 1, "goals": [], "actions": []}


def mock_provider(tmp_path, monkeypatch, events=None, *, stderr=b"", returncode=0, timeout=False):
    binary = tmp_path / "fake-claude.exe"
    binary.write_bytes(b"mock-only executable never run")
    calls = []
    raw = b"not-json\n" if events == "invalid" else ("\n".join(json.dumps(e) for e in (events or stream())) + "\n").encode()
    class Process:
        pid = 123
        def __init__(self, command, **kwargs):
            self.returncode = returncode
            self.call = {"command": command, "kwargs": kwargs, "files": {p.name: p.read_bytes() for p in Path(kwargs["cwd"]).iterdir()}}
            calls.append(self.call)
            self.count = 0
        def communicate(self, input=None, timeout=None):
            self.count += 1
            self.call.update(stdin=input, timeout=timeout)
            if timeout is not None and timeout_flag:
                raise subprocess.TimeoutExpired("mock", timeout)
            return raw, stderr
        def kill(self):
            self.call["killed"] = True
    timeout_flag = timeout
    monkeypatch.setattr(cli.subprocess, "Popen", Process)
    monkeypatch.setattr(cli, "_terminate_tree", lambda proc: proc.kill())
    return cli.ClaudeCliProvider(executable=binary, executable_sha256=bench._sha(binary.read_bytes())), calls


def test_exact_input_bytes_prompt_controls_and_one_natural_response(tmp_path, monkeypatch, sample, taxonomy):
    monkeypatch.setenv("CLAUDE_CODE_RETRY_WATCHDOG", "1")
    p, calls = mock_provider(tmp_path, monkeypatch)
    result = p.predict(sample, taxonomy)
    assert result == {"answer": "cup", "evidence_frame_ids": ["v000"]}
    assert len(calls) == 1 and len(p.traces) == 1
    call, trace = calls[0], p.traces[0]
    command, env = call["command"], call["kwargs"]["env"]
    assert call["kwargs"]["shell"] is False
    assert "--bare" not in command and "--json-schema" not in command
    for flag in ("-p", "--safe-mode", "--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands"):
        assert flag in command
    for flag, value in (("--model", cli.MODEL), ("--effort", "high"), ("--tools", ""), ("--disallowedTools", "*"),
                        ("--max-turns", "1"), ("--input-format", "stream-json"), ("--output-format", "stream-json")):
        assert command[command.index(flag) + 1] == value
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "4096"
    assert "CLAUDE_CODE_RETRY_WATCHDOG" not in env
    assert all(env[k] == v for k, v in cli.ENV_CONTROLS.items())
    assert set(call["files"]) == {"system-prompt.txt", "settings.json", "mcp.json"}
    assert call["files"]["system-prompt.txt"] == bench.PROMPT.encode()
    assert json.loads(call["files"]["settings.json"]) == {"fallbackModel": [], "disableAllHooks": True}
    assert json.loads(call["files"]["mcp.json"]) == {"mcpServers": {}}
    assert call["stdin"].endswith(b"\n") and len(call["stdin"].splitlines()) == 1
    body = json.loads(call["stdin"])
    content = body["message"]["content"]
    assert content[0]["text"] == bench._json_text(bench.visible_payload(sample, taxonomy))
    images = [part for part in content if part["type"] == "image"]
    assert len(images) == 4 and len(content) == 9
    for part, frame in zip(images, sample["frames"]):
        assert part["source"]["media_type"] == "image/jpeg"
        assert base64.b64decode(part["source"]["data"]) == Path(frame["image_path"]).read_bytes()
    assert trace["generation_parameters"] == {"max_tokens": 4096, "effort": "high"}
    assert trace["finish_reason"] == "stop" and trace["native_stop_reason"] == "end_turn"
    assert trace["cli_assistant_message_count"] == 1 and trace["cli_num_turns"] == 1
    assert trace["cli_stdin_sha256"] == bench._sha(call["stdin"])
    encoded = json.dumps(trace)
    for private in ("private-session", "private-auth", "C:/Users/private", "Private reasoning", "opaque"):
        assert private not in encoded


@pytest.mark.parametrize("events,error", [
    ("invalid", "cli_non_json_stream"),
    (stream(text="not JSON"), "invalid_complete_prediction"),
    (stream(stop="max_tokens"), "cli_incomplete_generation"),
    (stream(tools=["Read"]), "cli_tools_enabled"),
    (stream(extra={"type": "system", "subtype": "api_retry", "attempt": 1}), "cli_retry_or_fallback_detected"),
    (stream(model="claude-sonnet-5-0"), "cli_model_mismatch"),
    (stream(turns=2), "cli_unsuccessful_or_multiple_turns"),
    (stream(models={cli.MODEL: {}, "claude-haiku-4-5": {}}), "cli_multiple_or_unknown_usage_models"),
])
def test_invalid_responses_are_once_only_failures(tmp_path, monkeypatch, sample, taxonomy, events, error):
    p, calls = mock_provider(tmp_path, monkeypatch, events)
    with pytest.raises(RuntimeError):
        p.predict(sample, taxonomy)
    assert len(calls) == 1 and len(p.traces) == 1
    assert p.traces[0]["status"] == "error" and p.traces[0]["error_code"] == error


def test_tool_content_and_multiple_assistant_events_fail(tmp_path, monkeypatch, sample, taxonomy):
    events = stream()
    events[1]["message"]["content"] = [{"type": "tool_use", "name": "Read", "input": {}}]
    p, _ = mock_provider(tmp_path, monkeypatch, events)
    with pytest.raises(RuntimeError, match="tool"):
        p.predict(sample, taxonomy)
    events = stream()
    events.insert(2, deepcopy(events[1]))
    p, _ = mock_provider(tmp_path, monkeypatch, events)
    with pytest.raises(RuntimeError, match="multiple_or_missing"):
        p.predict(sample, taxonomy)
    assert p.traces[0]["cli_assistant_message_count"] == 2


def test_timeout_kills_own_process_once_and_retains_failure(tmp_path, monkeypatch, sample, taxonomy):
    p, calls = mock_provider(tmp_path, monkeypatch, timeout=True)
    with pytest.raises(RuntimeError, match="timeout"):
        p.predict(sample, taxonomy)
    assert len(calls) == 1 and calls[0]["killed"]
    assert p.traces[0]["error_code"] == "cli_timeout"


def test_stderr_is_not_persisted_and_retry_warning_rejected(tmp_path, monkeypatch, sample, taxonomy):
    p, _ = mock_provider(tmp_path, monkeypatch, stderr=b"retrying; private@example.com; C:/Users/private/key")
    with pytest.raises(RuntimeError, match="retry"):
        p.predict(sample, taxonomy)
    encoded = json.dumps(p.traces)
    assert "private@example.com" not in encoded and "C:/Users/private" not in encoded
    assert p.traces[0]["cli_stderr_bytes"] > 0


def test_each_sample_has_new_isolated_process(tmp_path, monkeypatch, sample, taxonomy):
    p, calls = mock_provider(tmp_path, monkeypatch)
    p.predict(sample, taxonomy)
    p.predict(sample, taxonomy)
    assert len(calls) == 2
    assert calls[0]["kwargs"]["cwd"] != calls[1]["kwargs"]["cwd"]


def test_corrupted_image_fails_before_process(tmp_path, monkeypatch, sample, taxonomy):
    p, calls = mock_provider(tmp_path, monkeypatch)
    sample["frames"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen hash"):
        p.predict(sample, taxonomy)
    assert not calls


def test_usage_over_budget_is_failure_with_raw_evidence(tmp_path, monkeypatch, sample, taxonomy):
    events = stream()
    events[-1]["usage"]["output_tokens"] = 4097
    p, calls = mock_provider(tmp_path, monkeypatch, events)
    with pytest.raises(RuntimeError, match="budget_exceeded"):
        p.predict(sample, taxonomy)
    assert len(calls) == 1
    assert p.traces[0]["usage"]["output_tokens"] == 4097
    assert p.traces[0]["raw_json_text"] == events[-1]["result"]


def test_real_cli_partial_assistant_metadata_uses_final_result(tmp_path, monkeypatch, sample, taxonomy):
    events = stream()
    events[1]["message"]["stop_reason"] = None
    events[1]["message"]["usage"]["output_tokens"] = 8
    events[-1]["usage"]["output_tokens"] = 62
    events[-1]["modelUsage"][cli.MODEL]["outputTokens"] = 62
    events.insert(2, {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed"}})
    p, calls = mock_provider(tmp_path, monkeypatch, events)
    assert p.predict(sample, taxonomy)["answer"] == "cup"
    trace = p.traces[0]
    assert len(calls) == 1 and trace["status"] == "success"
    assert trace["cli_assistant_stop_reason"] is None
    assert trace["native_stop_reason"] == "end_turn" and trace["finish_reason"] == "stop"
    assert trace["usage"]["output_tokens"] == 62
    assert trace["cli_events_redacted"][1]["message"]["usage"]["output_tokens"] == 8


@pytest.mark.parametrize("assistant_stop,final_stop", [("max_tokens", "end_turn"),
                                                       ("end_turn", "max_tokens"),
                                                       (None, "max_tokens"), ("end_turn", None)])
def test_contradictory_or_incomplete_final_stop_is_rejected(tmp_path, monkeypatch, sample, taxonomy,
                                                         assistant_stop, final_stop):
    events = stream()
    events[1]["message"]["stop_reason"] = assistant_stop
    events[-1]["stop_reason"] = final_stop
    p, calls = mock_provider(tmp_path, monkeypatch, events)
    with pytest.raises(RuntimeError, match="cli_incomplete_generation"):
        p.predict(sample, taxonomy)
    assert len(calls) == 1 and p.traces[0]["status"] == "error"
    assert p.traces[0]["native_stop_reason"] == final_stop
