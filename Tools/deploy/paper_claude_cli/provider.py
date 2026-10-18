"""One-process Claude Code/ReClaude provider for frozen paper-data probes.

Uses the existing CLI authentication; never reads or writes credential files.
CLI transport is recorded explicitly, including its unavailable temperature
control. Each sample gets a fresh no-GT directory and an independent process.
"""

from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

TOOLS = Path(__file__).resolve().parents[2]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import satori_paper_benchmark as bench


DEFAULT_EXE = Path.home() / "AppData/Local/Programs/reclaude/bin/reclaude.exe"
DEFAULT_EXE_SHA256 = "d866d40ef7e477844977793b5a044b006dc340684a11d867234bd1d29377a7a2"
MODEL = "claude-opus-5-5"
ENV_CONTROLS = {
    "CLAUDE_CODE_MAX_RETRIES": "0",
    "CLAUDE_CODE_NONSTREAMING_TIMEOUT_RETRIES": "0",
    "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK": "1",
    "CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
}


class CliContractError(RuntimeError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _safe_text(text):
    if not isinstance(text, str):
        raise CliContractError("cli_non_text_response")
    # Never persist credential-shaped output, account email, or local user paths.
    if re.search(r"\b(?:rck_|sk-|Bearer\s+)[A-Za-z0-9_.-]{8,}|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|[A-Za-z]:[\\/]Users[\\/]|/home/[^ /]+/", text):
        raise CliContractError("cli_sensitive_output")
    return text


def _numeric_usage(value):
    if not isinstance(value, dict):
        return None
    return {key: val for key, val in value.items()
            if isinstance(key, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", key)
            and type(val) in (int, float) and math.isfinite(val) and val >= 0}


def _event_summary(event):
    """Whitelist evidence fields; drop cwd, session/account IDs and auth paths."""
    kind = event.get("type")
    result = {"type": kind}
    if kind == "system":
        result.update(subtype=event.get("subtype"), model=event.get("model"),
                      tool_count=len(event.get("tools", [])), mcp_server_count=len(event.get("mcp_servers", [])))
    elif kind == "assistant":
        msg = event.get("message", {})
        content = []
        for part in msg.get("content", []):
            if part.get("type") == "text":
                content.append({"type": "text", "text": _safe_text(part.get("text"))})
            else:
                # Do not export private reasoning or opaque signatures.
                content.append({"type": part.get("type"), "omitted_sha256": bench._sha(bench._json_text(part).encode())})
        result["message"] = {"model": msg.get("model"), "stop_reason": msg.get("stop_reason"),
                             "usage": _numeric_usage(msg.get("usage")), "content": content}
    elif kind == "result":
        for key in ("subtype", "is_error", "num_turns", "stop_reason", "duration_ms", "duration_api_ms", "total_cost_usd"):
            if key in event:
                result[key] = event[key]
        if isinstance(event.get("result"), str):
            result["result"] = _safe_text(event["result"])
        result["usage"] = _numeric_usage(event.get("usage"))
        result["modelUsage"] = {key: _numeric_usage(value) for key, value in event.get("modelUsage", {}).items()
                                if isinstance(key, str) and re.fullmatch(r"[A-Za-z0-9_.:-]+", key)}
    else:
        result["subtype"] = event.get("subtype")
    return result


def parse_cli_output(stdout, stderr, *, expected_model, evidence=None):
    """Accept exactly one natural assistant completion and one success result."""
    try:
        lines = stdout.decode("utf-8").splitlines()
        events = [bench._strict_response_json(line) for line in lines if line.strip()]
    except (UnicodeError, ValueError):
        raise CliContractError("cli_non_json_stream") from None
    if not events or any(not isinstance(event, dict) for event in events):
        raise CliContractError("cli_invalid_stream")
    summaries = [_event_summary(event) for event in events]
    if evidence is not None:
        evidence["cli_events_redacted"] = summaries
        evidence["cli_assistant_message_count"] = sum(e.get("type") == "assistant" for e in events)
        evidence["cli_result_count"] = sum(e.get("type") == "result" for e in events)
        messages = [e.get("message", {}) for e in events if e.get("type") == "assistant"]
        if len(messages) == 1:
            evidence["returned_model"] = messages[0].get("model")
            evidence["cli_assistant_stop_reason"] = messages[0].get("stop_reason")
        results = [e for e in events if e.get("type") == "result"]
        if len(results) == 1:
            evidence["native_stop_reason"] = results[0].get("stop_reason")
            evidence["finish_reason"] = "stop" if results[0].get("stop_reason") == "end_turn" else results[0].get("stop_reason")
            evidence["usage"] = _numeric_usage(results[0].get("usage"))
    for event in events:
        signature = str(event.get("type", "")) + ":" + str(event.get("subtype", ""))
        if re.search(r"retry|fallback|compact", signature, re.I):
            raise CliContractError("cli_retry_or_fallback_detected")
        if event.get("type") not in ("system", "assistant", "result", "rate_limit_event"):
            raise CliContractError("cli_unexpected_event")
        if event.get("type") == "system" and event.get("subtype") != "init":
            raise CliContractError("cli_unexpected_system_event")
    if re.search(rb"\bretrying\b|\bretry attempt\b|\bfallback\b", stderr, re.I):
        raise CliContractError("cli_retry_or_fallback_detected")
    initial = [e for e in events if e.get("type") == "system" and e.get("subtype") == "init"]
    assistant = [e for e in events if e.get("type") == "assistant"]
    final = [e for e in events if e.get("type") == "result"]
    if len(initial) != 1 or len(assistant) != 1 or len(final) != 1:
        raise CliContractError("cli_multiple_or_missing_model_calls")
    if events.index(initial[0]) > events.index(assistant[0]) or events[-1] is not final[0]:
        raise CliContractError("cli_invalid_event_order")
    init, message, result = initial[0], assistant[0].get("message", {}), final[0]
    if init.get("model") != expected_model or message.get("model") != expected_model:
        raise CliContractError("cli_model_mismatch")
    if init.get("tools") or init.get("mcp_servers"):
        raise CliContractError("cli_tools_enabled")
    if assistant[0].get("parent_tool_use_id") is not None:
        raise CliContractError("cli_tool_activity")
    content = message.get("content")
    if not isinstance(content, list) or any(part.get("type") not in ("thinking", "redacted_thinking", "text") for part in content):
        raise CliContractError("cli_tool_or_unexpected_content")
    texts = [part["text"] for part in content if part.get("type") == "text"]
    if len(texts) != 1 or not texts[0].strip():
        raise CliContractError("cli_multiple_or_missing_final_text")
    # The CLI assistant event may precede final streaming metadata and carry
    # stop_reason=null and partial usage. The final result is authoritative;
    # a non-null assistant reason must still agree with it.
    native_stop = result.get("stop_reason")
    if native_stop != "end_turn" or message.get("stop_reason") not in (None, native_stop):
        raise CliContractError("cli_incomplete_generation")
    if result.get("subtype") != "success" or result.get("is_error") is not False or result.get("num_turns") != 1:
        raise CliContractError("cli_unsuccessful_or_multiple_turns")
    model_usage = result.get("modelUsage")
    if not isinstance(model_usage, dict) or set(model_usage) != {expected_model}:
        raise CliContractError("cli_multiple_or_unknown_usage_models")
    if result.get("result") != texts[0]:
        raise CliContractError("cli_result_differs_from_assistant")
    usage = _numeric_usage(result.get("usage"))
    if not usage or "input_tokens" not in usage or "output_tokens" not in usage:
        raise CliContractError("cli_missing_token_usage")
    return {"raw_text": _safe_text(texts[0]), "returned_model": message["model"], "native_stop_reason": native_stop,
            "usage": usage, "cli_model_usage": {key: _numeric_usage(value) for key, value in model_usage.items()},
            "cli_num_turns": 1, "cli_assistant_message_count": 1, "cli_init_model": init["model"],
            "cli_events_redacted": summaries}


def _terminate_tree(process):
    if os.name == "nt":
        taskkill = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/taskkill.exe"
        subprocess.run([str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=15, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.kill()


class ClaudeCliProvider:
    synthetic = False
    prompt = bench.PROMPT
    MAX_RESPONSE_BYTES = 8 * 1024 * 1024

    def __init__(self, *, executable=DEFAULT_EXE, executable_sha256=DEFAULT_EXE_SHA256,
                 model=MODEL, expected_response_model=MODEL, max_output_tokens=4096, effort="high", timeout=180):
        if model != MODEL or expected_response_model != MODEL or effort != "high":
            raise ValueError("This execution profile requires exact Opus 5.5 and high effort")
        if (type(max_output_tokens) is not int or max_output_tokens <= 0
                or type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("Positive output token budget and timeout required")
        self.executable = Path(executable).resolve()
        self.executable_sha256 = bench._sha(self.executable.read_bytes())
        if self.executable_sha256 != executable_sha256:
            raise ValueError("CLI binary differs from the frozen executable hash")
        self.model, self.expected_response_model = model, expected_response_model
        self.base_url, self.timeout = "cli://reclaude/" + model, timeout
        self.generation_parameters = {"max_tokens": max_output_tokens, "effort": effort}
        self.traces = []

    def predict(self, sample, taxonomy):
        bench.validate_manifest({"schema_version": 1, "benchmark_id": "cli-request", "samples": [sample]}, taxonomy)
        context = bench._json_text(bench.visible_payload(sample, taxonomy))
        parts, hashes = [{"type": "text", "text": context}], {}
        for frame in sample["frames"]:
            encoded = bench._image_data_url(frame["image_path"])
            header, payload = encoded.split(",", 1)
            raw = base64.b64decode(payload)
            digest = bench._sha(raw)
            if digest != frame["sha256"]:
                raise ValueError("Input image differs from frozen hash")
            hashes[frame["frame_id"]] = digest
            parts.extend([{"type": "text", "text": f"Frame ID: {frame['frame_id']}; timestamp_ms: {frame['timestamp_ms']}"},
                          {"type": "image", "source": {"type": "base64", "media_type": header[5:].split(";", 1)[0], "data": payload}}])
        stdin = (bench._json_text({"type": "user", "message": {"role": "user", "content": parts}, "parent_tool_use_id": None}) + "\n").encode("utf-8")
        trace = {"status": "pending", "requested_model": self.model, "returned_model": None,
                 "expected_response_model": self.expected_response_model, "task_kind": sample["task_kind"],
                 "image_sha256": hashes, "system_prompt_sha256": bench._sha(self.prompt.encode()),
                 "payload_sha256": bench._sha(context.encode()), "usage": None,
                 "generation_parameters": deepcopy(self.generation_parameters), "timeout_seconds": self.timeout,
                 "transport": "claude_code_cli", "cli_binary_sha256": self.executable_sha256,
                 "cli_stdin_sha256": bench._sha(stdin), "input_representation": "claude_cli_image_blocks",
                 "temperature_control": "not available; omitted", "cli_invocations": 1,
                 "retry_controls": {**ENV_CONTROLS, "CLAUDE_CODE_RETRY_WATCHDOG": "removed"}}
        started, process = time.perf_counter(), None
        try:
            with tempfile.TemporaryDirectory(prefix="satori-claude-no-gt-") as directory:
                cwd = Path(directory)
                prompt_file, settings_file, mcp_file = cwd / "system-prompt.txt", cwd / "settings.json", cwd / "mcp.json"
                prompt_file.write_bytes(self.prompt.encode("utf-8"))
                settings_file.write_text('{"fallbackModel":[],"disableAllHooks":true}', encoding="utf-8")
                mcp_file.write_text('{"mcpServers":{}}', encoding="utf-8")
                command = [str(self.executable), "-p", "--safe-mode", "--model", self.model, "--effort", "high",
                           "--system-prompt-file", str(prompt_file), "--input-format", "stream-json",
                           "--output-format", "stream-json", "--verbose", "--tools", "", "--disallowedTools", "*",
                           "--strict-mcp-config", "--mcp-config", str(mcp_file), "--settings", str(settings_file),
                           "--max-turns", "1", "--no-session-persistence", "--disable-slash-commands"]
                env = os.environ.copy()
                env.update(ENV_CONTROLS)
                env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(self.generation_parameters["max_tokens"])
                env.pop("CLAUDE_CODE_RETRY_WATCHDOG", None)
                kwargs = {"stdin": subprocess.PIPE, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
                          "cwd": str(cwd), "env": env, "shell": False}
                if os.name == "nt":
                    kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                else:
                    kwargs["start_new_session"] = True
                process = subprocess.Popen(command, **kwargs)
                try:
                    stdout, stderr = process.communicate(input=stdin, timeout=self.timeout)
                except BaseException:
                    _terminate_tree(process)
                    process.communicate()
                    raise
                trace.update(cli_exit_code=process.returncode, cli_stdout_sha256=bench._sha(stdout),
                             cli_stderr_sha256=bench._sha(stderr), cli_stderr_bytes=len(stderr))
                if len(stdout) > self.MAX_RESPONSE_BYTES or len(stderr) > self.MAX_RESPONSE_BYTES:
                    raise CliContractError("cli_output_size_limit")
                if process.returncode != 0:
                    raise CliContractError("cli_nonzero_exit")
                parsed = parse_cli_output(stdout, stderr, expected_model=self.expected_response_model, evidence=trace)
                trace.update({key: value for key, value in parsed.items() if key != "raw_text"})
                trace.update(raw_json_text=parsed["raw_text"], finish_reason="stop")
                if parsed["usage"]["output_tokens"] > self.generation_parameters["max_tokens"]:
                    raise CliContractError("cli_output_budget_exceeded")
                try:
                    value = bench._strict_response_json(parsed["raw_text"])
                    trace["raw_json"] = value
                    result = bench.validate_prediction(value, sample, taxonomy)
                except (ValueError, TypeError, KeyError):
                    raise CliContractError("invalid_complete_prediction") from None
                trace["status"] = "success"
                return result
        except subprocess.TimeoutExpired:
            trace.update(status="error", error_code="cli_timeout")
            raise RuntimeError("CLI benchmark timeout") from None
        except CliContractError as exc:
            trace.update(status="error", error_code=exc.code)
            raise RuntimeError("CLI benchmark contract failure: " + exc.code) from None
        except BaseException as exc:
            trace.update(status="error", error_code="cli_" + type(exc).__name__)
            raise
        finally:
            trace["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            self.traces.append(trace)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "taxonomy", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--executable-sha256", default=DEFAULT_EXE_SHA256)
    args = parser.parse_args(argv)
    from run_paper_parallel import infer_parallel
    def factory():
        return ClaudeCliProvider(executable=args.executable, executable_sha256=args.executable_sha256)
    result = infer_parallel(args.manifest, args.taxonomy, args.output, factory, protocol_path=args.protocol,
                            max_workers=1, progress=lambda item: print(json.dumps(item), flush=True))
    print(json.dumps({key: result[key] for key in ("status", "planned_samples", "successful_samples", "failed_samples", "api_calls")}))
    return 0 if result["failed_samples"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
