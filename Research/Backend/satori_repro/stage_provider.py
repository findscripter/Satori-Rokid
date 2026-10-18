"""Auditable, one-request-per-stage Chat Completions transport for Satori.

Stage schemas remain the responsibility of SatoriMethodSession. Credentials are
read only when a request is made and are never retained in trace data.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from urllib import error, request

from .providers import _NoRedirect, _image_data_url, _json_loads, _validate_base_url
from .satori_method import build_stage_prompt


_CHINESE_FIELDS = (
    "\n请将 scene、action、objects 中的 name/state、guidance 的 text 等供人阅读的文字字段"
    "使用简体中文。保留所有 ID、JSON 字段名、状态枚举值及 modalities 的约定字符串，不翻译它们。"
)


def _json_text(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError("Satori stage payload must be finite JSON data") from None


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _redacted_trace(value, key):
    if isinstance(value, dict):
        return {str(name).replace(key, "[REDACTED]"): _redacted_trace(item, key)
                for name, item in value.items() if str(name).lower() != "reasoning_content"}
    if isinstance(value, list):
        return [_redacted_trace(item, key) for item in value]
    if isinstance(value, str):
        return value.replace(key, "[REDACTED]")
    return value


class SatoriStageProvider:
    """Explicit vision model transport with JSONL traces and no retries."""

    name = "satori-stage-openai-compatible"
    synthetic = False
    _MAX_RESPONSE_BYTES = 2 * 1024 * 1024

    def __init__(self, *, model, base_url, api_key_env="SATORI_API_KEY",
                 timeout=90, trace_path=None):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("An explicit non-empty stage model is required")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("Timeout must be a finite positive number")
        if not isinstance(api_key_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
            raise ValueError("Invalid API key environment variable name")
        self.model = model.strip()
        self.base_url = _validate_base_url(base_url)
        self.api_key_env = api_key_env
        self.timeout = float(timeout)
        self.trace_path = Path(trace_path) if trace_path is not None else None
        self.traces = []

    def _record(self, trace, started, key):
        trace["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        clean = _redacted_trace(trace, key)
        self.traces.append(clean)
        if self.trace_path is not None:
            try:
                self.trace_path.parent.mkdir(parents=True, exist_ok=True)
                with self.trace_path.open("a", encoding="utf-8") as output:
                    output.write(_json_text(clean) + "\n")
            except OSError:
                raise RuntimeError("Could not append Satori stage trace") from None

    def complete_json(self, stage, frames, payload) -> dict:
        if stage == "context_question":
            from .context_assistant import build_context_prompt
            system_prompt = build_context_prompt(payload) + _CHINESE_FIELDS
        else:
            system_prompt = build_stage_prompt(stage, payload) + _CHINESE_FIELDS
        if not isinstance(payload, dict):
            raise ValueError("Satori stage payload must be an object")
        payload_text = _json_text(payload)
        if not isinstance(frames, list) or not frames:
            raise ValueError("Satori stage inference requires at least one frame")
        metadata, image_parts, image_hashes = [], [], {}
        previous_time = -1
        for frame in frames:
            if not isinstance(frame, dict):
                raise ValueError("Each stage frame must be an object")
            frame_id, stamp = frame.get("frame_id"), frame.get("timestamp_ms")
            if (not isinstance(frame_id, str) or not frame_id.strip()
                    or frame_id in image_hashes or type(stamp) is not int or stamp < 0
                    or stamp <= previous_time):
                raise ValueError("Stage frames require unique IDs and increasing nonnegative timestamps")
            data_url = _image_data_url(frame.get("image_path"))
            image_hashes[frame_id] = _sha256(base64.b64decode(data_url.split(",", 1)[1]))
            metadata.append({"frame_id": frame_id, "timestamp_ms": stamp})
            image_parts.extend([
                {"type": "text", "text": f"Frame ID: {frame_id}; timestamp_ms: {stamp}"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ])
            previous_time = stamp
        content = [{"type": "text", "text": _json_text({"payload": payload, "frames": metadata})}]
        content.extend(image_parts)
        body = {
            "model": self.model, "stream": False, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": content}],
        }
        key = os.environ.get(self.api_key_env)
        if not key or not key.isascii() or any(char.isspace() for char in key):
            raise ValueError("API key environment variable is missing or invalid")
        req = request.Request(
            self.base_url + "/chat/completions", data=_json_text(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json",
                     "Authorization": "Bearer " + key}, method="POST",
        )
        trace = {
            "stage": stage, "status": "pending", "requested_model": self.model,
            "returned_model": None, "usage": None, "image_ids": [item["frame_id"] for item in metadata],
            "image_sha256": image_hashes, "system_prompt_sha256": _sha256(system_prompt.encode("utf-8")),
            "payload_sha256": _sha256(payload_text.encode("utf-8")),
        }
        started = time.perf_counter()
        try:
            with request.build_opener(_NoRedirect()).open(req, timeout=self.timeout) as response:
                raw = response.read(self._MAX_RESPONSE_BYTES + 1)
        except error.HTTPError as exc:
            status = exc.code
            exc.close()  # Do not consume or retain potentially sensitive error bodies.
            trace.update(status="error", error="http_error", http_status=status)
            self._record(trace, started, key)
            raise RuntimeError(f"Satori stage API returned HTTP {status}") from None
        except (error.URLError, OSError, TimeoutError):
            trace.update(status="error", error="connection_error")
            self._record(trace, started, key)
            raise RuntimeError("Satori stage API connection failed or timed out") from None
        if len(raw) > self._MAX_RESPONSE_BYTES:
            trace.update(status="error", error="response_size_limit")
            self._record(trace, started, key)
            raise RuntimeError("Satori stage API response exceeds the size limit") from None
        try:
            envelope = _json_loads(raw.decode("utf-8"))
            returned_model = envelope.get("model")
            trace["returned_model"] = returned_model if isinstance(returned_model, str) else None
            usage = envelope.get("usage")
            trace["usage"] = usage if isinstance(usage, dict) else None
            choice = envelope["choices"][0]
            message = choice["message"]
            if choice["finish_reason"] != "stop" or message.get("refusal"):
                raise ValueError
            raw_content = message["content"]
            if not isinstance(raw_content, str):
                raise ValueError
            result = _json_loads(raw_content)
            if not isinstance(result, dict):
                raise ValueError
        except (ValueError, TypeError, KeyError, IndexError, AttributeError, UnicodeError):
            trace.update(status="error", error="invalid_complete_json")
            self._record(trace, started, key)
            raise RuntimeError("Satori stage API did not return a complete JSON object") from None
        trace.update(status="success", raw_json=result)
        self._record(trace, started, key)
        return result
