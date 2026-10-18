"""Loopback-only native-video backend adapter for the frozen thirty-two-JPEG ablation.

This accepts the existing gt_benchmark wire format. It never reads GT, opens
source videos, retries generation, strips markdown, or repairs model JSON.
Native model loaders live in sibling backend.py files, imported only at startup.
"""

import argparse
import asyncio
import base64
import binascii
from contextlib import asynccontextmanager
import hashlib
import hmac
import importlib.util
import io
import json
import math
from pathlib import Path
import re
import sys
import time
import uuid


MAX_REQUEST_BYTES = 64 * 1024 * 1024
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_777_216


class APIError(Exception):
    def __init__(self, code, status=400, *, finish_reason=None, usage=None):
        super().__init__(code)
        self.code, self.status = code, status
        self.finish_reason, self.usage = finish_reason, usage


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate key")
        value[key] = item
    return value


def strict_json(raw):
    def invalid(_):
        raise ValueError("nonfinite number")
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid)


def input_fingerprint(hashes, times):
    return sha(compact({"sha256": hashes, "timestamp_ms": times}).encode())


def decode_image(url):
    from PIL import Image
    import warnings
    match = re.fullmatch(r"data:(image/jpeg|image/png);base64,([A-Za-z0-9+/=]+)", url or "")
    if not match or len(match[2]) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise APIError("invalid_inline_image")
    try:
        raw = base64.b64decode(match[2], validate=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as opened:
                if opened.format != {"image/jpeg": "JPEG", "image/png": "PNG"}[match[1]]:
                    raise ValueError("mime mismatch")
                if opened.width * opened.height > MAX_IMAGE_PIXELS:
                    raise ValueError("image too large")
                opened.load()
                image = opened.convert("RGB").copy()
        return image, sha(raw)
    except (ValueError, OSError, binascii.Error, Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise APIError("invalid_inline_image") from None


def parse_request(body, model_id, registry, *, decoder=decode_image):
    try:
        allowed = {"model", "stream", "response_format", "messages", "max_tokens", "temperature"}
        if not isinstance(body, dict) or set(body) - allowed:
            raise ValueError
        if body["model"] != model_id or body["stream"] is not False:
            raise ValueError
        if body["response_format"] != {"type": "json_object"}:
            raise ValueError
        budget, temperature = body.get("max_tokens", 4096), body.get("temperature", 0)
        if type(budget) is not int or not 1 <= budget <= 8192:
            raise ValueError
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature != 0:
            raise ValueError
        messages = body["messages"]
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError
        system, user = messages
        if set(system) != {"role", "content"} or system["role"] != "system":
            raise ValueError
        if set(user) != {"role", "content"} or user["role"] != "user":
            raise ValueError
        prompt = system["content"]
        if not isinstance(prompt, str) or not 1 <= len(prompt) <= 100_000:
            raise ValueError
        parts = user["content"]
        if not isinstance(parts, list) or len(parts) != 65 or set(parts[0]) != {"type", "text"} or parts[0]["type"] != "text":
            raise ValueError
        context = strict_json(parts[0]["text"])
        if set(context) != {"scene_candidates", "action_candidates", "frames"}:
            raise ValueError
        if len(context["frames"]) != 32:
            raise ValueError
        for name, pattern, count in (("scene_candidates", r"s\d{3}", 16), ("action_candidates", r"c\d{3}", 157)):
            candidates = context[name]
            if not isinstance(candidates, list) or len(candidates) != count:
                raise ValueError
            if any(set(c) != {"id", "label"} or not re.fullmatch(pattern, c["id"]) or not isinstance(c["label"], str) for c in candidates):
                raise ValueError
            if len({c["id"] for c in candidates}) != count:
                raise ValueError
        if {c["id"] for c in context["action_candidates"]} != {f"c{i:03d}" for i in range(157)}:
            raise ValueError
        frames, hashes, times = [], [], []
        for i, info in enumerate(context["frames"]):
            if set(info) != {"frame_id", "timestamp_ms"} or info["frame_id"] != f"v{i:03d}":
                raise ValueError
            stamp = info["timestamp_ms"]
            if type(stamp) is not int or stamp < 0 or (times and stamp <= times[-1]):
                raise ValueError
            label, picture = parts[1 + 2*i:3 + 2*i]
            if label != {"type": "text", "text": f"Frame ID: v{i:03d}; timestamp_ms: {stamp}"}:
                raise ValueError
            if set(picture) != {"type", "image_url"} or picture["type"] != "image_url" or set(picture["image_url"]) != {"url"}:
                raise ValueError
            image, digest = decoder(picture["image_url"]["url"])
            frames.append(image)
            hashes.append(digest)
            times.append(stamp)
        fingerprint = input_fingerprint(hashes, times)
        metadata = registry.get(fingerprint)
        if metadata is None:
            raise APIError("unregistered_frozen_input")
        return {"system_prompt": prompt, "context": context, "frames": frames,
                "max_new_tokens": budget, "temperature": temperature, "frame_metadata": metadata}, {
                    "image_sha256": hashes, "timestamp_ms": times, "input_fingerprint": fingerprint,
                    "system_prompt_sha256": sha(prompt.encode()), "context_sha256": sha(compact(context).encode()),
                    "requested_max_new_tokens": budget, "temperature": temperature,
                }
    except APIError:
        raise
    except (KeyError, ValueError, TypeError, AttributeError, IndexError):
        raise APIError("invalid_benchmark_request") from None


def build_completion(result, spec, model_id, revision, token):
    usage = {}
    for key in ("prompt_tokens", "completion_tokens"):
        value = result.get(key)
        if value is not None:
            if type(value) is not int or value < 0:
                raise APIError("invalid_backend_usage", 502)
            usage[key] = value
    if len(usage) == 2:
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    reason = result.get("finish_reason")
    text = result.get("text")
    if reason != "stop":
        raise APIError("incomplete_generation", 502, finish_reason=reason if reason == "length" else None, usage=usage)
    try:
        if not isinstance(text, str) or len(text) > 500_000 or token in text:
            raise ValueError
        value = strict_json(text)
        if not isinstance(value, dict) or set(value) != {"scene_id", "action_ids", "evidence_frame_ids", "answer"}:
            raise ValueError
        scene, actions = value["scene_id"], value["action_ids"]
        if scene is not None and scene not in {c["id"] for c in spec["context"]["scene_candidates"]}:
            raise ValueError
        if actions is not None and (not isinstance(actions, list) or len(actions) != len(set(actions)) or any(a not in {c["id"] for c in spec["context"]["action_candidates"]} for a in actions)):
            raise ValueError
        evidence = value["evidence_frame_ids"]
        if not isinstance(evidence, list) or not evidence or len(evidence) != len(set(evidence)) or any(v not in {f"v{i:03d}" for i in range(32)} for v in evidence):
            raise ValueError
        if not isinstance(value["answer"], str) or not value["answer"].strip() or len(value["answer"]) > 4000:
            raise ValueError
    except (ValueError, TypeError, KeyError):
        raise APIError("invalid_complete_prediction", 502, finish_reason="stop", usage=usage) from None
    return {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
            "model": model_id, "system_fingerprint": revision,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": usage}


def create_app(*, model_id, revision, token_file, metadata_file, audit_file, loader):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    token = Path(token_file).read_text("ascii").strip()
    if not 16 <= len(token) <= 512 or any(c.isspace() for c in token):
        raise ValueError("invalid bearer token file")
    registry_raw = Path(metadata_file).read_bytes()
    registry = strict_json(registry_raw)["inputs"]
    registry_sha256 = sha(registry_raw)
    state, gate = {}, asyncio.Lock()
    @asynccontextmanager
    async def lifespan(_):
        state["backend"] = loader()
        if state["backend"].model_id != model_id:
            raise ValueError("backend model ID mismatch")
        yield
    app = FastAPI(lifespan=lifespan)
    def auth(req):
        value = req.headers.get("authorization", "")
        if not value.isascii() or not hmac.compare_digest(value, "Bearer " + token):
            raise APIError("unauthorized", 401)
    @app.exception_handler(APIError)
    async def api_error(_, exc):
        error = {"code": exc.code, "message": exc.code}
        if exc.finish_reason:
            error["finish_reason"] = exc.finish_reason
        value = {"error": error}
        if exc.usage is not None:
            value["usage"] = exc.usage
        return JSONResponse(value, status_code=exc.status)
    @app.get("/health")
    async def health(req: Request):
        auth(req)
        return {"status": "ready", "model": model_id, "revision": revision, "busy": gate.locked(),
                "registered_inputs": len(registry), "source_unique_frames": 32, "max_new_tokens": 8192,
                "metadata_registry_sha256": registry_sha256,
                "generation_retry": False, "json_repair": False}
    @app.get("/v1/models")
    async def models(req: Request):
        auth(req)
        return {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "local"}]}
    @app.post("/v1/chat/completions")
    async def complete(req: Request):
        auth(req)
        if gate.locked():
            raise APIError("model_busy", 429)
        if req.headers.get("content-type", "").split(";", 1)[0] != "application/json":
            raise APIError("unsupported_media_type", 415)
        async with gate:
            raw, size = [], 0
            async for part in req.stream():
                size += len(part)
                if size > MAX_REQUEST_BYTES:
                    raise APIError("request_too_large", 413)
                raw.append(part)
            try:
                body = strict_json(b"".join(raw))
            except (ValueError, UnicodeError):
                raise APIError("invalid_json") from None
            spec, trace = await asyncio.to_thread(parse_request, body, model_id, registry)
            trace.update(model=model_id, revision=revision, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), status="failed")
            started = time.perf_counter()
            result = None
            try:
                result = await asyncio.to_thread(state["backend"].generate, **spec)
                trace["result"] = {k: result.get(k) for k in ("text", "finish_reason", "prompt_tokens", "completion_tokens", "preprocessing")}
                response = build_completion(result, spec, model_id, revision, token)
                trace["status"] = "completed"
                return response
            except APIError as exc:
                trace["error_code"] = exc.code
                raise
            except Exception as exc:
                trace["error_code"], trace["error_type"] = "generation_failed", type(exc).__name__
                raise APIError("generation_failed", 502) from None
            finally:
                trace["generation_and_validation_ms"] = round((time.perf_counter()-started)*1000, 3)
                # The bearer credential never enters trace. Drop any unexpected
                # backend text containing it rather than serializing a secret.
                if token in compact(trace):
                    trace.pop("result", None)
                    trace["output_redacted"] = True
                with Path(audit_file).open("a", encoding="utf-8") as handle:
                    handle.write(compact(trace) + "\n")
    return app


def main():
    parser = argparse.ArgumentParser()
    for flag in ("backend-file", "model-path", "model-id", "revision", "token-file", "metadata-file", "audit-file"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    def loader():
        path = Path(args.backend_file).resolve()
        spec = importlib.util.spec_from_file_location("native_video_backend", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.Backend(args.model_path)
    import uvicorn
    app = create_app(model_id=args.model_id, revision=args.revision, token_file=args.token_file,
                     metadata_file=args.metadata_file, audit_file=args.audit_file, loader=loader)
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, access_log=False, log_level="warning",
                timeout_graceful_shutdown=120)


if __name__ == "__main__":
    main()
