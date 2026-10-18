"""Small local-only-weight Mage-VL HTTP adapter.

Runtime: Python 3.11+, PyTorch with CUDA/BF16, Transformers compatible with the
pinned model, Pillow, FastAPI, uvicorn. No model is downloaded by this service.
Only image data URLs are accepted; native video/SSH/tool execution is unsupported.
JSON mode is prompting plus output validation, NOT constrained decoding.
"""

import argparse
import asyncio
import base64
import binascii
from contextlib import asynccontextmanager
from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import hmac
import io
import json
import math
from pathlib import Path
import re
import threading
import time
import uuid
import warnings


MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGES = 32
MAX_IMAGE_PIXELS = 16_777_216
MAX_REQUEST_BYTES = 64 * 1024 * 1024
MAX_TEXT_CHARS = 100_000
MAX_NEW_TOKENS = 1024
HARD_CONTEXT_TOKENS = 131072
HARD_NEW_TOKENS = 8192
HARD_TEXT_CHARS = 2_000_000
HARD_IMAGES = 128


class APIError(Exception):
    def __init__(self, message, status_code=400, *, code="invalid_request",
                 finish_reason=None, usage=None, budget=None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.finish_reason = finish_reason
        self.usage = usage
        self.budget = budget


@dataclass
class ChatRequest:
    model: str
    messages: list
    images: list
    max_tokens: int
    temperature: float
    json_mode: bool


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    prompt_sha256: str | None = None
    budget: dict | None = None


@dataclass(frozen=True)
class ServerConfig:
    model_dir: Path
    served_model_name: str
    revision: str
    token_file: Path
    manifest_out: Path | None = None
    max_context_tokens: int = HARD_CONTEXT_TOKENS
    max_new_tokens: int = MAX_NEW_TOKENS
    default_new_tokens: int = MAX_NEW_TOKENS
    max_text_chars: int = MAX_TEXT_CHARS
    max_images: int = MAX_IMAGES

    def __post_init__(self):
        for name, ceiling in (("max_context_tokens", HARD_CONTEXT_TOKENS),
                              ("max_new_tokens", HARD_NEW_TOKENS),
                              ("default_new_tokens", HARD_NEW_TOKENS),
                              ("max_text_chars", HARD_TEXT_CHARS), ("max_images", HARD_IMAGES)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be an integer from 1 to {ceiling}")
        if self.default_new_tokens > self.max_new_tokens:
            raise ValueError("default_new_tokens must not exceed max_new_tokens")
        if self.max_new_tokens >= self.max_context_tokens:
            raise ValueError("Output ceiling must leave room for input within the total context")


def _meaningful_window(value):
    # Hugging Face uses huge placeholder integers when a tokenizer limit is unset.
    return value if type(value) is int and 0 < value < 10**12 else None


def context_metadata(config, model_config, tokenizer):
    text_config = getattr(model_config, "text_config", model_config)
    declared = _meaningful_window(getattr(text_config, "max_position_embeddings", None))
    tokenizer_window = _meaningful_window(getattr(tokenizer, "model_max_length", None))
    if declared is None:
        raise ValueError("The model must declare a finite text context window")
    declared_limits = [declared] + ([tokenizer_window] if tokenizer_window is not None else [])
    if config.max_context_tokens > min(declared_limits):
        raise ValueError("Operational context exceeds the declared model/tokenizer window")
    return {
        "model_context_window": declared,
        "tokenizer_context_window": tokenizer_window,
        "operational_context_tokens": config.max_context_tokens,
        "default_new_tokens": config.default_new_tokens,
        "max_new_tokens": config.max_new_tokens,
        "max_text_chars": config.max_text_chars,
        "max_images": config.max_images,
        "input_budget_at_default_output": config.max_context_tokens - config.default_new_tokens,
        "one_million_supported": False,
        "validated_max_input_tokens": None,
        "observed_successful_input_tokens": 0,
        "observed_input_scope": "Largest input whose generation and decode returned; this does not validate output quality or long-context capability.",
        "capacity_validation": "Configured admission ceilings only; startup does not validate these lengths on this GPU.",
        "context_budget_policy": "Expanded input_ids length plus reserved output must fit operational, model and meaningful tokenizer limits.",
    }


def check_context_budget(input_tokens, reserved_output, config, *, model_context_window, tokenizer_context_window=None):
    if type(input_tokens) is not int or input_tokens < 0:
        raise APIError("Invalid tokenized input length")
    if type(reserved_output) is not int or not 1 <= reserved_output <= config.max_new_tokens:
        raise APIError("Invalid reserved output token count")
    limits = [config.max_context_tokens, model_context_window]
    if tokenizer_context_window is not None:
        limits.append(tokenizer_context_window)
    maximum = min(limits)
    budget = {"input_tokens": input_tokens, "reserved_output": reserved_output,
              "max_context": maximum, "operational_context": config.max_context_tokens,
              "model_context_window": model_context_window,
              "tokenizer_context_window": tokenizer_context_window}
    if input_tokens + reserved_output > maximum:
        raise APIError("Tokenized input plus reserved output exceeds the context budget", 400,
                       code="context_length_exceeded", budget=budget)
    return budget


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON keys")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Nonfinite JSON number")


def decode_request_json(raw, max_bytes=MAX_REQUEST_BYTES):
    if len(raw) > max_bytes:
        raise APIError("Request exceeds the size limit", 413, code="request_too_large")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_json_pairs,
                          parse_constant=_reject_constant)
    except (ValueError, UnicodeError):
        raise APIError("Request must contain valid finite JSON", code="invalid_json") from None


def read_token(path):
    try:
        token = Path(path).read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        raise ValueError("Cannot read bearer token file") from None
    if not 16 <= len(token) <= 512 or any(character.isspace() for character in token):
        raise ValueError("Bearer token must be a nonempty ASCII token without whitespace")
    return token


def valid_authorization(header, token):
    if not isinstance(header, str) or not header.startswith("Bearer "):
        return False
    supplied = header[7:]
    if not supplied.isascii():
        return False
    return hmac.compare_digest(supplied, token)


def _pillow_decode(raw, mime):
    from PIL import Image, UnidentifiedImageError
    expected = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}[mime]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                width, height = image.size
                if (image.format != expected or width <= 0 or height <= 0
                        or width * height > MAX_IMAGE_PIXELS):
                    raise ValueError("Unsupported image dimensions or format")
                image.load()
                return image.convert("RGB").copy()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombWarning,
            Image.DecompressionBombError):
        raise APIError("Image data could not be decoded safely", code="invalid_image") from None


def _decode_data_url(url, decoder):
    if not isinstance(url, str):
        raise APIError("Image must use an inline image data URL", code="invalid_image")
    match = re.fullmatch(r"data:(image/png|image/jpeg|image/webp);base64,([A-Za-z0-9+/=]+)", url)
    if not match:
        raise APIError("Only PNG, JPEG or WebP base64 data URLs are accepted", code="invalid_image")
    mime, encoded = match.groups()
    if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise APIError("Image exceeds the decoded byte limit", 413, code="image_too_large")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise APIError("Invalid base64 image data", code="invalid_image") from None
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise APIError("Image exceeds the decoded byte limit", 413, code="image_too_large")
    signature_ok = {
        "image/png": raw.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/jpeg": raw.startswith(b"\xff\xd8\xff"),
        "image/webp": raw.startswith(b"RIFF") and raw[8:12] == b"WEBP",
    }[mime]
    if not signature_ok:
        raise APIError("Image MIME type does not match its bytes", code="invalid_image")
    try:
        return decoder(raw, mime)
    except APIError:
        raise
    except Exception:
        raise APIError("Image data could not be decoded safely", code="invalid_image") from None


def parse_chat_request(payload, served_model_name, *, image_decoder=None, config=None):
    if not isinstance(payload, dict):
        raise APIError("Request must be a JSON object")
    allowed = {"model", "messages", "stream", "response_format", "max_tokens",
               "max_completion_tokens", "temperature", "n"}
    if set(payload) - allowed:
        raise APIError("Unsupported request fields; tools and external resources are disabled")
    if payload.get("model") != served_model_name:
        raise APIError("Requested model is not served here", 404, code="model_not_found")
    if payload.get("stream", False) is not False or type(payload.get("n", 1)) is not int or payload.get("n", 1) != 1:
        raise APIError("Only one non-streaming completion is supported")
    output_ceiling = config.max_new_tokens if config is not None else MAX_NEW_TOKENS
    output_default = config.default_new_tokens if config is not None else MAX_NEW_TOKENS
    text_ceiling = config.max_text_chars if config is not None else MAX_TEXT_CHARS
    image_ceiling = config.max_images if config is not None else MAX_IMAGES
    max_tokens = payload.get("max_tokens", payload.get("max_completion_tokens", output_default))
    if ("max_tokens" in payload and "max_completion_tokens" in payload
            and payload["max_tokens"] != payload["max_completion_tokens"]):
        raise APIError("Conflicting token limits")
    if type(max_tokens) is not int or not 1 <= max_tokens <= output_ceiling:
        raise APIError(f"max_tokens must be an integer from 1 to {output_ceiling}")
    temperature = payload.get("temperature", 0.0)
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or not 0 <= temperature <= 2):
        raise APIError("temperature must be a finite number between 0 and 2")
    mode = payload.get("response_format")
    if mode is not None and mode != {"type": "json_object"}:
        raise APIError("Only response_format=json_object is supported")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 32:
        raise APIError("messages must contain 1 to 32 entries")
    decoder = image_decoder or _pillow_decode
    normalized, images, text_count = [], [], 0
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise APIError("Each message requires exactly role and content")
        role = message["role"]
        if role not in ("system", "user", "assistant"):
            raise APIError("Unsupported message role")
        content = message["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list) or not content:
            raise APIError("Message content must be text or a nonempty content array")
        parts = []
        for part in content:
            if not isinstance(part, dict):
                raise APIError("Content entries must be objects")
            if part.get("type") == "text":
                if set(part) != {"type", "text"} or not isinstance(part["text"], str):
                    raise APIError("Text entries require text strings")
                text_count += len(part["text"])
                if text_count > text_ceiling:
                    raise APIError("Text exceeds the supported length", 413, code="text_too_large")
                parts.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image_url":
                if role != "user" or set(part) != {"type", "image_url"}:
                    raise APIError("Images are accepted only in user messages")
                image = part["image_url"]
                if (not isinstance(image, dict) or set(image) - {"url", "detail"}
                        or "url" not in image or image.get("detail", "auto") not in ("auto", "low", "high")):
                    raise APIError("Invalid image_url entry")
                if len(images) >= image_ceiling:
                    raise APIError(f"At most {image_ceiling} images are accepted", 413, code="too_many_images")
                images.append(_decode_data_url(image["url"], decoder))
                parts.append({"type": "image"})
            else:
                raise APIError("Only text and inline image_url content is supported")
        normalized.append({"role": role, "content": parts})
    if not images:
        raise APIError("This vision endpoint requires at least one image")
    if mode is not None:
        instruction = "Return exactly one valid JSON object. Do not output markdown fences, commentary, or reasoning."
        if normalized[0]["role"] == "system":
            normalized[0]["content"].append({"type": "text", "text": instruction})
        else:
            normalized.insert(0, {"role": "system", "content": [{"type": "text", "text": instruction}]})
    return ChatRequest(served_model_name, normalized, images, max_tokens, float(temperature), mode is not None)


def build_completion(spec, result, served_model_name, revision, latency_ms):
    usage = {"prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
             "total_tokens": result.prompt_tokens + result.completion_tokens}
    if result.finish_reason not in {"stop", "length"}:
        raise APIError("Model generation ended with an unsupported termination", 502,
                       code="invalid_generation", usage=usage)
    if spec.json_mode:
        if result.finish_reason != "stop":
            raise APIError("Model output reached its token limit", 502, code="incomplete_json",
                           finish_reason=result.finish_reason, usage=usage, budget=result.budget)
        try:
            parsed = json.loads(result.text, object_pairs_hook=_json_pairs, parse_constant=_reject_constant)
            if not isinstance(parsed, dict):
                raise ValueError
        except (ValueError, TypeError):
            raise APIError("Model did not return a complete JSON object", 502, code="invalid_json_output",
                           finish_reason=result.finish_reason, usage=usage, budget=result.budget) from None
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
        "model": served_model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": result.text},
                     "finish_reason": result.finish_reason}],
        "usage": usage,
        "server_metadata": {"revision": revision, "revision_source": "operator_declared",
                            "constrained_decoding": False,
                            "json_mode": "prompt_and_validation" if spec.json_mode else "disabled",
                            "latency_ms": round(latency_ms, 3), "input_images": len(spec.images),
                            "rendered_prompt_sha256": result.prompt_sha256, "context_budget": result.budget},
    }


def configure_image_limits(processor):
    """Apply the official pixel-area limits to recognized processor settings."""
    image_processor = getattr(processor, "image_processor", None)
    configured = False
    if hasattr(image_processor, "min_pixels") and hasattr(image_processor, "max_pixels"):
        image_processor.min_pixels = 3136
        image_processor.max_pixels = 150000
        configured = True
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict) and {"shortest_edge", "longest_edge"} <= set(size):
        image_processor.size = {"shortest_edge": 3136, "longest_edge": 150000}
        configured = True
    elif is_dataclass(size) and not isinstance(size, type):
        names = {field.name for field in fields(size)}
        if {"shortest_edge", "longest_edge"} <= names:
            updates = {"shortest_edge": 3136, "longest_edge": 150000}
            updates.update({key: None for key in ("height", "width", "max_height", "max_width") if key in names})
            image_processor.size = replace(size, **updates)
            configured = True
    if not configured:
        raise RuntimeError("Image processor has no recognized pixel-bound settings; review pinned model compatibility")
    return {"min_pixels": 3136, "max_pixels": 150000}


def register_local_mage_config(model_dir):
    """Avoid nested AutoTokenizer trust prompts without editing model sources."""
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config_class = get_class_from_dynamic_module(
        "configuration_mage_vl.MageVLConfig", str(model_dir), local_files_only=True)
    AutoConfig.register("mage_vl", config_class, exist_ok=True)


class ModelBackend:
    """Single local weight load; CUDA device zero is relative to gpurun visibility."""

    def __init__(self, config):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoProcessor

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("A visible CUDA device with BF16 support is required")
        if not config.model_dir.is_dir():
            raise RuntimeError("Local model directory does not exist")
        self.torch = torch
        self.config = config
        self._lock = threading.Lock()
        register_local_mage_config(config.model_dir)
        self.processor = AutoProcessor.from_pretrained(
            str(config.model_dir), trust_remote_code=True, local_files_only=True,
            min_pixels=3136, max_pixels=150000)
        image_limits = configure_image_limits(self.processor)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(config.model_dir), trust_remote_code=True, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}).eval()
        self.metadata = {
            "served_model": config.served_model_name, "revision": config.revision,
            "revision_source": "operator_declared", "config_commit": getattr(self.model.config, "_commit_hash", None),
            "logical_device": "cuda:0", "dtype": "bfloat16", "attention": "sdpa",
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "native_video_supported": False, "constrained_json_decoding": False,
            "image_processor_limits": image_limits,
        }
        self.metadata.update(context_metadata(config, self.model.config, getattr(self.processor, "tokenizer", None)))
        self.model_context_window = self.metadata["model_context_window"]
        self.tokenizer_context_window = self.metadata["tokenizer_context_window"]

    def generate(self, spec):
        with self._lock, self.torch.inference_mode():
            prompt = self.processor.apply_chat_template(spec.messages, tokenize=False, add_generation_prompt=True,
                                                        enable_thinking=False)
            inputs = self.processor(text=[prompt], images=spec.images, padding=True, return_tensors="pt", truncation=False)
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            budget = check_context_budget(prompt_tokens, spec.max_tokens, self.config,
                                          model_context_window=self.model_context_window,
                                          tokenizer_context_window=self.tokenizer_context_window)
            if hasattr(inputs, "to"):
                inputs = inputs.to(self.model.device)
            else:
                inputs = {key: value.to(self.model.device) if hasattr(value, "to") else value
                          for key, value in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.model.dtype)
            options = {"max_new_tokens": spec.max_tokens, "do_sample": spec.temperature > 0,
                       "return_dict_in_generate": True, "output_scores": False, "use_cache": True}
            if spec.temperature > 0:
                options["temperature"] = spec.temperature
            output = self.model.generate(**inputs, **options)
            sequence = output.sequences if hasattr(output, "sequences") else output
            generated = sequence[0, prompt_tokens:]
            completion_tokens = int(generated.shape[-1])
            eos = getattr(self.model.generation_config, "eos_token_id", None)
            eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) if eos is not None else set()
            ended_with_eos = completion_tokens > 0 and int(generated[-1].item()) in eos_ids
            reason = "length" if completion_tokens >= spec.max_tokens and not ended_with_eos else "stop"
            text = self.processor.batch_decode(
                [generated], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            self.metadata["observed_successful_input_tokens"] = max(
                self.metadata["observed_successful_input_tokens"], prompt_tokens)
            return GenerationResult(text, prompt_tokens, completion_tokens, reason,
                                    hashlib.sha256(prompt.encode("utf-8")).hexdigest(), budget)

    def close(self):
        with self._lock:
            self.model = None
            self.processor = None
            self.torch.cuda.empty_cache()


def _event(name, **metadata):
    print(json.dumps({"event": name, **metadata}, ensure_ascii=True, allow_nan=False), flush=True)


def create_app(config, backend_factory=ModelBackend):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    active = {"ready": False, "token": None, "backend": None}
    gate = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        active["token"] = read_token(config.token_file)
        _event("model_loading", model=config.served_model_name, revision=config.revision,
               logical_device="cuda:0", dtype="bfloat16", attention="sdpa")
        try:
            active["backend"] = await asyncio.to_thread(backend_factory, config)
            active["ready"] = True
            metadata = active["backend"].metadata
            if config.manifest_out is not None:
                config.manifest_out.parent.mkdir(parents=True, exist_ok=True)
                config.manifest_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            _event("ready", **metadata)
            yield
        except Exception as exc:
            _event("server_error", error_type=type(exc).__name__)
            raise RuntimeError("Mage-VL server lifecycle failed; inspect deployment and model compatibility") from None
        finally:
            active["ready"] = False
            _event("shutdown_started")
            if active["backend"] is not None:
                await asyncio.to_thread(active["backend"].close)
            active["token"] = None
            _event("shutdown_complete")

    app = FastAPI(title="Local Mage-VL image adapter", lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url=None)

    @app.exception_handler(APIError)
    async def api_error_handler(request, exc):
        body = {"error": {"message": str(exc), "type": "server_error" if exc.status_code >= 500 else "invalid_request_error",
                          "code": exc.code}}
        if exc.finish_reason:
            body["error"]["finish_reason"] = exc.finish_reason
        if exc.usage is not None:
            body["usage"] = exc.usage
        if exc.budget is not None:
            body["error"]["budget"] = exc.budget
        headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
        return JSONResponse(body, status_code=exc.status_code, headers=headers)

    def authorize(req):
        if not valid_authorization(req.headers.get("authorization"), active["token"] or ""):
            raise APIError("Valid bearer authentication is required", 401, code="unauthorized")
        if not active["ready"]:
            raise APIError("Model is not ready", 503, code="not_ready")

    @app.get("/health")
    async def health(req: Request):
        authorize(req)
        return {"status": "ready", "busy": gate.locked(), **active["backend"].metadata}

    @app.get("/v1/models")
    async def models(req: Request):
        authorize(req)
        return {"object": "list", "data": [{"id": config.served_model_name, "object": "model",
                                           "created": int(time.time()), "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    async def completions(req: Request):
        authorize(req)
        if gate.locked():
            raise APIError("The single model worker is busy", 429, code="model_busy")
        async with gate:
            started = time.perf_counter()
            content_length = req.headers.get("content-length")
            if content_length is not None:
                try:
                    declared = int(content_length)
                except ValueError:
                    raise APIError("Invalid Content-Length") from None
                if declared < 0 or declared > MAX_REQUEST_BYTES:
                    raise APIError("Request exceeds the size limit", 413, code="request_too_large")
            if req.headers.get("content-type", "").split(";", 1)[0].lower() != "application/json":
                raise APIError("Content-Type must be application/json", 415, code="unsupported_media_type")
            chunks, total = [], 0
            async for chunk in req.stream():
                total += len(chunk)
                if total > MAX_REQUEST_BYTES:
                    raise APIError("Request exceeds the size limit", 413, code="request_too_large")
                chunks.append(chunk)
            payload = decode_request_json(b"".join(chunks))
            spec = await asyncio.to_thread(parse_chat_request, payload, config.served_model_name, config=config)
            try:
                result = await asyncio.to_thread(active["backend"].generate, spec)
            except APIError:
                raise
            except Exception as exc:
                _event("generation_failed", error_type=type(exc).__name__)
                raise APIError("Model generation failed", 500, code="generation_failed") from None
            latency_ms = (time.perf_counter() - started) * 1000
            response = build_completion(spec, result, config.served_model_name, config.revision, latency_ms)
            _event("completion", model=config.served_model_name, image_count=len(spec.images),
                   prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
                   finish_reason=result.finish_reason, latency_ms=round(latency_ms, 3))
            return response

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description="Single-process local-weight Mage-VL image HTTP adapter")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--revision", required=True, help="Exact downloaded model revision; recorded as operator-declared")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-context-tokens", type=int, default=HARD_CONTEXT_TOKENS,
                        help="Total expanded input plus reserved output; ceiling 131072, not a validated GPU capacity")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--default-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--max-text-chars", type=int, default=MAX_TEXT_CHARS)
    parser.add_argument("--max-images", type=int, default=MAX_IMAGES)
    args = parser.parse_args(argv)
    if not args.served_model_name.strip() or not args.revision.strip():
        parser.error("served-model-name and revision must not be empty")
    config = ServerConfig(args.model_dir.resolve(), args.served_model_name, args.revision,
                          args.token_file.resolve(), args.manifest_out,
                          max_context_tokens=args.max_context_tokens, max_new_tokens=args.max_new_tokens,
                          default_new_tokens=args.default_new_tokens, max_text_chars=args.max_text_chars,
                          max_images=args.max_images)
    import uvicorn
    uvicorn.run(create_app(config), host=args.host, port=args.port, workers=1,
                access_log=False, log_level="warning", timeout_graceful_shutdown=120)


if __name__ == "__main__":
    main()
