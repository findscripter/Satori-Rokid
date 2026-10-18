"""Independent four-image adapter preserving Dispider's native silence outcome."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import time

import satori_paper_benchmark as benchmark

MODEL_REVISION = "7c5517eaaa4dc3788602af40e077063792c75f6c"
SOURCE_REVISION = "803f6b4634b3232293e89a38bcce9f5815c33ed4"
_SPEC = importlib.util.spec_from_file_location("satori_paper_dispider_backend", Path(__file__).with_name("backend.py"))
native = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(native)


class NativeSilenceError(ValueError):
    """Reaction generated its native keep_silent token, not a JSON prediction."""


class PaperDispiderProvider:
    synthetic = False
    prompt = benchmark.PROMPT
    model = "Dispider"
    expected_response_model = "Dispider"
    base_url = "local://Dispider/" + MODEL_REVISION
    timeout = None

    def __init__(self, backend):
        self.backend = backend
        self.generation_parameters = {"max_tokens": 4096, "temperature": 0.0}
        self.traces = []

    def predict(self, sample, taxonomy):
        benchmark.validate_manifest({"schema_version": 1, "benchmark_id": "request", "samples": [sample]}, taxonomy)
        if len(sample["frames"]) != 4:
            raise ValueError("This fixed protocol requires exactly four source frames")
        visible = benchmark.visible_payload(sample, taxonomy)
        context_text = benchmark._json_text(visible)
        hashes = {}
        for frame in sample["frames"]:
            digest = benchmark._sha(Path(frame["image_path"]).read_bytes())
            if digest != frame["sha256"]:
                raise ValueError("Input image differs from frozen hash")
            hashes[frame["frame_id"]] = digest
        trace = {
            "status": "pending", "requested_model": self.model, "returned_model": None,
            "expected_response_model": self.expected_response_model, "task_kind": sample["task_kind"],
            "image_sha256": hashes, "system_prompt_sha256": benchmark._sha(self.prompt.encode("utf-8")),
            "payload_sha256": benchmark._sha(context_text.encode("utf-8")), "usage": None,
            "generation_parameters": deepcopy(self.generation_parameters), "timeout_seconds": self.timeout,
            "model_revision": MODEL_REVISION, "source_revision": SOURCE_REVISION,
        }
        started = time.perf_counter()
        try:
            from PIL import Image
            frames = []
            for frame in sample["frames"]:
                with Image.open(frame["image_path"]) as source:
                    frames.append(source.convert("RGB"))
            result = self.backend.generate(system_prompt=self.prompt, context=visible, frames=frames,
                                           max_new_tokens=4096, temperature=0)
            trace.update(returned_model=self.model, finish_reason=result["finish_reason"],
                         raw_json_text=result["text"], raw_native_text=result.get("raw_native_text"),
                         generated_token_ids=result.get("generated_token_ids"),
                         preprocessing=result.get("preprocessing"),
                         usage={"prompt_tokens": result.get("prompt_tokens"),
                                "completion_tokens": result.get("completion_tokens")})
            if result.get("model_revision") != MODEL_REVISION:
                raise ValueError("Returned model revision differs from pinned revision")
            if result.get("preprocessing", {}).get("keep_silent_emitted"):
                raise NativeSilenceError("Native Reaction keep_silent; no JSON prediction")
            value = benchmark._strict_response_json(result["text"])
            trace["raw_json"] = value
            if result["finish_reason"] != "stop":
                raise ValueError("Generation did not finish with EOS")
            value = benchmark.validate_prediction(value, sample, taxonomy)
            trace["status"] = "success"
            return value
        except Exception as exc:
            trace.update(status="error", error_code=type(exc).__name__)
            raise
        finally:
            trace["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            self.traces.append(trace)
