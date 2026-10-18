"""VideoChat3 adapter for the frozen Satori-source module probes.

No GT, source filenames or action history are accepted by this provider. Model
loading is delegated to the existing pinned backend and requires gpurun.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import time

import satori_paper_benchmark as benchmark


_SPEC = importlib.util.spec_from_file_location(
    "satori_paper_pinned_videochat3", Path(__file__).parents[1] / "videochat3" / "backend.py"
)
native = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(native)


def validate_registry(value, manifest):
    """Bind anonymous sample IDs to only native decoder technical metadata."""
    benchmark._fields(value, {"benchmark_id", "samples"}, "technical metadata registry")
    if value["benchmark_id"] != manifest["benchmark_id"]:
        raise ValueError("Registry benchmark ID mismatch")
    entries = value["samples"]
    expected = {sample["sample_id"] for sample in manifest["samples"]}
    if not isinstance(entries, dict) or set(entries) != expected:
        raise ValueError("Registry must cover exactly the frozen manifest")
    clean = {}
    for sample in manifest["samples"]:
        sid = sample["sample_id"]
        metadata = native.validate_metadata(entries[sid], len(sample["frames"]))
        if any(name in metadata and (type(metadata[name]) is not int or metadata[name] <= 0)
               for name in ("width", "height")):
            raise ValueError("Registry dimensions must be positive integers")
        if (not math.isfinite(metadata["duration"])
                or ("duration" in entries[sid] and
                    (isinstance(entries[sid]["duration"], bool)
                     or not isinstance(entries[sid]["duration"], (int, float))
                     or not math.isfinite(entries[sid]["duration"])))):
            raise ValueError("Registry duration must be finite")
        if "video_backend" in metadata and metadata["video_backend"] not in ("opencv", "pyav", "decord"):
            raise ValueError("Registry backend must be a decoder name")
        if any(abs(frame["timestamp_ms"] - index / metadata["fps"] * 1000) > 1
               for frame, index in zip(sample["frames"], metadata["frames_indices"])):
            raise ValueError("Registry and frozen frame timestamps disagree")
        clean[sid] = metadata
    return clean


def audit_native_metadata(run_dir, registry):
    """Reconcile native timestamps with the protocol and all attempted inputs."""
    run_dir = Path(run_dir)
    registry_bytes = (run_dir / "metadata-registry.json").read_bytes()
    registry_hash = benchmark._sha(registry_bytes)
    saved = benchmark._strict_response_json(registry_bytes.decode("utf-8-sig"))
    if saved != registry:
        raise ValueError("Technical metadata snapshot differs from frozen registry")
    manifest, _ = benchmark._read(run_dir / "manifest.json")
    taxonomy, _ = benchmark._read(run_dir / "taxonomy.json")
    protocol, _ = benchmark._read(run_dir / "protocol.json")
    if protocol.get("technical_metadata_registry_sha256") != registry_hash:
        raise ValueError("Protocol does not bind technical metadata registry")
    entries = validate_registry(saved, manifest)
    samples = {row["sample_id"]: row for row in manifest["samples"]}
    count = 0
    for line in (run_dir / "api-calls.jsonl").read_text(encoding="utf-8-sig").splitlines():
        trace = benchmark._strict_response_json(line)
        if (trace.get("metadata_registry_sha256") != registry_hash
                or trace.get("video_metadata") != entries[trace["sample_id"]]):
            raise ValueError("Trace metadata differs from protocol-bound native metadata")
        if trace.get("preprocessing") is not None:
            prep = trace["preprocessing"]
            metadata = entries[trace["sample_id"]]
            if (prep.get("original_input_frames") != len(metadata["frames_indices"])
                    or prep.get("source_frame_indices") != metadata["frames_indices"]
                    or prep.get("source_fps") != metadata["fps"]
                    or prep.get("timestamps_seconds") != [i / metadata["fps"] for i in metadata["frames_indices"]]
                    or prep.get("do_sample_frames") is not False
                    or not isinstance(prep.get("video_grid_thw"), list)
                    or len(prep["video_grid_thw"]) != 1
                    or prep["video_grid_thw"][0][0] != len(metadata["frames_indices"])
                    or prep.get("user_text") != native.render_user_text(
                        benchmark.visible_payload(samples[trace["sample_id"]], taxonomy))):
                raise ValueError("Native preprocessing differs from frozen timeline")
        count += 1
    return {"passed": True, "attempted_inputs_checked": count, "metadata_registry_sha256": registry_hash}


class PaperBackend(native.Backend):
    """Same pinned model/generation; support four or eight unique input frames."""

    def prepare_inputs(self, *, system_prompt, context, frames, frame_metadata):
        import numpy as np
        import torch

        count = len(frames)
        if count not in (4, 8):
            raise ValueError("Exactly four or eight frozen frames required")
        metadata = native.validate_metadata(frame_metadata, count)
        context_frames = context.get("frames", [])
        if len(context_frames) != count or any(
            abs(item["timestamp_ms"] - index / metadata["fps"] * 1000) > 1
            for item, index in zip(context_frames, metadata["frames_indices"])
        ):
            raise ValueError("Frozen prompt and original timestamps disagree")
        arrays = [np.array(frame.convert("RGB"), copy=True) for frame in frames]
        if len({array.shape for array in arrays}) != 1:
            raise ValueError("Native video frames require equal dimensions")
        tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "video", "video": "<provided-frozen-frame-tensor>"},
                {"type": "text", "text": native.render_user_text(context)},
            ]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], videos=[tensor],
            videos_kwargs={"size": {"shortest_edge": native.MIN_PIXELS, "longest_edge": native.MAX_PIXELS},
                           "do_sample_frames": False, "video_metadata": [metadata]},
            return_tensors="pt",
        )
        grid = inputs["video_grid_thw"].tolist()
        if len(grid) != 1 or grid[0][0] != count:
            raise RuntimeError("Processor resampled frozen frames")
        return inputs, {
            "representation": "native_video_tensor", "original_input_frames": count,
            "do_sample_frames": False, "video_grid_thw": grid,
            "temporal_merge_size": self.processor.video_processor.temporal_merge_size,
            "frame_min_pixels": native.MIN_PIXELS, "frame_max_pixels": native.MAX_PIXELS,
            "source_frame_indices": metadata["frames_indices"], "source_fps": metadata["fps"],
            "timestamps_seconds": [index / metadata["fps"] for index in metadata["frames_indices"]],
            "user_text": native.render_user_text(context),
        }


class PaperVideoChat3Provider:
    synthetic = False
    prompt = benchmark.PROMPT
    model = native.MODEL_ID
    expected_response_model = native.MODEL_ID
    base_url = "local://VideoChat3-4B/" + native.MODEL_REVISION
    timeout = None  # Direct local generation has no HTTP request or hard timeout.

    def __init__(self, backend, *, registry, manifest, registry_sha256=None):
        self.backend = backend
        self.metadata = validate_registry(registry, manifest)
        self.registry_sha256 = registry_sha256
        self.generation_parameters = {"max_tokens": 4096, "temperature": 0.0}
        self.traces = []

    def predict(self, sample, taxonomy):
        benchmark.validate_manifest({"schema_version": 1, "benchmark_id": "request", "samples": [sample]}, taxonomy)
        visible = benchmark.visible_payload(sample, taxonomy)
        context_text = benchmark._json_text(visible)
        hashes = {}
        for frame in sample["frames"]:
            digest = benchmark._sha(Path(frame["image_path"]).read_bytes())
            if digest != frame["sha256"]:
                raise ValueError("Input image differs from frozen hash")
            hashes[frame["frame_id"]] = digest
        metadata = deepcopy(self.metadata[sample["sample_id"]])
        if len(metadata["frames_indices"]) != len(sample["frames"]) or any(
            abs(frame["timestamp_ms"] - index / metadata["fps"] * 1000) > 1
            for frame, index in zip(sample["frames"], metadata["frames_indices"])
        ):
            raise ValueError("Registry and inference frame timestamps disagree")
        trace = {"status": "pending", "requested_model": self.model, "returned_model": None,
                 "expected_response_model": self.expected_response_model, "task_kind": sample["task_kind"],
                 "image_sha256": hashes, "system_prompt_sha256": benchmark._sha(self.prompt.encode("utf-8")),
                 "payload_sha256": benchmark._sha(context_text.encode("utf-8")), "usage": None,
                 "generation_parameters": deepcopy(self.generation_parameters), "timeout_seconds": self.timeout,
                 "metadata_registry_sha256": self.registry_sha256, "video_metadata": metadata,
                 "model_revision": native.MODEL_REVISION}
        started = time.perf_counter()
        try:
            from PIL import Image
            frames = []
            for frame in sample["frames"]:
                with Image.open(frame["image_path"]) as source:
                    frames.append(source.convert("RGB"))
            result = self.backend.generate(system_prompt=self.prompt, context=visible, frames=frames,
                                           max_new_tokens=4096, temperature=0, frame_metadata=metadata)
            trace.update(returned_model=self.model, finish_reason=result["finish_reason"],
                         raw_json_text=result["text"], preprocessing=result.get("preprocessing"),
                         usage={"prompt_tokens": result.get("prompt_tokens"),
                                "completion_tokens": result.get("completion_tokens")})
            if result.get("model_revision") != native.MODEL_REVISION:
                raise ValueError("Returned model revision differs from pinned revision")
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
