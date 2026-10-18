"""VideoChat3 native video backend; no labels, filenames or sample history are used.

Weights and remote model code must be the verified, fixed local snapshot.
Construct this backend ONLY inside a GPU allocation made by the task owner.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

MODEL_REVISION = "9f6107bdfee9686dfe493444022f5cf79a1a6acf"
MODEL_ID = "VideoChat3-4B"
MIN_PIXELS = 56 * 56
MAX_PIXELS = 448 * 448


def require_gpu_allocation():
    if not os.environ.get("GPURUN_JOB") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("model_loading_requires_gpurun_GPU_allocation")


def render_user_text(context: dict) -> str:
    """Frozen shared text, also used by the independently deployed Dispider."""
    text = json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    frame_info = context.get("frames", [])
    return text + "\n" + "\n".join(
        f"Frame ID: {frame['frame_id']}; timestamp_ms: {frame['timestamp_ms']}"
        for frame in frame_info
    )


def validate_metadata(metadata: dict, frame_count: int = 8) -> dict:
    allowed = {"fps", "total_num_frames", "duration", "width", "height", "frames_indices", "video_backend"}
    if not isinstance(metadata, dict) or set(metadata) - allowed:
        raise ValueError("invalid_or_unexpected_video_metadata_fields")
    fps = metadata.get("fps")
    total = metadata.get("total_num_frames")
    indices = metadata.get("frames_indices")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("invalid_original_fps")
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise ValueError("invalid_original_frame_count")
    if not isinstance(indices, list) or len(indices) != frame_count:
        raise ValueError("incorrect_frame_index_count")
    if any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < total for x in indices):
        raise ValueError("invalid_original_frame_indices")
    if sorted(set(indices)) != indices:
        raise ValueError("frames_must_be_unique_and_in_time_order")
    if "duration" in metadata and abs(float(metadata["duration"]) * fps - total) > 1e-6:
        raise ValueError("original_video_duration_frame_count_mismatch")
    clean = dict(metadata)
    clean["duration"] = total / fps
    return clean


def classify_finish(token_ids: list[int], eos_token_id: int | list[int] | None) -> str:
    eos = [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id or []
    return "stop" if token_ids and token_ids[-1] in eos else "length"


class Backend:
    model_id = MODEL_ID

    def __init__(self, model_path: str):
        require_gpu_allocation()
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

        self.model_path = str(Path(model_path).resolve())
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)
        # The official implementation explicitly offers SDPA. No checkpoint edit.
        config.vision_config.attn_impl = "sdpa"
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True, local_files_only=True,
            min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, config=config, trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa",
        ).eval()

    def prepare_inputs(self, *, system_prompt: str, context: dict, frames: list, frame_metadata: dict):
        """Build one video tensor from exactly the supplied 8 unique images."""
        import numpy as np
        import torch

        if len(frames) != 8:
            raise ValueError("exactly_eight_frozen_frames_required")
        metadata = validate_metadata(frame_metadata, len(frames))
        context_frames = context.get("frames", [])
        if len(context_frames) != 8 or any(
            abs(item["timestamp_ms"] - index / metadata["fps"] * 1000) > 1
            for item, index in zip(context_frames, metadata["frames_indices"])
        ):
            raise ValueError("frozen_prompt_and_original_timestamps_disagree")
        arrays = [np.array(frame.convert("RGB"), copy=True) for frame in frames]
        if len({array.shape for array in arrays}) != 1:
            raise ValueError("native_video_frames_require_equal_dimensions")
        native_video = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "video", "video": "<provided-eight-frame-tensor>"},
                {"type": "text", "text": render_user_text(context)},
            ]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], videos=[native_video],
            videos_kwargs={"size": {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS},
                           "do_sample_frames": False, "video_metadata": [metadata]},
            return_tensors="pt",
        )
        # The official processor must not silently alter the 8-frame timeline.
        grid = inputs["video_grid_thw"].tolist()
        if len(grid) != 1 or grid[0][0] != 8:
            raise RuntimeError("processor_resampled_frozen_frames")
        return inputs, {
            "representation": "native_video_tensor", "original_input_frames": 8,
            "do_sample_frames": False, "video_grid_thw": grid,
            "temporal_merge_size": self.processor.video_processor.temporal_merge_size,
            "frame_min_pixels": MIN_PIXELS, "frame_max_pixels": MAX_PIXELS,
            "source_frame_indices": metadata["frames_indices"], "source_fps": metadata["fps"],
            "timestamps_seconds": [index / metadata["fps"] for index in metadata["frames_indices"]],
            "user_text": render_user_text(context),
        }

    def generate(self, *, system_prompt: str, context: dict, frames: list,
                 max_new_tokens: int, temperature: float = 0,
                 frame_metadata: dict) -> dict[str, Any]:
        import torch

        if temperature != 0:
            raise ValueError("frozen_protocol_requires_greedy_decoding")
        if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 8192:
            raise ValueError("invalid_output_budget")
        inputs, preprocessing = self.prepare_inputs(
            system_prompt=system_prompt, context=context, frames=frames, frame_metadata=frame_metadata,
        )
        prompt_tokens = inputs.input_ids.shape[-1]
        if prompt_tokens + max_new_tokens > 32768:
            raise ValueError("operational_32768_token_limit_exceeded")
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        ids = output[0, prompt_tokens:].tolist()
        text = self.processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return {"text": text, "finish_reason": classify_finish(ids, self.model.generation_config.eos_token_id),
                "prompt_tokens": int(prompt_tokens), "completion_tokens": len(ids),
                "preprocessing": preprocessing, "model_revision": MODEL_REVISION}


def load_model(model_path: str) -> Backend:
    return Backend(model_path)


def classify_8frames(backend: Backend, frames: list, prompt: str, *, context: dict,
                     frame_metadata: dict, max_new_tokens: int = 4096) -> dict:
    return backend.generate(system_prompt=prompt, context=context, frames=frames,
                            frame_metadata=frame_metadata, max_new_tokens=max_new_tokens)
