"""Real frozen 32-JPEG processor check; no model, CUDA, labels or generation."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from backend32 import Backend, MIN_PIXELS, MAX_PIXELS
from server import input_fingerprint


def main():
    parser = argparse.ArgumentParser()
    for name in ("project-root", "model", "manifest", "taxonomy", "registry", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    sys.path[:0] = [str(args.project_root / "Tools"), str(args.project_root / "Research/Backend"), str(args.project_root)]
    from gt_benchmark import PROMPT
    from PIL import Image
    import torch
    from transformers import AutoProcessor
    assert not torch.cuda.is_initialized()
    manifest = json.loads(args.manifest.read_text("utf-8"))
    taxonomy = json.loads(args.taxonomy.read_text("utf-8"))
    registry = json.loads(args.registry.read_text("utf-8"))["inputs"]
    sample = manifest["samples"][0]
    hashes, frames, times = [], [], []
    for frame in sample["frames"]:
        path = args.manifest.parent / frame["image_path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == frame["sha256"]
        hashes.append(digest)
        times.append(frame["timestamp_ms"])
        with Image.open(path) as image:
            frames.append(image.convert("RGB").copy())
    context = {"scene_candidates": taxonomy["scenes"], "action_candidates": taxonomy["actions"],
               "frames": [{"frame_id": f["frame_id"], "timestamp_ms": f["timestamp_ms"]} for f in sample["frames"]]}
    backend = Backend.__new__(Backend)
    backend.processor = AutoProcessor.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True,
                                                     min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    inputs, preprocessing = backend.prepare_inputs(system_prompt=PROMPT.replace("eight", "thirty-two"),
        context=context, frames=frames, frame_metadata=registry[input_fingerprint(hashes, times)])
    assert preprocessing["video_grid_thw"][0][0] == 32
    assert preprocessing["do_sample_frames"] is False
    assert not torch.cuda.is_initialized()
    value = {"passed": True, "model_loaded": False, "generation_calls": 0, "cuda_initialized": False,
             "unique_input_frames": len(frames), "image_sha256": hashes,
             "prompt_tokens": inputs.input_ids.shape[-1], "preprocessing": preprocessing,
             "shapes": {key: list(value.shape) for key, value in inputs.items() if hasattr(value, "shape")}}
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: value[k] for k in ("passed", "model_loaded", "generation_calls", "cuda_initialized", "unique_input_frames", "prompt_tokens", "shapes")}))


if __name__ == "__main__":
    main()
