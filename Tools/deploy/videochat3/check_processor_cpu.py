"""Load only the official processor and model class; never instantiate GPU model."""
import argparse
import json
from pathlib import Path

from backend import Backend, MAX_PIXELS, MIN_PIXELS


def main():
    from PIL import Image
    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model_class = get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], args.model, local_files_only=True)
    config.vision_config.attn_impl = "sdpa"
    config._attn_implementation = "sdpa"
    # Structural instantiation allocates no tensor data and never uses CUDA.
    with torch.device("meta"):
        meta_model = model_class(config)
    parameter_count = sum(parameter.numel() for parameter in meta_model.parameters())
    del meta_model
    instance = Backend.__new__(Backend)
    instance.processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
        min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
    )
    # Unique colors reveal accidental repeats/reordering; original metadata stays fixed.
    frames = [Image.new("RGB", (320, 240), (i * 31, 255 - i * 31, i * 7)) for i in range(8)]
    context = {"frames": [{"frame_id": f"v{i:03d}", "timestamp_ms": i * 500} for i in range(8)]}
    metadata = {"fps": 30.0, "total_num_frames": 120, "duration": 4.0,
                "frames_indices": [0, 15, 30, 45, 60, 75, 90, 105], "width": 320, "height": 240,
                "video_backend": "frozen-jpeg"}
    inputs, preprocessing = instance.prepare_inputs(system_prompt="Describe this clip.", context=context,
                                                    frames=frames, frame_metadata=metadata)
    result = {"passed": True, "device": "cpu", "transformers": transformers.__version__,
              "torch": torch.__version__, "model_class_imported": model_class.__name__,
              "meta_model_constructed": True, "parameter_count": parameter_count,
              "preprocessing": preprocessing, "input_ids_tokens": inputs.input_ids.shape[-1],
              "shapes": {key: list(value.shape) for key, value in inputs.items() if hasattr(value, "shape")}}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
