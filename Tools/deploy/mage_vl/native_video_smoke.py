"""One-process, one-GPU native MAGE-VL video smoke test; no HTTP server.

Run inside the GPU allocation, with a complete local model snapshot. Heavy
imports are lazy. The revision is a declaration, not a verification of weights.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

MIN_PIXELS = 3136
MAX_PIXELS = 150000
CODEC_CONFIG = {"engine": "hevc", "target_canvas": 8, "patch": 16}


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--backend", choices=("frames", "codec"), default="frames")
    parser.add_argument("--num-frames", type=positive_int, default=8)
    parser.add_argument("--max-new-tokens", type=positive_int, default=256)
    parser.add_argument("--question", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--revision", required=True, help="Declared snapshot revision; not verified by this script")
    args = parser.parse_args(argv)
    if not args.question.strip() or not args.revision.strip():
        parser.error("question and revision must not be blank")
    return args


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_pixels(processor):
    """Preserve Transformers 5.x SizeDict's dataclass type when present."""
    size = processor.image_processor.size
    if dataclasses.is_dataclass(size) and not isinstance(size, type):
        size = dataclasses.replace(size, shortest_edge=MIN_PIXELS, longest_edge=MAX_PIXELS)
    elif isinstance(size, Mapping):
        size = {**size, "shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}
    else:
        raise TypeError("Unsupported image processor size representation")
    processor.image_processor.size = size
    processor.video_processor.min_pixels = MIN_PIXELS
    processor.video_processor.max_pixels = MAX_PIXELS
    return dataclasses.asdict(size) if dataclasses.is_dataclass(size) else dict(size)


def finish_reason(generated_ids, eos_ids, max_new_tokens):
    # HF generate has no OpenAI-style finish_reason. Infer stop only from the
    # final generated token matching the EOS configuration used by generation.
    if generated_ids and int(generated_ids[-1]) in eos_ids:
        return "stop"
    if len(generated_ids) >= max_new_tokens:
        return "length"
    return "unknown"


def eos_configuration(model, tokenizer):
    candidates = (
        ("generation_config", getattr(model, "generation_config", None)),
        ("model_config", getattr(model, "config", None)),
        ("text_config", getattr(getattr(model, "config", None), "text_config", None)),
        ("tokenizer", tokenizer),
    )
    for name, config in candidates:
        value = getattr(config, "eos_token_id", None)
        if value is not None:
            if isinstance(value, int):
                value = [value]
            ids = {int(item) for item in value}
            if ids:
                return ids, name
    raise ValueError("No EOS configuration is available; normal stopping cannot be verified")


def input_metadata(inputs, tokenizer, torch):
    result = {"tensor_shapes": {}, "time_grid": {}}
    for name, value in inputs.items():
        if torch.is_tensor(value):
            result["tensor_shapes"][name] = {
                "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)
            }
    for name in ("image_grid_thw", "video_grid_thw", "second_per_grid_ts", "frame_timestamps", "frame_indices"):
        if name in inputs:
            value = inputs[name]
            result["time_grid"][name] = value.detach().cpu().tolist() if torch.is_tensor(value) else value
    positions = inputs.get("patch_positions")
    if positions is not None and torch.is_tensor(positions) and positions.ndim == 2 and positions.shape[-1] >= 1:
        result["time_grid"]["patch_time_coordinates"] = positions[:, 0].detach().cpu().unique(sorted=True).tolist()
        result["time_grid"]["patch_time_coordinate_unit"] = "native processor coordinates; not converted to seconds"
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    result["time_grid"]["encoded_timestamp_tags_seconds"] = [
        float(value) for value in re.findall(r"<([0-9]+(?:\.[0-9]+)?) seconds>", prompt_text)
    ]
    result["time_grid"]["time_basis"] = "original video path decoded by the native processor; no synthetic frame numbering or FPS override"
    result["time_grid"]["timestamp_tag_precision"] = "processor text tags rounded to one decimal; raw patch coordinates also retained"
    return result


def infer(args, report):
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    report["versions"] = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__}
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Run inside a gpurun allocation exposing exactly one CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The allocated GPU does not support BF16")
    report["gpu"] = {"logical_device": 0, "visible_count": 1, "name": torch.cuda.get_device_name(0)}
    torch.cuda.reset_peak_memory_stats(0)
    try:
        started = time.perf_counter()
        report["phase"] = "load_processor"
        # The official processor's internal AutoTokenizer call drops the trust
        # flag. Register its local config class first to avoid an interactive prompt.
        config_class = get_class_from_dynamic_module(
            "configuration_mage_vl.MageVLConfig", str(args.model_dir), local_files_only=True
        )
        AutoConfig.register("mage_vl", config_class, exist_ok=True)
        processor = AutoProcessor.from_pretrained(str(args.model_dir), trust_remote_code=True, local_files_only=True)
        report["effective_image_processor_size"] = configure_pixels(processor)
        report["effective_video_processor_pixels"] = {"min_pixels": processor.video_processor.min_pixels, "max_pixels": processor.video_processor.max_pixels}
        report["phase"] = "load_model"
        model = AutoModelForImageTextToText.from_pretrained(
            str(args.model_dir), trust_remote_code=True, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0},
        )
        model.eval()
        torch.cuda.synchronize(0)
        report["timing_seconds"]["load_processor_and_model"] = time.perf_counter() - started

        report["phase"] = "native_video_preprocess"
        started = time.perf_counter()
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": args.question}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[prompt], videos=[str(args.video)], video_backend=args.backend,
            num_frames=args.num_frames, max_pixels=MAX_PIXELS,
            codec_config=dict(CODEC_CONFIG), return_tensors="pt",
        )
        report.update(input_metadata(inputs, processor.tokenizer, torch))
        if len(inputs["input_ids"].shape) != 2 or inputs["input_ids"].shape[0] != 1:
            raise ValueError("Native smoke test requires a single input sequence")
        input_count = int(inputs["input_ids"].shape[1])
        report["input_tokens"] = input_count
        if "pixel_values" not in inputs or "patch_positions" not in inputs:
            raise ValueError("Native video processor did not provide visual input tensors")
        inputs = {
            name: value.to(device="cuda:0", dtype=torch.bfloat16) if torch.is_tensor(value) and value.is_floating_point()
            else value.to(device="cuda:0") if torch.is_tensor(value) else value
            for name, value in inputs.items()
        }
        torch.cuda.synchronize(0)
        report["timing_seconds"]["native_preprocess_and_transfer"] = time.perf_counter() - started

        report["phase"] = "generate"
        eos_ids, eos_source = eos_configuration(model, processor.tokenizer)
        report["eos_token_ids"] = sorted(eos_ids)
        report["eos_configuration_source"] = eos_source
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                eos_token_id=sorted(eos_ids), return_dict_in_generate=True,
            )
        torch.cuda.synchronize(0)
        report["timing_seconds"]["generation"] = time.perf_counter() - started
        sequence = output.sequences[0]
        generated = sequence[input_count:].detach().cpu().tolist()
        answer = processor.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        raw = processor.tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        report["phase"] = "generation_completed"
        return {
            "output_tokens": len(generated), "total_tokens": int(sequence.shape[0]),
            "finish_reason": finish_reason(generated, eos_ids, args.max_new_tokens),
            "finish_reason_source": "terminal generated EOS token, not an HTTP completion field",
            "terminal_token_id": int(generated[-1]) if generated else None,
            "answer": answer, "raw_answer": raw,
        }
    finally:
        report["max_gpu_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(0))
        report["max_gpu_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(0))


def write_report(path, report):
    path.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def run_smoke(args):
    started = time.perf_counter()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError("Output already exists; choose a new evidence filename")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running", "phase": "validate_local_inputs", "finish_reason": "error",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend, "num_frames_requested": args.num_frames,
        "max_new_tokens": args.max_new_tokens, "question": args.question,
        "model_revision_declared": args.revision, "revision_verified_by_this_script": False,
        "codec_config": dict(CODEC_CONFIG) if args.backend == "codec" else None,
        "sampling_note": "Codec uses target_canvas=8; num_frames is a frames-backend sampling request, not a promise about codec canvas count.",
        "dtype": "bfloat16", "attention_implementation": "sdpa", "device_map": {"": 0},
        "timing_seconds": {}, "input_tokens": None, "output_tokens": None,
        "answer": None, "raw_answer": None,
        "max_gpu_memory_allocated_bytes": None, "max_gpu_memory_reserved_bytes": None,
        "cleanup_policy": "single process exits after writing result; no persistent model or HTTP service",
        "evaluation_scope": "native video inference smoke test only; no accuracy or task-success metric",
    }
    # Reserve a new result path without replacing previous experiment evidence.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
    try:
        args.model_dir = args.model_dir.resolve(strict=True)
        args.video = args.video.resolve(strict=True)
        if not args.model_dir.is_dir() or not (args.model_dir / "config.json").is_file():
            raise ValueError("model-dir must be a complete local model directory")
        if not args.video.is_file() or args.video.stat().st_size == 0:
            raise ValueError("video must be a nonempty local video file")
        report.update(model_dir=str(args.model_dir), video=str(args.video), video_sha256=sha256_file(args.video), video_bytes=args.video.stat().st_size)
        report["local_model_source_sha256"] = {
            name: sha256_file(args.model_dir / name)
            for name in ("config.json", "configuration_mage_vl.py", "modeling_mage_vl.py", "processing_mage_vl.py", "video_processing_mage_vl.py", "codec_video_processing_mage_vl.py")
            if (args.model_dir / name).is_file()
        }
        write_report(args.output, report)
        report.update(infer(args, report))
        report["status"] = "success" if report["finish_reason"] == "stop" and bool((report.get("answer") or "").strip()) else "failed"
        if report["status"] == "failed":
            report["failure"] = "Generation did not finish with a verified stop and nonempty answer"
    except Exception as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        # Exception bodies can include external library request details. Do not
        # copy them, environment variables, credentials, or traceback locals.
        report["failure"] = "Native inference failed; see phase and error_type"
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["timing_seconds"]["total"] = time.perf_counter() - started
        write_report(args.output, report)
    return report


def main(argv=None):
    args = parse_args(argv)
    try:
        report = run_smoke(args)
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "failure": "Could not create the requested new output file"}))
        return 1
    print(json.dumps({"status": report["status"], "finish_reason": report["finish_reason"], "output": str(args.output), "phase": report["phase"]}, ensure_ascii=True))
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
