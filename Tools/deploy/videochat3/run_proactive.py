"""Run the fixed official proactive StreamingSession and save every emitted state.

GPU invocation is the coordinator's responsibility (gpurun only).
This is a functional online-protocol smoke, not an action or timing benchmark.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import time

from backend import Backend, classify_finish, require_gpu_allocation


def resize_official(frame, max_pixels):
    """Same force-resize rule as fixed official demo_vc3_proactive.py."""
    width, height = frame.size
    scale = math.sqrt(height * width / max_pixels)
    height = max(28, math.floor(height / scale / 28) * 28)
    width = max(28, math.floor(width / scale / 28) * 28)
    return frame.resize((width, height))


def main():
    require_gpu_allocation()
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--question", default="Where is the wearer and what are they doing? Respond when you have enough visual evidence.")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("new_result_path_required")
    target.parent.mkdir(parents=True, exist_ok=True)
    source = Path(args.model) / "inference_fast_vc3.py"
    spec = importlib.util.spec_from_file_location("official_vc3_stream", source)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    backend = Backend(args.model)
    records = []

    class Engine:
        def infer(self, messages, max_tokens=args.max_new_tokens, **_):
            start = time.monotonic()
            inputs = backend.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(backend.model.device).to(backend.model.dtype)
            prompt_tokens = inputs.input_ids.shape[-1]
            with torch.inference_mode():
                generated = backend.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            ids = generated[0, prompt_tokens:].tolist()
            raw = backend.processor.batch_decode([ids], skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
            text = official.VideoChat3StreamEngine._strip_end_tokens(raw)
            records.append({"raw_decoded": raw, "text": text,
                            "finish_reason": classify_finish(ids, backend.model.generation_config.eos_token_id),
                            "prompt_tokens": int(prompt_tokens), "completion_tokens": len(ids),
                            "generation_seconds": time.monotonic() - start})
            return text

    session = official.StreamingSession(Engine(), system=official.SYSTEM, question=args.question,
                                        question_time=0, max_rounds=16, max_tokens=args.max_new_tokens)
    extractor = official.VideoFrameExtractor(args.video, target_fps=args.fps)
    actual_fps = extractor.actual_fps
    frames_per_round = max(1, round(actual_fps))
    standby = False
    try:
        for round_idx, frame_start in enumerate(range(0, extractor.get_total_rounds(), frames_per_round)):
            if round_idx >= args.max_rounds:
                break
            frame_end = min(frame_start + frames_per_round, extractor.get_total_rounds())
            frames = [extractor.get_frame_at_round(i) for i in range(frame_start, frame_end)]
            # Official demo boosts each next round after Standby from 224² to 448².
            max_pixels = 448 * 448 if standby else 224 * 224
            frames = [resize_official(frame, max_pixels) for frame in frames]
            response = session.step(frames, round_idx=round_idx, frame_max_pixels=max_pixels,
                                    time_start=frame_start / actual_fps, time_end=frame_end / actual_fps)
            standby = "</Standby>" in response
            records[-1].update({"round_index": round_idx, "frame_start": frame_start,
                                "frame_end": frame_end, "frame_max_pixels": max_pixels,
                                "time_start": frame_start / actual_fps, "time_end": frame_end / actual_fps})
            target.write_text(json.dumps({"model": backend.model_id,
                              "question": args.question, "official_protocol": True,
                              "target_fps": args.fps, "actual_fps": actual_fps,
                              "scored": False, "rounds": records}, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(records[-1], ensure_ascii=False), flush=True)
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
