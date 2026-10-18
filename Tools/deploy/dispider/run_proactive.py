"""Native Dispider streaming smoke; run only inside a gpurun allocation."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
import types
from pathlib import Path

from backend import Backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="Tell me when the person cleans a surface using a cloth.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite native streaming evidence")
    started = time.perf_counter()
    backend = Backend(args.model)
    import torch
    from dispider.constants import DEFAULT_SILENT_TOKEN
    from dispider.streaming import (DispiderStreamingAdapter,
                                    ReleasedPerceptionDecisionReactionBackend,
                                    stream_video_file)

    generations, windows, answers = [], [], []
    original_generate = backend.model.generate

    def record_generate(owner, *positional, **kwargs):
        del owner
        generated_start = time.perf_counter()
        result = original_generate(*positional, **kwargs)
        sequences = getattr(result, "sequences", result)
        ids = sequences[0].tolist()
        eos = backend.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
        stop_ids = backend.tokenizer("<|im_end|>").input_ids
        ended = bool(ids and ids[-1] in eos_ids)
        if stop_ids and ids[-len(stop_ids):] == stop_ids:
            ended = True
        text = backend.tokenizer.batch_decode(sequences, skip_special_tokens=True)[0].strip()
        raw_text = backend.tokenizer.batch_decode(sequences, skip_special_tokens=False)[0].strip()
        silent = DEFAULT_SILENT_TOKEN in raw_text
        generations.append({
            "generation_seconds": time.perf_counter() - generated_start,
            "completion_tokens": len(ids), "max_new_tokens": kwargs.get("max_new_tokens"),
            "finish_reason": "stop" if ended else "length", "actual_eos_observed": ended,
            "text": text, "raw_text": raw_text, "keep_silent_emitted": silent,
            "effective_answer": bool(text) and not silent and ended,
        })
        return result

    backend.model.generate = types.MethodType(record_generate, backend.model)
    runtime = ReleasedPerceptionDecisionReactionBackend(
        backend.model, backend.tokenizer, backend.image_processor, backend.time_tokenizer,
        generation_kwargs={"max_new_tokens": args.max_new_tokens},
    )

    class RecordedAdapter(DispiderStreamingAdapter):
        def perceive(self, window):
            current = {
                "window_start_s": window.start_s, "window_end_s": window.end_s,
                "timestamp_s": window.timestamp_s, "complete_window": window.complete,
                "sampled_frame_timestamps_s": list(window.timestamps_s),
                "max_timestamp_supplied_to_model_s": max(window.timestamps_s),
                "model_frame_slots": len(window.frames),
                "unique_sampled_timestamps": len(set(window.timestamps_s)),
                "tail_handling": "upstream StreamingSession.flush; timestamps below show any duplicate samples",
                "started_after_seconds": time.perf_counter() - started,
            }
            step = time.perf_counter()
            result = super().perceive(window)
            current["perception_preprocessing_seconds"] = time.perf_counter() - step
            windows.append(current)
            return result

        def should_respond(self, perception, *, window, history):
            step = time.perf_counter()
            answer = super().should_respond(perception, window=window, history=history)
            windows[-1]["decision_seconds"] = time.perf_counter() - step
            windows[-1]["decision"] = dataclasses.asdict(self.decisions[-1])
            return answer

    # Preserve the upstream 16-second interval and 16 uniformly sampled frames.
    # Source decoder visits all source frames; it is not a claimed 2fps input.
    adapter = RecordedAdapter(runtime, args.prompt, enable_decision_kv_cache=True,
                              verify_cache_with_oracle=True)
    session = adapter.new_session(window_seconds=16.0, frames_per_window=16,
                                  max_history_windows=8)
    output = {
        "status": "running", "model": "Dispider", "prompt": args.prompt,
        "source_revision": "803f6b4634b3232293e89a38bcce9f5815c33ed4",
        "model_revision": "7c5517eaaa4dc3788602af40e077063792c75f6c",
        "attention_backend": "sdpa", "official_stack": False,
        "video_sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
        "sampling": {"native_streaming": True, "window_seconds": 16.0,
                     "frames_per_window": 16, "source_chunk_frames": 64,
                     "source_fps_forced": False, "decision_cache": "on",
                     "verify_cache_with_oracle": True},
        "windows": windows, "answers": answers, "generations": generations,
        "limitations": "Deployment smoke without independent event GT; no accuracy claim. Max supplied timestamp describes selected model frames, not CPU decoder read-ahead.",
    }
    try:
        with torch.inference_mode():
            for event in stream_video_file(session, str(args.video), chunk_frames=64):
                generation = generations[-1] if generations else {}
                answers.append({"timestamp_s": event.timestamp_s, "answer": event.answer,
                                "keep_silent_emitted": generation.get("keep_silent_emitted"),
                                "effective_answer": generation.get("effective_answer", False),
                                "emitted_after_seconds": time.perf_counter() - started})
        output["status"] = "completed"
    except Exception as error:
        output["status"] = "failed"
        output["error_type"] = type(error).__name__
        raise
    finally:
        output["elapsed_seconds"] = time.perf_counter() - started
        output["emitted_event_count"] = len(answers)
        output["effective_answer_count"] = sum(item["effective_answer"] for item in answers)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": output["status"], "windows": len(windows),
                          "emitted_events": len(answers), "effective_answers": output["effective_answer_count"],
                          "elapsed_seconds": output["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
