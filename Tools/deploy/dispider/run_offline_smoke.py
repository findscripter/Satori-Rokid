"""One official freeform Dispider VideoStream.run call, without JSON or GT.

Run only after the formal benchmark has ended, in a separate gpurun allocation.
The only generation override is the requested 256-token output budget.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

QUERY = "Describe the visible environment and what the person is doing."
MAX_NEW_TOKENS = 256
HF_REVISION = "7c5517eaaa4dc3788602af40e077063792c75f6c"
SOURCE_REVISION = "803f6b4634b3232293e89a38bcce9f5815c33ed4"
MODEL_ROOT = Path("/dev/shm/gpu-host-satori-dispider-7c5517e")


def source_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def summarize_new_tokens(tokenizer, output, eos_config, stop_ids, silence_token):
    sequences = getattr(output, "sequences", output)
    ids = sequences[0].tolist()
    raw = tokenizer.batch_decode(sequences, skip_special_tokens=False)[0]
    decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]
    eos_ids = set(eos_config if isinstance(eos_config, list) else [eos_config]) if eos_config is not None else set()
    eos_observed = bool(ids and ids[-1] in eos_ids)
    stopping_suffix_observed = bool(stop_ids and ids[-len(stop_ids):] == stop_ids)
    ended = eos_observed or stopping_suffix_observed
    special_ids = set(tokenizer.all_special_ids)
    return {
        "completion_tokens": len(ids), "generated_token_ids": ids,
        "raw_response_with_special_tokens": raw,
        "raw_special_tokens": [{"position": index, "token_id": token,
                                "token": tokenizer.convert_ids_to_tokens(token)}
                               for index, token in enumerate(ids) if token in special_ids],
        "decoded_without_special_tokens": decoded,
        "keep_silent_emitted": silence_token in raw,
        "actual_eos_observed": eos_observed,
        "official_stopping_suffix_observed": stopping_suffix_observed,
        "finish_reason": "stop" if ended else "length",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL_ROOT)
    parser.add_argument("--video", type=Path,
                        default=Path("/home/gpu-user/work/satori-video-eval/smoke/native-smoke.mp4"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite a previous offline smoke")
    allocation = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not os.environ.get("GPURUN_JOB") or not allocation or "," in allocation or allocation == "-1":
        raise SystemExit("Run this single-GPU smoke through gpurun")
    if args.model.absolute() != MODEL_ROOT or args.model.resolve() != MODEL_ROOT:
        raise SystemExit("Use the fixed, verified Dispider RAM snapshot")
    integrity = json.loads((args.model / "restored-integrity.json").read_text())
    if integrity.get("revision") != HF_REVISION or not integrity.get("all_24_files_verified"):
        raise SystemExit("The snapshot has not passed pinned integrity verification")
    project = Path(__file__).resolve().parent
    runtime = Path(os.environ.get("DISPIDER_RUNTIME", str(project / "runtime"))).resolve(strict=True)
    os.environ["DISPIDER_ATTN_IMPLEMENTATION"] = "sdpa"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(runtime))

    started = time.perf_counter()
    sampling, generations = [], []
    result = {
        "status": "running", "model": "Dispider", "model_revision": HF_REVISION,
        "source_revision": SOURCE_REVISION, "runtime": str(runtime),
        "runner_sha256": source_sha256(Path(__file__)),
        "official_inference_file_sha256": source_sha256(runtime / "inference.py"),
        "query": QUERY, "max_new_tokens": MAX_NEW_TOKENS,
        "source_video_sha256": source_sha256(args.video),
        "source_video_bytes": args.video.stat().st_size,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scheduler_job": os.environ["GPURUN_JOB"], "cuda_visible_devices": allocation,
        "entry_point": "official inference.VideoStream.run", "task": "single_freeform_video_question",
        "json_constraint": False, "ground_truth_read": False, "automatic_retry": False,
        "forced_response": False, "silence_threshold_changed": False,
        "checkpoint_weight_changes": False, "sampling": sampling, "generations": generations,
        "limitations": "One non-GT smoke; no accuracy claim or answer selection across retries.",
    }
    original_reader = original_generate = streamer = inference = None
    try:
        import inference
        from dispider.constants import DEFAULT_SILENT_TOKEN

        # Observe actual indices requested by the unchanged official sampler.
        original_reader = inference.VideoReader

        class ObservedVideoReader:
            def __init__(self, *positional, **kwargs):
                self.reader = original_reader(*positional, **kwargs)
                self.info = {"source_frames": len(self.reader),
                             "source_fps": float(self.reader.get_avg_fps()),
                             "source_fps_forced": False, "requests": []}
                sampling.append(self.info)

            def __len__(self):
                return len(self.reader)

            def __getattr__(self, name):
                return getattr(self.reader, name)

            def get_batch(self, indices):
                indices = [int(index) for index in indices]
                timestamps = self.reader.get_frame_timestamp(indices)
                if hasattr(timestamps, "asnumpy"):
                    timestamps = timestamps.asnumpy()
                self.info["requests"].append({
                    "frame_indices": indices, "sampled_frames": len(indices),
                    "unique_frame_indices": len(set(indices)),
                    "frame_timestamp_intervals_s": timestamps.tolist(),
                    "max_supplied_frame_timestamp_s": float(timestamps[:, 0].max()),
                })
                return self.reader.get_batch(indices)

        inference.VideoReader = ObservedVideoReader
        load_started = time.perf_counter()
        streamer = inference.VideoStream(str(args.model))
        result["model_load_seconds"] = time.perf_counter() - load_started
        result["reaction_attention_backend"] = streamer.model.config._attn_implementation
        result["decision_attention_backend"] = (
            streamer.model.get_perception_decision().decision.get_model().config._attn_implementation)
        original_generate = streamer.model.generate

        def observed_generate(owner, *positional, **kwargs):
            del owner
            if generations:
                raise RuntimeError("Only one native generation is permitted")
            original_budget = kwargs.get("max_new_tokens")
            kwargs["max_new_tokens"] = MAX_NEW_TOKENS
            generation_started = time.perf_counter()
            output = original_generate(*positional, **kwargs)
            record = summarize_new_tokens(
                streamer.tokenizer, output, streamer.model.generation_config.eos_token_id,
                streamer.tokenizer("<|im_end|>").input_ids, DEFAULT_SILENT_TOKEN)
            record.update(
                native_default_output_budget=original_budget,
                effective_output_budget=MAX_NEW_TOKENS,
                do_sample=kwargs.get("do_sample"),
                generation_seconds=time.perf_counter() - generation_started,
                image_tensor_shape=list(kwargs["images"].shape),
            )
            generations.append(record)
            return output

        streamer.model.generate = types.MethodType(observed_generate, streamer.model)
        run_started = time.perf_counter()
        visible = streamer.run(str(args.video), QUERY)
        result["official_run_seconds"] = time.perf_counter() - run_started
        if len(generations) != 1:
            raise RuntimeError("Official run did not produce exactly one generation")
        result["visible_answer"] = visible
        result["effective_answer"] = (bool(visible.strip()) and not generations[0]["keep_silent_emitted"]
                                      and generations[0]["finish_reason"] == "stop")
        result["status"] = "completed"
    except Exception as error:
        result["status"], result["error_type"] = "failed", type(error).__name__
        raise
    finally:
        if original_reader is not None:
            inference.VideoReader = original_reader
        if streamer is not None and original_generate is not None:
            streamer.model.generate = original_generate
        result["elapsed_seconds"] = time.perf_counter() - started
        result["generation_calls_completed"] = len(generations)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": result["status"], "generation_calls_completed": len(generations),
                          "effective_answer": result.get("effective_answer", False),
                          "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
