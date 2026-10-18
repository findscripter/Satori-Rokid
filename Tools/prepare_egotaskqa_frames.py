"""Attach verified public EgoTaskQA clips to the already-frozen 200 questions.

No inference and no answer-dependent selection. Four uniformly spaced mid-bin
frames are extracted from each complete, pre-cropped clip. Missing assets keep
the original selection pending; an executable manifest appears only when all
questions have verified media and real frames.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from prepare_egotaskqa_satori import safe_video_name, save_json


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def midpoint_indices(frame_count: int, count: int = 4) -> list[int]:
    if type(frame_count) is not int or frame_count < count or count not in (4, 8):
        raise ValueError("Insufficient real frames or unsupported frame count")
    return [math.floor((index + 0.5) * frame_count / count) for index in range(count)]


def verified_video(data_root: Path, interval: str, receipt: dict) -> Path:
    if receipt.get("source_interval") != interval:
        raise ValueError("Media receipt interval differs from frozen source")
    if receipt.get("archive_member") != "qa_videos/" + interval + ".mp4":
        raise ValueError("Archive member identity differs from original interval")
    target = (data_root / receipt["path"]).resolve()
    if not target.is_relative_to(data_root.resolve()):
        raise ValueError("Media receipt path escapes the data directory")
    if target.name != safe_video_name(interval):
        raise ValueError("Windows-safe filename mapping disagrees with source")
    if target.stat().st_size != receipt["bytes"] or file_sha(target) != receipt["sha256"]:
        raise ValueError("Media size or SHA256 does not match download receipt")
    return target


def extract_clip(video: Path, directory: Path, interval: str,
                 video_hash: str, number: int = 4) -> dict:
    import cv2

    proof_path = directory / "proof.json"
    if proof_path.exists():
        old = json.loads(proof_path.read_text(encoding="utf-8"))
        if old.get("video_sha256") != video_hash or old.get("source_interval") != interval:
            raise ValueError("Existing frame proof belongs to different media")
        if len(old.get("frames", [])) != number:
            raise ValueError("Existing frame policy differs")
        for row in old["frames"]:
            if file_sha(directory / row["filename"]) != row["sha256"]:
                raise ValueError("A previously frozen frame changed")
        return old
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Partial frame directory requires inspection before continuing")
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError("Cannot decode verified media")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid decoder FPS")
        indices = midpoint_indices(frame_count, number)
        directory.mkdir(parents=True, exist_ok=True)
        proof = {"source_interval": interval, "video_sha256": video_hash,
                 "decoded_frame_count": frame_count, "decoder_fps": fps,
                 "duration_seconds": frame_count / fps,
                 "source_interval_span": int(interval.split("|")[-1]) - int(interval.split("|")[-2]),
                 "frame_policy": f"floor((i+0.5)*decoded_frame_count/{number}), local zero-based clip frames",
                 "cutoff_ms": round((frame_count - 1) / fps * 1000),
                 "jpeg_quality": 90, "resize": None, "frames": []}
        proof["frame_count_equals_inclusive_interval_span"] = frame_count == proof["source_interval_span"] + 1
        for index, source_index in enumerate(indices):
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, source_index):
                raise ValueError("Decoder cannot seek to the requested frame")
            ok, image = capture.read()
            if not ok or image is None:
                raise ValueError("Decoder failed to return a requested frame")
            actual_index = capture.get(cv2.CAP_PROP_POS_FRAMES) - 1
            if abs(actual_index - source_index) > 0.1:
                raise ValueError("Decoder frame position differs from requested index")
            ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise ValueError("Cannot encode selected frame")
            filename = f"{index:03d}.jpg"
            raw = data.tobytes()
            (directory / filename).write_bytes(raw)
            proof["frames"].append({
                "frame_id": f"v{index:03d}", "filename": filename,
                "source_frame_index_zero_based": source_index,
                "decoder_reported_frame_index": actual_index,
                "timestamp_ms": round(source_index / fps * 1000),
                "decoder_pos_msec": capture.get(cv2.CAP_PROP_POS_MSEC),
                "width": int(image.shape[1]), "height": int(image.shape[0]),
                "sha256": hashlib.sha256(raw).hexdigest(),
            })
        save_json(proof_path, proof)
        return proof
    finally:
        capture.release()


def prepare(frozen_root: Path, receipt_path: Path) -> dict:
    # Import only once package paths are configured; no provider is constructed.
    from satori_paper_benchmark import validate_manifest, validate_taxonomy

    frozen_root = frozen_root.resolve()
    data_root = frozen_root.parent
    if (frozen_root / "infer-manifest.json").exists():
        raise FileExistsError("Inference inputs already frozen; refusing to replace them")
    pending = json.loads((frozen_root / "pending-media-manifest.json").read_text(encoding="utf-8"))
    selection = json.loads((frozen_root / "selection.json").read_text(encoding="utf-8"))
    requirements = json.loads((frozen_root / "media-requirements.json").read_text(encoding="utf-8"))
    gt = json.loads((frozen_root / "gt.json").read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    source = receipt.get("source", {})
    if (source.get("archive_verified") is not True
            or not isinstance(source.get("lfs_sha256"), str)
            or len(source["lfs_sha256"]) != 64
            or source.get("archive_sha256") != source["lfs_sha256"]):
        raise ValueError("Public archive has not completed its upstream LFS SHA256 verification")
    videos = receipt["videos"]
    by_interval = {row["source_interval"]: row for row in videos}
    if len(by_interval) != len(videos):
        raise ValueError("Duplicate media receipt source intervals")
    count = selection["fixed_denominator"]
    if len(pending["samples"]) != count or len(requirements) != count or len(gt["samples"]) != count:
        raise ValueError("Frozen sample denominator changed")
    for index, (sample, media, label) in enumerate(zip(pending["samples"], requirements, gt["samples"])):
        sid = f"s{index:03d}"
        if any(row["sample_id"] != sid for row in (sample, media, label)):
            raise ValueError("Frozen anonymous sample order changed")
        if label["metadata"]["question_id"] != selection["question_ids"][index]:
            raise ValueError("GT question mapping differs from frozen selection")
        if media["source_interval"] not in by_interval:
            raise ValueError("A frozen video is missing; selection stays pending")
    # Validate every receipt before extracting even the first frame.
    paths = {interval: verified_video(data_root, interval, row) for interval, row in by_interval.items()}
    manifest = {"schema_version": 1, "benchmark_id": pending["benchmark_id"], "samples": []}
    taxonomy = {"schema_version": 1, "goals": [], "actions": []}
    proofs = []
    for sample, media in zip(pending["samples"], requirements):
        interval = media["source_interval"]
        directory = frozen_root / "frames" / sample["sample_id"]
        proof = extract_clip(paths[interval], directory, interval, by_interval[interval]["sha256"])
        frames = [{"frame_id": row["frame_id"], "timestamp_ms": row["timestamp_ms"],
                   "sha256": row["sha256"],
                   "image_path": (directory / row["filename"]).relative_to(frozen_root).as_posix()}
                  for row in proof["frames"]]
        manifest["samples"].append({
            "sample_id": sample["sample_id"], "task_kind": "video_qa",
            "question": sample["question"], "cutoff_ms": proof["cutoff_ms"],
            "candidate_group": None, "frames": frames,
        })
        proofs.append({"sample_id": sample["sample_id"], **proof})
    validate_manifest(manifest, taxonomy)
    validate_taxonomy(taxonomy)
    save_json(frozen_root / "frame-provenance.json", {"samples": proofs})
    save_json(frozen_root / "infer-manifest.json", manifest)
    save_json(frozen_root / "taxonomy.json", taxonomy)
    protocol = {
        "schema_version": 1, "benchmark_id": pending["benchmark_id"], "seed": selection["selection_seed"],
        "source_manifest_sha256": file_sha(frozen_root / "infer-manifest.json"),
        "source_taxonomy_sha256": file_sha(frozen_root / "taxonomy.json"),
        "source_gt_sha256": file_sha(frozen_root / "gt.json"),
        "source_selection_sha256": file_sha(frozen_root / "selection.json"),
        "source_media_provenance_sha256": file_sha(receipt_path),
        "source_frame_provenance_sha256": file_sha(frozen_root / "frame-provenance.json"),
        "sampling": "200 frozen seeded original indirect-test questions; 189 unique pre-cropped clips; no media- or answer-dependent replacement.",
        "observation_policy": "Four local zero-based midpoint bins over each complete clip: floor((i+0.5)*N/4). Original resolution, JPEG90, no audio or IMU. Timestamps use decoded frame index / decoder-reported FPS; cutoff is the final clip frame.",
        "source_media": deepcopy(receipt.get("source", {})),
        "annotation_indexing": "Pre-cropped author-release clip names match exact source interval. No source recropping is performed; each clip has an audited count vs inclusive interval span. QA references the complete original clip.",
        "gt_isolation": "Model sees question and four anonymous image/timestamp pairs only. Original question IDs, categories, source intervals, labels and programs stay in GT/provenance.",
        "metric": "Raw and declared lexical-normalized exact match; not learned-answer-vocabulary classification, an LLM judge, or a synonym-based semantic score.",
        "outcome_policy": "Exactly one attempt per fixed question; errors, truncated replies and abstentions remain in the 200 denominator. Missing media prevents execution; never replace a question.",
        "paper_comparison_limit": "New fixed sample from original indirect split; Satori authors' 200 IDs, exact original observation protocol, ensemble prompts and scoring scripts were not released. This is a zero-shot sparse-frame QA module probe, not the complete Satori system or its user study.",
    }
    save_json(frozen_root / "protocol.json", protocol)
    registry = {
        "benchmark_id": manifest["benchmark_id"],
        "samples": {
            row["sample_id"]: {
                "fps": row["decoder_fps"],
                "total_num_frames": row["decoded_frame_count"],
                "duration": row["duration_seconds"],
                "width": row["frames"][0]["width"],
                "height": row["frames"][0]["height"],
                "frames_indices": [frame["source_frame_index_zero_based"] for frame in row["frames"]],
                "video_backend": "opencv",
            }
            for row in proofs
        },
    }
    registry_path = frozen_root / "native-metadata-registry-20261017.json"
    from deploy.paper_videochat3.provider import validate_registry
    validate_registry(registry, manifest)
    save_json(registry_path, registry)
    native_protocol = {
        **protocol,
        "execution_model": "VideoChat3-4B",
        "model_revision": "9f6107bdfee9686dfe493444022f5cf79a1a6acf",
        "model_selection": "The configured GPT API returned HTTP503 on three distinct non-GT route probes. Select the already deployed VideoChat3 before inspecting any EgoTaskQA prediction; use the same model as the GTEA module probe.",
        "module_scope": "Frozen unified VLM module baseline; not the five-stage Satori system or end-to-end user assistance.",
        "generation_parameters": {"max_tokens": 4096, "temperature": 0.0},
        "decoding": "native greedy; exactly four frozen unique images; no frame duplication, resampling, retries or answer repair",
        "technical_metadata_registry_sha256": file_sha(registry_path),
        "technical_metadata_scope": "Only clip-local original fps/count/dimensions, duration and observed frame indices. No source IDs, question categories, annotations or answers. Original duration is offline metadata, not an online stream claim.",
        "frame_representation": "Native video tensor with official VideoChat3 processor, BF16/SDPA; temporal_merge_size=4. This differs from GPT multi-image transport.",
        "execution_timeout": "No HTTP timeout for synchronous local generation; bounded by max_new_tokens. Do not compare latency with old API runs.",
    }
    prior = frozen_root / "train-mode-prior-20261017.json"
    if prior.exists():
        native_protocol["training_answer_prior_sha256"] = file_sha(prior)
        native_protocol["baseline_policy"] = "Constant answer chosen only from original indirect train 20,303 raw answer strings, maximum count then lexicographic tie-break. Prediction is frozen separately before test scoring; evaluate strict and identical lexical-normalized EM."
    save_json(frozen_root / "protocol-videochat3-20261017.json", native_protocol)
    result = {"status": "prepared", "questions": count, "unique_videos": len({row["source_interval"] for row in requirements}),
              "frames": sum(len(row["frames"]) for row in manifest["samples"]),
              "frame_count_inclusive_span_matches": sum(row["frame_count_equals_inclusive_interval_span"] for row in proofs),
              "model_calls": 0, "manifest_sha256": protocol["source_manifest_sha256"],
              "protocol_sha256": file_sha(frozen_root / "protocol.json")}
    save_json(frozen_root / "frame-preparation-result.json", result)
    summary_path = frozen_root / "preparation-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(status="prepared_for_inference", media_ready_questions=count, no_model_calls=True)
    save_json(summary_path, summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-root", type=Path, default=Path("Research/Data/egotaskqa-satori/frozen-original-indirect-200"))
    parser.add_argument("--media-provenance", type=Path, default=Path("Research/Data/egotaskqa-satori/media-provenance.json"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.frozen_root, args.media_provenance), indent=2))


if __name__ == "__main__":
    main()
