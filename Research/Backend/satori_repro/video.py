"""Extract timestamped JPEG frames from a specified local video, never a camera."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2


def extract_video(video_path: Path, output_dir: Path, *, fps: float = 1.0,
                  max_frames: int = 600) -> Path:
    """Sample the first source frame at/after each scheduled time without duplicates.

    OpenCV source timestamps are preferred. A frame-index/source-FPS fallback is
    explicit in extraction.json for codecs without monotonic source timestamps.
    """
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Sample FPS must be a finite positive number")
    if type(max_frames) is not int or max_frames <= 0:
        raise ValueError("max_frames must be a positive integer")
    try:
        source = Path(video_path).resolve()
        output = Path(output_dir)
    except (TypeError, ValueError):
        raise ValueError("Video input and output must be filesystem paths") from None
    if not source.is_file():
        raise ValueError("Video input must be an existing local file")
    if output.exists():
        raise FileExistsError("Video output directory already exists")

    capture = cv2.VideoCapture(str(source))
    created = False
    frames = []
    metadata = {
        "source": str(source), "sample_fps": float(fps), "max_frames": max_frames,
        "saved_frames": 0, "decoded_frames": 0, "truncated": False,
        "timestamp_fallback_frames": 0, "status": "running",
    }
    try:
        if not capture.isOpened():
            raise ValueError("Video file could not be opened or decoded")
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        valid_source_fps = math.isfinite(source_fps) and source_fps > 0
        metadata["source_fps"] = source_fps if valid_source_fps else None
        output.mkdir(parents=True, exist_ok=False)
        created = True
        next_sample_ms = 0.0
        sample_interval = 1000.0 / fps
        previous_time = -1.0
        index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            metadata["decoded_frames"] += 1
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)
            if not math.isfinite(timestamp) or timestamp < 0 or timestamp <= previous_time:
                if not valid_source_fps:
                    raise ValueError("Video lacks usable timestamps and source FPS")
                timestamp = index * 1000.0 / source_fps
                metadata["timestamp_fallback_frames"] += 1
                if timestamp <= previous_time:
                    raise ValueError("Video source timestamps are not monotonic")
            previous_time = timestamp
            index += 1
            timestamp_ms = int(round(timestamp))
            if timestamp + 1e-6 < next_sample_ms or (frames and timestamp_ms <= frames[-1]["timestamp_ms"]):
                continue
            if len(frames) >= max_frames:
                metadata["truncated"] = True
                break
            frame_id = f"frame_{len(frames):06d}"
            filename = frame_id + ".jpg"
            ok, encoded = cv2.imencode(".jpg", image)
            if not ok:
                raise ValueError("Video frame could not be encoded as JPEG")
            (output / filename).write_bytes(encoded.tobytes())
            frames.append({"frame_id": frame_id, "timestamp_ms": timestamp_ms, "image_path": filename})
            next_sample_ms = (math.floor((timestamp + 1e-6) / sample_interval) + 1) * sample_interval
        if not frames:
            raise ValueError("Video file contains no decodable frames")
        manifest = {"session_id": source.stem, "frames": frames, "events": []}
        manifest_path = output / "session.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        metadata["status"] = "complete"
        return manifest_path
    except cv2.error:
        raise ValueError("Video decoding or frame encoding failed") from None
    finally:
        capture.release()
        if created:
            metadata["saved_frames"] = len(frames)
            if metadata["status"] == "running":
                metadata["status"] = "failed"
            (output / "extraction.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
