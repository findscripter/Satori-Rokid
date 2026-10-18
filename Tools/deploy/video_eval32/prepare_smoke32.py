"""Freeze 32 unique JPEGs from the existing non-evaluation EgoDex video."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import cv2
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError("cannot open existing non-evaluation video")
    try:
        count, fps = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), float(capture.get(cv2.CAP_PROP_FPS))
        if count < 32 or fps <= 0:
            raise ValueError("requires at least 32 actual video frames")
        indices = [int((i+0.5)*count/32) for i in range(32)]
        if len(set(indices)) != 32:
            raise ValueError("requires unique native frame indices")
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "frames").mkdir()
        frames = []
        for i, index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if not ok:
                raise ValueError("source frame decode failed")
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise ValueError("JPEG encode failed")
            raw = encoded.tobytes()
            relative = f"frames/v{i:03d}.jpg"
            (args.output / relative).write_bytes(raw)
            frames.append({"frame_id": f"v{i:03d}", "timestamp_ms": round(index/fps*1000),
                           "image_path": relative, "sha256": hashlib.sha256(raw).hexdigest()})
        value = {"schema_version": 1, "benchmark_id": "egodex-32-schema-smoke-not-charades-evaluation",
                 "samples": [{"sample_id": "e000", "frames": frames}]}
        (args.output / "manifest.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        (args.output / "decoder-facts.json").write_text(json.dumps({"source_video_sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
            "decoded_frame_count": count, "decoder_fps": fps, "source_indices": indices,
            "sampling": "32 uniform time-bin midpoint source indices; non-evaluation source only",
            "source_ids_or_gt_emitted": False, "jpeg_quality": 90}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"prepared": True, "unique_frames": 32, "source_fps": fps, "source_frame_count": count,
                          "manifest_sha256": hashlib.sha256((args.output / "manifest.json").read_bytes()).hexdigest()}))
    finally:
        capture.release()


if __name__ == "__main__":
    main()
