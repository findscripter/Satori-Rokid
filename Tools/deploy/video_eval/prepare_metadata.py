"""Extract only anonymous decoder facts from provenance; no GT input exists."""

import argparse
import json
import math
from pathlib import Path

from server import compact, input_fingerprint, sha, strict_json


def prepare(manifest_path, provenance_path):
    manifest_path, provenance_path = Path(manifest_path), Path(provenance_path)
    manifest_raw, provenance_raw = manifest_path.read_bytes(), provenance_path.read_bytes()
    manifest, provenance = strict_json(manifest_raw), strict_json(provenance_raw)
    by_id = {row["sample_id"]: row for row in provenance["samples"]}
    if len(by_id) != len(provenance["samples"]):
        raise ValueError("duplicate provenance sample")
    inputs = {}
    for sample in manifest["samples"]:
        if set(sample) != {"sample_id", "frames"} or len(sample["frames"]) != 8:
            raise ValueError("requires eight anonymous frames")
        source = by_id[sample["sample_id"]]
        fps, count = source["decoder_fps"], source["decoded_frame_count"]
        if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0 or type(count) is not int or count <= 0:
            raise ValueError("invalid true decoder facts")
        compact_metadata = "source_frame_indices" in source
        if compact_metadata:
            from PIL import Image
            if source["image_sha256"] != [f["sha256"] for f in sample["frames"]] or source["timestamps_ms"] != [f["timestamp_ms"] for f in sample["frames"]]:
                raise ValueError("anonymous decoder table differs from manifest")
            if len(source["source_frame_indices"]) != 8:
                raise ValueError("anonymous decoder table requires eight source indices")
            facts = {}
            for frame, index in zip(sample["frames"], source["source_frame_indices"]):
                with Image.open(manifest_path.parent / frame["image_path"]) as image:
                    facts[frame["frame_id"]] = {"source_frame_index_zero_based": index,
                        "sha256": frame["sha256"], "width": image.width, "height": image.height}
        else:
            facts = {row["frame_id"]: row for row in source["frames"]}
        hashes, times, indices, sizes = [], [], [], set()
        for i, frame in enumerate(sample["frames"]):
            if set(frame) != {"frame_id", "timestamp_ms", "image_path", "sha256"} or frame["frame_id"] != f"v{i:03d}":
                raise ValueError("input contains unexpected metadata")
            image = (manifest_path.parent / frame["image_path"]).resolve()
            if sha(image.read_bytes()) != frame["sha256"]:
                raise ValueError("frozen JPEG hash differs")
            fact = facts[frame["frame_id"]]
            index = fact["source_frame_index_zero_based"]
            if type(index) is not int or not 0 <= index < count or fact["sha256"] != frame["sha256"]:
                raise ValueError("source-frame provenance differs")
            if round(index / fps * 1000) != frame["timestamp_ms"]:
                raise ValueError("true source indices disagree with frozen timestamps")
            hashes.append(frame["sha256"])
            times.append(frame["timestamp_ms"])
            indices.append(index)
            sizes.add((fact["width"], fact["height"]))
        if len(sizes) != 1 or indices != sorted(set(indices)):
            raise ValueError("frame sizes/order are inconsistent")
        width, height = sizes.pop()
        key = input_fingerprint(hashes, times)
        metadata = {"fps": fps, "total_num_frames": count, "duration": count / fps,
                    "frames_indices": indices, "width": width, "height": height,
                    "video_backend": "frozen-jpeg"}
        if key in inputs:
            raise ValueError("duplicate frozen image sequence")
        inputs[key] = metadata
    return {"schema_version": 1, "inputs": inputs,
            "source_manifest_sha256": sha(manifest_raw), "source_provenance_sha256": sha(provenance_raw),
            "policy": "Only decoder fps/count/dimensions and source-frame indices are retained. No video identifiers, GT, scene labels, scripts, objects or descriptions are emitted."}


def prepare_smoke(manifest_path, video_path):
    """Read non-evaluation video decoder facts only; no video frames reach model."""
    import cv2
    from PIL import Image
    manifest_path, video_path = Path(manifest_path), Path(video_path)
    raw = manifest_path.read_bytes()
    manifest = strict_json(raw)
    if len(manifest["samples"]) != 1 or len(manifest["samples"][0]["frames"]) != 8:
        raise ValueError("smoke requires one eight-JPEG sample")
    decoder = cv2.VideoCapture(str(video_path))
    if not decoder.isOpened():
        raise ValueError("cannot open non-evaluation source")
    try:
        fps, count = float(decoder.get(cv2.CAP_PROP_FPS)), int(decoder.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        decoder.release()
    if not math.isfinite(fps) or fps <= 0 or count <= 0:
        raise ValueError("invalid decoder facts")
    hashes, times, indices, sizes = [], [], [], set()
    for frame in manifest["samples"][0]["frames"]:
        image_path = manifest_path.parent / frame["image_path"]
        if sha(image_path.read_bytes()) != frame["sha256"]:
            raise ValueError("frozen smoke JPEG hash differs")
        index = round(frame["timestamp_ms"] * fps / 1000)
        if not 0 <= index < count or abs(index/fps*1000-frame["timestamp_ms"]) > 1:
            raise ValueError("smoke timestamps do not map to original frame times")
        with Image.open(image_path) as image:
            sizes.add((image.width, image.height))
        hashes.append(frame["sha256"])
        times.append(frame["timestamp_ms"])
        indices.append(index)
    if len(sizes) != 1:
        raise ValueError("smoke images differ in dimensions")
    width, height = sizes.pop()
    return {"schema_version": 1, "inputs": {input_fingerprint(hashes, times): {
        "fps": fps, "total_num_frames": count, "duration": count/fps, "frames_indices": indices,
        "width": width, "height": height, "video_backend": "frozen-jpeg"}},
        "source_manifest_sha256": sha(raw), "source_video_sha256": sha(video_path.read_bytes()),
        "policy": "Non-evaluation schema smoke. Only source decoder facts and the supplied eight JPEGs; video media is not passed to the model."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--provenance", type=Path)
    source.add_argument("--smoke-video", type=Path)
    parser.add_argument("--merge-registry", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    value = prepare(args.manifest, args.provenance) if args.provenance else prepare_smoke(args.manifest, args.smoke_video)
    if args.merge_registry:
        previous = strict_json(args.merge_registry.read_bytes())
        for key, metadata in previous["inputs"].items():
            if key in value["inputs"] and value["inputs"][key] != metadata:
                raise ValueError("conflicting registry metadata")
            value["inputs"][key] = metadata
        value["merged_registry_sha256"] = sha(args.merge_registry.read_bytes())
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(compact(value) + "\n")
    print(json.dumps({"inputs": len(value["inputs"]), "output_sha256": sha(args.output.read_bytes())}))


if __name__ == "__main__":
    main()
