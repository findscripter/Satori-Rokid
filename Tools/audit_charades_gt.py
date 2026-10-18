"""Independently match frozen benchmark inputs/labels to the official CSV."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "infer-manifest.json").read_text(encoding="utf-8-sig"))
    taxonomy = json.loads((root / "taxonomy.json").read_text(encoding="utf-8-sig"))
    truth = json.loads((root / "gt.json").read_text(encoding="utf-8-sig"))
    with (root / "source/CharadesEgo_v1_test_only1st.csv").open(encoding="utf-8-sig", newline="") as stream:
        official = {row["id"]: row for row in csv.DictReader(stream)}
    scene_names = {item["id"]: item["label"] for item in taxonomy["scenes"]}
    classes = {}
    for line in (root / "source/Charades_v1_classes.txt").read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            key, value = line.strip().split(" ", 1)
            classes[key] = value
    assert {item["id"]: item["label"] for item in taxonomy["actions"]} == classes
    assert len(classes) == 157 and len(scene_names) == 16
    labels = {item["sample_id"]: item for item in truth["labels"]}
    samples = manifest["samples"]
    assert len(samples) == len(labels) == 32
    assert manifest["benchmark_id"] == truth["benchmark_id"] == "charades-ego-32-v1"
    source_ids, seen = set(), set()
    scene_support, action_support = Counter(), Counter()
    dimensions = Counter()
    for sample in samples:
        sid = sample["sample_id"]
        assert sid in labels and sid not in seen
        seen.add(sid)
        gt = labels[sid]
        source_id = gt["source_video_id"]
        assert source_id not in source_ids and source_id in official
        source_ids.add(source_id)
        row = official[source_id]
        assert row["egocentric"] == "Yes" and source_id.endswith("EGO")
        assert scene_names[gt["scene_id"]] == row["scene"]
        expected = {item.split()[0] for item in row["actions"].split(";") if item.strip()}
        assert set(gt["action_ids"]) == expected and len(gt["action_ids"]) == len(expected)
        assert expected <= classes.keys()
        scene_support[row["scene"]] += 1
        action_support.update(expected)
        frames = sample["frames"]
        assert len(frames) == 8
        assert [frame["frame_id"] for frame in frames] == [f"v{i:03d}" for i in range(8)]
        times = [frame["timestamp_ms"] for frame in frames]
        assert times == sorted(set(times)) and min(times) >= 0
        for frame in frames:
            path = (root / frame["image_path"]).resolve()
            assert path.is_relative_to(root) and digest(path) == frame["sha256"]
            image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            assert image is not None and image.shape[0] > 0 and image.shape[1] > 0
            dimensions[f"{image.shape[1]}x{image.shape[0]}"] += 1
    assert set(scene_support) == {
        "Bedroom", "Kitchen", "Living room", "Home Office / Study (A room in a house used for work)"
    }
    assert set(scene_support.values()) == {8}
    return {
        "success": True,
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": manifest["benchmark_id"],
        "samples": len(samples), "distinct_official_ego_videos": len(source_ids),
        "images": len(samples) * 8, "scene_support": dict(scene_support),
        "gt_positive_action_labels": sum(action_support.values()),
        "gt_present_action_classes": len(action_support),
        "image_dimensions": dict(dimensions),
        "input_sha256": {name: digest(root / name) for name in (
            "infer-manifest.json", "taxonomy.json", "gt.json", "baseline-protocol.json"
        )},
        "official_source_sha256": {name: digest(root / "source" / name) for name in (
            "CharadesEgo.zip", "CharadesEgo_v1_test_only1st.csv", "Charades_v1_classes.txt"
        )},
        "checks": ["GT scene/action IDs exactly match official CSV", "all 32 source videos distinct and egocentric",
                   "four predeclared scenes each have eight samples", "all 256 decoded JPEGs match their input SHA256",
                   "all 157 action class names match the official definition"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("Research/Data/charades-ego-gt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"success": True, "samples": result["samples"], "images": result["images"]}))


if __name__ == "__main__":
    main()
