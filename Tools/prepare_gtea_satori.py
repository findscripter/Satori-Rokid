"""Prepare frozen, anonymous GTEA module probes from the official archives.

Keeps every original verb+noun spelling. The distributed '71' archive actually
contains 528 action rows and 73 spellings; this script never deletes rare rows
to reproduce the paper's reported 525/71 counts. No model or GT scoring is run.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import zipfile


SEED = 20261017
ACTION = re.compile(r"<([^<>]+)><([^<>]+)>\s+\((\d+)-(\d+)\)\s+\[([01])\]")
OBJECT = re.compile(r"<[^<>]+>\s+(?:\(\d+-\d+\)|<\d+-\d+>)")
GOALS = {
    "Cheese": "Make a cheese sandwich",
    "Coffee": "Make coffee",
    "CofHoney": "Make coffee with honey",
    "Hotdog": "Make a hotdog sandwich",
    "Pealate": "Make a peanut butter and jam sandwich",
    "Peanut": "Make a peanut butter sandwich",
    "Tea": "Make sweet tea",
}
ARCHIVES = {
    "gtea_labels_71.zip": {
        "url": "https://www.dropbox.com/s/yb6x4ez10bhx8sb/gtea_labels_71.zip?dl=1",
        "sha256": "d8ea31c7ae2e760d271e33f27e1881d0d2dd4c9f7b3738899053e05d03fb1b32",
        "size_bytes": 13005,
    },
    "gtea_videos.zip": {
        "url": "https://www.dropbox.com/s/xzdw0mloohtxjn5/gtea_videos.zip?dl=1",
        "sha256": "cfcdeadb05340679381930573ac44fffd4d52eb1ed923dd10a1a9030b70c44bd",
        "size_bytes": 128566965,
    },
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parse_annotation(text, source_id):
    """Object-presence annotations are explicitly excluded from action rows."""
    actions, objects, unknown = [], [], []
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        match = ACTION.fullmatch(line)
        if match:
            verb, nouns, start, end, flag = match.groups()
            actions.append({"verb": verb, "nouns": nouns, "start_frame": int(start),
                            "end_frame": int(end), "flag": int(flag),
                            "source_video_id": source_id, "source_line": line_number,
                            "original_line": raw})
        elif OBJECT.fullmatch(line):
            objects.append({"source_line": line_number, "original_line": raw})
        else:
            unknown.append({"source_line": line_number, "original_line": raw})
    if unknown:
        raise ValueError(f"Unrecognized source annotation lines in {source_id}: {unknown}")
    return actions, objects


def timestamp_ms(frame_index):
    return round(frame_index * 1000 / 15)


def forecast_indices(start_frame, look_ahead_ms):
    # Source annotations are interpreted as one-based inclusive frame numbers.
    onset_index = start_frame - 1
    if look_ahead_ms not in (0, 1000, 3000):
        raise ValueError("Unsupported frozen horizon")
    final_index = onset_index - look_ahead_ms * 15 // 1000 - 1
    indices = [final_index - 45, final_index - 30, final_index - 15, final_index]
    if min(indices) < 0:
        raise ValueError("Insufficient causal observation history")
    return indices


def select_boundaries(actions, per_subject=8, seed=SEED):
    by_subject = defaultdict(list)
    for row in actions:
        # Strictly-before-cutoff sampling consumes one extra 15Hz frame. An
        # onset at exactly 6s would require history frame -1 at the 3s horizon.
        if row["start_frame"] - 1 >= 91:
            by_subject[row["source_video_id"].split("_")[0]].append(row)
    rng, selected, counts = random.Random(seed), [], {}
    for subject in ("S1", "S2", "S3", "S4"):
        eligible = sorted(by_subject[subject], key=lambda row: (row["source_video_id"], row["source_line"]))
        counts[subject] = len(eligible)
        if len(eligible) < per_subject:
            raise ValueError(f"Insufficient eligible boundaries for {subject}")
        selected.extend(sorted(rng.sample(eligible, per_subject), key=lambda row: (row["source_video_id"], row["source_line"])))
    return selected, counts


def extract_frames(video_path, indices, output_dir):
    import cv2
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        for number, index in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode {video_path.name} frame {index}")
            path = output_dir / f"v{number:03d}.jpg"
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise OSError(f"Cannot write frame {path}")
            frames.append({"frame_id": f"v{number:03d}", "timestamp_ms": timestamp_ms(index),
                           "image_path": f"frames/{output_dir.name}/{path.name}", "sha256": sha(path)})
    finally:
        cap.release()
    return frames


def inspect_data(root):
    import cv2
    root = Path(root)
    for name, record in ARCHIVES.items():
        path = root / "source" / name
        if path.stat().st_size != record["size_bytes"] or sha(path) != record["sha256"]:
            raise ValueError(f"Archive differs from frozen download: {name}")
        with zipfile.ZipFile(path) as archive:
            raw_root = (root / "raw").resolve()
            if any(not (raw_root / name).resolve().is_relative_to(raw_root) for name in archive.namelist()):
                raise ValueError("Archive path escapes raw directory")
            archive.extractall(raw_root)
    videos, actions, objects = [], [], []
    for path in sorted((root / "raw" / "Videos").glob("*.mp4")):
        source_id = path.stem
        match = re.fullmatch(r"(S[1-4])_(\w+)_C1", source_id)
        if not match or match[2] not in GOALS:
            raise ValueError(f"Unexpected source video ID: {source_id}")
        cap = cv2.VideoCapture(str(path))
        frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(3)), int(cap.get(4))
        cap.release()
        if fps != 15 or frames < 4 or (width, height) != (720, 404):
            raise ValueError(f"Unexpected official video format: {source_id}")
        annotation = root / "raw" / "labels" / f"{source_id}.txt"
        rows, obj = parse_annotation(annotation.read_text(), source_id)
        for row in rows:
            if not 1 <= row["start_frame"] <= row["end_frame"] <= frames:
                raise ValueError(f"Out-of-range annotation: {row}")
        actions.extend(rows)
        objects.extend([dict(row, source_video_id=source_id) for row in obj])
        videos.append({"source_video_id": source_id, "subject": match[1], "goal_key": match[2],
                       "frame_count": frames, "fps": fps, "width": width, "height": height,
                       "video_sha256": sha(path), "annotation_sha256": sha(annotation),
                       "action_count": len(rows)})
    if len(videos) != 28 or len(actions) != 528:
        raise ValueError("Unexpected source inventory; re-audit before changing the frozen protocol")
    return videos, actions, objects


def prepare(root, task="all", revision="v2"):
    root = Path(root).resolve()
    videos, actions, objects = inspect_data(root)
    pairs = sorted({(row["verb"], row["nouns"]) for row in actions})
    if len(pairs) != 73:
        raise ValueError("Unexpected raw taxonomy; re-audit")
    action_ids = {pair: f"a{index:03d}" for index, pair in enumerate(pairs)}
    goal_keys = sorted(GOALS)
    goal_ids = {key: f"g{index:02d}" for index, key in enumerate(goal_keys)}
    taxonomy = {"schema_version": 1,
                "goals": [{"id": goal_ids[key], "label": GOALS[key]} for key in goal_keys],
                "actions": [{"id": action_ids[pair], "label": f"{pair[0]} ({pair[1]})"} for pair in pairs]}
    selected, eligible_counts = select_boundaries(actions)
    provenance = {"archives": ARCHIVES,
                  "official_page": "https://cbs.ic.gatech.edu/fpv/",
                  "legacy_official_page": "https://ai.stanford.edu/~alireza/GTEA/",
                  "links_snapshot": "https://raw.githubusercontent.com/amitsou/EGTEA_Gaze_Plus_Downloader/f21900e2032640643b8cdf4bf2e53a4f1c349486/data/results.json",
                  "license": "No explicit dataset license found in distributed readme or accessible official page. Preserve academic attribution; do not infer a CC license.",
                  "citation": "Li, Ye, Rehg. Delving into Egocentric Actions. CVPR 2015; Fathi, Ren, Rehg. Learning to Recognize Objects in Egocentric Activities. CVPR 2011."}
    discrepancies = {"readme_claimed_action_classes": 71, "paper_reported_action_instances": 525,
                     "observed_action_classes": 73, "observed_action_instances": 528,
                     "unmapped_rare_spellings": [{"verb": "put", "nouns": "tea", "count": 2}, {"verb": "stir", "nouns": "cup", "count": 1}],
                     "policy": "Keep all source spellings and all 528 rows. Do not merge/drop three rows to manufacture 71/525 equivalence."}
    audit_path = root / f"source-audit-{revision}.json"
    if audit_path.exists():
        raise FileExistsError(f"Refuse to replace frozen source audit: {audit_path}")
    write(audit_path, {"schema_version": 1, "provenance": provenance,
                                       "taxonomy_discrepancy": discrepancies, "videos": videos,
                                       "actions": actions, "object_presence_rows_excluded": objects,
                                       "eligible_boundaries_by_subject": eligible_counts,
                                       "selected_boundaries_before_inference": selected,
                                       "early_onsets_excluded_from_sampling": [row for row in actions if row["start_frame"] - 1 < 91],
                                       "strict_history_boundary_rule": "Onsets >=91/15 seconds provide four 1Hz frames whose last frame is one 15Hz frame before cutoff at the 3s horizon. The exact-6s source onset is excluded before random sampling.",
                                       "raw_taxonomy_counts": [{"verb": pair[0], "nouns": pair[1],
                                                                "count": sum((r["verb"], r["nouns"]) == pair for r in actions)} for pair in pairs]})
    for kind, outname in (("goal_recognition", "goal28"), ("action_forecast", "forecast96-raw73")):
        if task != "all" and task != kind:
            continue
        outname = f"{outname}-{revision}"
        output = root / outname
        output.mkdir(parents=True, exist_ok=True)
        if (output / "infer-manifest.json").exists():
            raise FileExistsError(f"Refuse to replace frozen inference inputs: {output}")
        sample_rows, gt_rows = [], []
        if kind == "goal_recognition":
            rows = [(video, None, None) for video in videos]
        else:
            by_video = {video["source_video_id"]: video for video in videos}
            rows = [(by_video[action["source_video_id"]], action, horizon) for action in selected for horizon in (0, 1000, 3000)]
        for number, (video, action, horizon) in enumerate(rows):
            sid = f"s{number:03d}"
            if kind == "goal_recognition":
                indices = [int((index + 0.5) * video["frame_count"] / 4) for index in range(4)]
                cutoff = timestamp_ms(video["frame_count"] - 1)
                question = "What high-level food or drink preparation task is the camera wearer performing across these sampled clip frames?"
                answer = goal_ids[video["goal_key"]]
            else:
                indices = forecast_indices(action["start_frame"], horizon)
                cutoff = timestamp_ms(action["start_frame"] - 1) - horizon
                question = ("Predict the action whose onset is immediately after the observation cutoff." if horizon == 0 else
                            f"Predict the action whose onset is approximately {horizon} milliseconds after the observation cutoff.")
                answer = action_ids[(action["verb"], action["nouns"])]
            frames = extract_frames(root / "raw" / "Videos" / f"{video['source_video_id']}.mp4", indices, output / "frames" / sid)
            sample_rows.append({"sample_id": sid, "task_kind": kind, "question": question,
                                "cutoff_ms": cutoff, "candidate_group": "goals" if action is None else "actions", "frames": frames})
            metadata = {"subject": video["subject"], "source_video_id": video["source_video_id"],
                        "source_frame_indices_zero_based": indices}
            if action:
                metadata.update({"target_onset_ms": timestamp_ms(action["start_frame"] - 1),
                                 "look_ahead_ms": horizon, "source_annotation_line": action["source_line"],
                                 "target_end_ms": timestamp_ms(action["end_frame"] - 1)})
            gt_rows.append({"sample_id": sid, "answers": [answer], "metadata": metadata})
        benchmark_id = f"gtea-satori-{outname}-20261017"
        manifest = {"schema_version": 1, "benchmark_id": benchmark_id, "samples": sample_rows}
        write(output / "infer-manifest.json", manifest)
        write(output / "taxonomy.json", taxonomy)
        write(output / "gt.json", {"schema_version": 1, "benchmark_id": benchmark_id,
                                   "provenance": provenance, "samples": gt_rows})
        protocol = {"schema_version": 1, "benchmark_id": benchmark_id, "seed": SEED,
                    "source_manifest_sha256": sha(output / "infer-manifest.json"),
                    "source_taxonomy_sha256": sha(output / "taxonomy.json"),
                    "source_gt_sha256": sha(output / "gt.json"),
                    "source_audit_sha256": sha(audit_path),
                    "sampling": "All 28 videos, four mid-bin frames uniformly across each complete clip." if kind == "goal_recognition" else
                                "Before inference: all valid original action onsets >=91/15s (6.0667s) are eligible to support strictly causal 4x1Hz history at 3s horizon; sorted by source ID and line; Python random.Random(20261017), eight per subject, subjects S1-S4 in order; shared 32 targets at 0/1/3s horizons.",
                    "observation_policy": "Goal: four whole-clip frames. Forecast: four frames 1s apart, ending one 15Hz frame strictly before cutoff; no target onset/future frames.",
                    "annotation_indexing": "Interpret source boundaries as one-based inclusive 15Hz frames. Readme states 15Hz but not base; an end equal to frame_count supports this convention. Uncertainty <=one frame (66.7ms).",
                    "taxonomy_discrepancy": discrepancies,
                    "paper_comparison_limit": "Reconstructed zero-shot sparse-frame module probe, not the original Satori ensemble, not trained LOSO, and not an exact reconstruction of unspecified original temporal/metric details. Forecast keeps raw 73 source spellings, not a verified 71-class mapping.",
                    "timing_limit": "Forecast evaluation supplies the observation cutoff and look-ahead horizon. It scores future action identity at that given point; it does not test autonomous timing prediction.",
                    "gt_isolation": "Inference reads only anonymous manifest, candidate taxonomy, protocol and frame bytes. All source IDs, subjects, target onsets and answers are in GT/audit only.",
                    "outcome_policy": "One request per fixed sample; keep errors, truncations and abstentions in the denominator. Never select replacement samples using outcomes."}
        write(output / "protocol.json", protocol)
        print(json.dumps({"output": str(output), "samples": len(sample_rows),
                          "manifest_sha256": protocol["source_manifest_sha256"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("Research/Data/gtea-satori"))
    parser.add_argument("--task", choices=("all", "goal_recognition", "action_forecast"), default="all")
    parser.add_argument("--revision", default="v2", help="New output revision; frozen artifacts are never overwritten")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.revision):
        parser.error("revision must be a simple directory-safe ID")
    prepare(args.root, args.task, args.revision)


if __name__ == "__main__":
    main()
