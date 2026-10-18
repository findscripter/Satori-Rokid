"""Data-isolation and causal-history regressions for real GTEA preparation."""
from collections import Counter
import json
from pathlib import Path

import pytest

import prepare_gtea_satori as prep
import satori_paper_benchmark as bench


def test_actions_do_not_include_object_presence_rows():
    actions, objects = prep.parse_annotation(
        "<take><cup> (19-59) [0]\n<cup> (19-99)\n<spoon> <221-359>\n", "S1_Coffee_C1")
    assert len(actions) == 1
    assert len(objects) == 2
    assert actions[0]["verb"] == "take"
    assert actions[0]["nouns"] == "cup"


def test_unknown_source_syntax_requires_audit():
    with pytest.raises(ValueError, match="Unrecognized"):
        prep.parse_annotation("<take><cup> (59-19) maybe\n", "S1_Coffee_C1")


@pytest.mark.parametrize("horizon", [0, 1000, 3000])
def test_strict_causal_history_has_no_target_or_future_frame(horizon):
    indices = prep.forecast_indices(92, horizon)
    onset = prep.timestamp_ms(91)
    cutoff = onset - horizon
    assert min(indices) >= 0
    assert all(prep.timestamp_ms(index) < cutoff for index in indices)
    assert all(b - a == 15 for a, b in zip(indices, indices[1:]))
    assert onset - cutoff == horizon


def test_exact_six_second_onset_is_ineligible_for_strict_history():
    rows = []
    for subject in ("S1", "S2", "S3", "S4"):
        for line in range(1, 12):
            rows.append({"source_video_id": f"{subject}_Coffee_C1", "source_line": line,
                         "start_frame": 91 if line == 1 else 92 + line * 15})
    selected, counts = prep.select_boundaries(rows)
    assert counts == {"S1": 10, "S2": 10, "S3": 10, "S4": 10}
    assert all(row["start_frame"] > 91 for row in selected)
    assert Counter(row["source_video_id"].split("_")[0] for row in selected) == {"S1": 8, "S2": 8, "S3": 8, "S4": 8}
    assert selected == prep.select_boundaries(list(reversed(rows)))[0]


def test_source_spellings_are_preserved_including_rare_extra_classes():
    actions, _ = prep.parse_annotation(
        "<put><tea> (529-546) [0]\n<stir><cup> (931-975) [0]\n<stir><spoon,cup> (950-975) [1]", "S4_Tea_C1")
    assert {(row["verb"], row["nouns"]) for row in actions} == {
        ("put", "tea"), ("stir", "cup"), ("stir", "spoon,cup")}


@pytest.mark.parametrize("folder,expected_count", [("goal28", 28), ("forecast96-raw73-v2", 96)])
def test_real_frozen_artifacts_have_valid_images_gt_and_anonymous_payload(folder, expected_count):
    root = Path(__file__).resolve().parents[1] / "Research" / "Data" / "gtea-satori" / folder
    if not (root / "infer-manifest.json").exists():
        pytest.skip("Official GTEA artifacts are not installed")
    load = lambda name: json.loads((root / name).read_text(encoding="utf-8"))
    manifest, taxonomy, gt = load("infer-manifest.json"), load("taxonomy.json"), load("gt.json")
    bench.validate_manifest(manifest, taxonomy)
    bench._validate_gt(gt, manifest, taxonomy)
    assert len(manifest["samples"]) == expected_count
    assert bench._image_audit(manifest, root / "infer-manifest.json")["passed"]
    assert len(taxonomy["actions"]) == 73
    assert len(taxonomy["goals"]) == 7
    for sample in manifest["samples"]:
        visible = bench.visible_payload(sample, taxonomy)
        assert "source_video_id" not in visible
        assert "subject" not in visible
        assert "answers" not in visible
        assert all(frame["image_path"].startswith(f"frames/{sample['sample_id']}/v") for frame in sample["frames"])
    if folder.startswith("forecast"):
        grouped = {}
        for row in gt["samples"]:
            metadata = row["metadata"]
            key = (metadata["source_video_id"], metadata["source_annotation_line"])
            grouped.setdefault(key, set()).add(metadata["look_ahead_ms"])
        assert len(grouped) == 32
        assert all(horizons == {0, 1000, 3000} for horizons in grouped.values())
