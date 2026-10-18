"""Small evidence-only regression cases; no inference or external requests."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_video_models as comparison
from test_compare_gt_runs import evidence, mark_identity_failure, write, write_lines


def labelled(evidence):
    mage, gpt, data = evidence
    return {"mage": mage, "gpt": gpt}, data


def test_mixed_timeouts_and_different_snapshot_newlines_are_valid(evidence, tmp_path):
    mage, gpt, data = evidence
    metadata = comparison.read(gpt / "run.json")
    metadata["timeout_seconds"] = 240
    for filename, field in (("manifest.json", "manifest_snapshot_sha256"),
                            ("taxonomy.json", "taxonomy_snapshot_sha256")):
        path = gpt / filename
        original = path.read_bytes()
        path.write_bytes(original.replace(b"\r\n", b"\n") if b"\r\n" in original
                         else original.replace(b"\n", b"\r\n"))
        metadata[field] = comparison.digest(path)
    write(gpt / "run.json", metadata)
    traces = comparison.records(gpt / "api-calls.jsonl")
    for trace in traces:
        trace["timeout_seconds"] = 240
    write_lines(gpt / "api-calls.jsonl", traces)
    result = comparison.compare({"mage": mage, "gpt": gpt}, data)
    assert result["audit_passed"] is True
    assert result["latency_comparison"]["fair_ranking"] is False
    assert result["models"]["mage"]["snapshot_sha256"] != result["models"]["gpt"]["snapshot_sha256"]
    assert result["models"]["mage"]["source_data_sha256"] == result["models"]["gpt"]["source_data_sha256"]
    assert len(comparison.save(result, tmp_path / "comparison.json")) == 3
    with pytest.raises(ValueError, match="already exists"):
        comparison.save(result, tmp_path / "comparison.json")


def test_each_run_must_still_match_its_own_snapshot_hash(evidence):
    runs, data = labelled(evidence)
    path = runs["gpt"] / "manifest.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="snapshot has changed"):
        comparison.compare(runs, data)


@pytest.mark.parametrize("mutation,match", [
    ("payload", "visible context changed"),
    ("image", "image IDs or bytes changed"),
    ("duplicate", "precisely one ordered request trace"),
])
def test_no_payload_image_or_attempt_drift(evidence, mutation, match):
    runs, data = labelled(evidence)
    path = runs["gpt"] / "api-calls.jsonl"
    traces = comparison.records(path)
    if mutation == "payload":
        traces[0]["payload_sha256"] = "0" * 64
    elif mutation == "image":
        traces[0]["image_sha256"]["v000"] = "0" * 64
    else:
        traces.append(traces[0])
    write_lines(path, traces)
    with pytest.raises(ValueError, match=match):
        comparison.compare(runs, data)


def test_diagnosed_model_identity_failure_keeps_32_denominator(evidence):
    runs, data = labelled(evidence)
    mark_identity_failure(runs["gpt"])
    result = comparison.compare(runs, data)
    assert result["fixed_denominator"] == 32
    model = result["models"]["gpt"]
    assert model["failed_samples"] == 1 and model["metrics"]["action_fn"] == 1
    assert model["metrics"]["scene_top1"] == 31 / 32
    assert len(model["identity_anomalies"]) == 1


def test_large_enumeration_remains_scored_with_every_false_positive(evidence):
    runs, data = labelled(evidence)
    path = runs["gpt"]
    predictions = comparison.records(path / "predictions.jsonl")
    answer = predictions[0]["response"]
    answer["action_ids"] = [f"c{i:03d}" for i in range(157)]
    write_lines(path / "predictions.jsonl", predictions)
    traces = comparison.records(path / "api-calls.jsonl")
    traces[0].update(raw_json=answer, raw_json_text=json.dumps(answer))
    write_lines(path / "api-calls.jsonl", traces)
    result = comparison.compare(runs, data)
    model = result["models"]["gpt"]
    assert model["metrics"]["action_tp"] == 32
    assert model["metrics"]["action_fp"] == 156
    assert model["enumeration_diagnostics"]["full_candidate_enumeration_sample_ids"] == ["e000"]
    assert model["enumeration_diagnostics"]["large_sample_fp"] == 156


def test_no_output_writes_into_original_run_and_no_duplicate_labels(evidence):
    runs, data = labelled(evidence)
    result = comparison.compare(runs, data)
    with pytest.raises(ValueError, match="original run directories"):
        comparison.save(result, runs["mage"] / "comparison.json")
    with pytest.raises(ValueError, match="unique label"):
        comparison.parse_run_specs([f"mage={runs['mage']}", f"mage={runs['gpt']}"])


def test_original_source_hash_is_strict_even_when_snapshot_json_matches(evidence):
    runs, data = labelled(evidence)
    path = runs["gpt"] / "run.json"
    metadata = comparison.read(path)
    metadata["source_manifest_sha256"] = "0" * 64
    write(path, metadata)
    with pytest.raises(ValueError, match="Source hash mismatch"):
        comparison.compare(runs, data)
