"""Pinned CLI execution profiles must never weaken the default audit contract."""

from copy import deepcopy
import json

import pytest

import compare_paper_model_runs as compare
from test_compare_paper_model_runs import frozen, write


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def configure_claude(data, run):
    meta = load(run / "run.json")
    profile = {"schema_version": 1, "profile_id": "unit-claude-high", "model": "claude-opus-5-5",
               "expected_response_model": "claude-opus-5-5", "endpoint": "cli://reclaude/claude-opus-5-5",
               "generation_parameters": {"max_tokens": 4096, "effort": "high"}, "timeout_seconds": 180,
               "input_representation": "claude_cli_image_blocks", "prompt_sha256": meta["prompt_sha256"]}
    profile_path = data / "execution-profile.json"
    digest = write(profile_path, profile)
    protocol = load(run / "protocol.json")
    protocol["execution_profile_sha256"] = digest
    meta["source_protocol_sha256"] = write(data / "protocol.json", protocol, crlf=True)
    meta["protocol_snapshot_sha256"] = write(run / "protocol.json", protocol)
    for key in ("model", "expected_response_model", "endpoint", "generation_parameters", "timeout_seconds", "prompt_sha256"):
        meta[key] = profile[key]
    write(run / "run.json", meta)
    path = run / "api-calls.jsonl"
    traces = [json.loads(line) for line in path.read_text().splitlines()]
    for trace in traces:
        trace.update(requested_model=profile["model"], returned_model=profile["expected_response_model"],
                     expected_response_model=profile["expected_response_model"], generation_parameters=profile["generation_parameters"])
    path.write_text("".join(json.dumps(trace) + "\n" for trace in traces), encoding="utf-8")
    return profile_path, digest


def test_cli_requires_explicit_profile_even_if_run_claims_a_binding(frozen):
    data, run = frozen
    configure_claude(data, run)
    with pytest.raises(ValueError, match="explicitly pinned"):
        compare.audit_and_score(run, data)


def test_pinned_profile_preserves_actual_cli_parameters(frozen):
    data, run = frozen
    path, digest = configure_claude(data, run)
    result = compare.audit_and_score(run, data, execution_profile=path, execution_profile_sha256=digest)
    assert result["input_representation"] == "claude_cli_image_blocks"
    assert result["generation_parameters"] == {"max_tokens": 4096, "effort": "high"}
    assert "temperature" not in result["generation_parameters"]
    assert result["execution_profile"]["sha256"] == digest
    assert result["correct"] == 1


@pytest.mark.parametrize("case", ["missing_path", "missing_hash", "wrong_hash", "changed_file"])
def test_profile_hash_must_be_explicit_and_match_bytes(frozen, case):
    data, run = frozen
    path, digest = configure_claude(data, run)
    if case == "missing_path":
        path = None
    elif case == "missing_hash":
        digest = None
    elif case == "wrong_hash":
        digest = "0" * 64
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="profile"):
        compare.audit_and_score(run, data, execution_profile=path, execution_profile_sha256=digest)


@pytest.mark.parametrize("case", ["missing", "wrong"])
def test_run_protocol_must_bind_the_exact_profile(frozen, case):
    data, run = frozen
    path, digest = configure_claude(data, run)
    protocol = load(run / "protocol.json")
    if case == "missing":
        del protocol["execution_profile_sha256"]
    else:
        protocol["execution_profile_sha256"] = "0" * 64
    meta = load(run / "run.json")
    meta["source_protocol_sha256"] = write(data / "protocol.json", protocol, crlf=True)
    meta["protocol_snapshot_sha256"] = write(run / "protocol.json", protocol)
    write(run / "run.json", meta)
    with pytest.raises(ValueError, match="protocol does not bind"):
        compare.audit_and_score(run, data, execution_profile=path, execution_profile_sha256=digest)


@pytest.mark.parametrize("where", ["metadata", "trace"])
def test_parameter_tampering_rejected_with_valid_profile(frozen, where):
    data, run = frozen
    path, digest = configure_claude(data, run)
    if where == "metadata":
        meta = load(run / "run.json")
        meta["generation_parameters"]["effort"] = "low"
        write(run / "run.json", meta)
    else:
        trace_path = run / "api-calls.jsonl"
        traces = [json.loads(line) for line in trace_path.read_text().splitlines()]
        traces[0]["generation_parameters"]["temperature"] = 0
        trace_path.write_text("".join(json.dumps(trace) + "\n" for trace in traces), encoding="utf-8")
    with pytest.raises(ValueError):
        compare.audit_and_score(run, data, execution_profile=path, execution_profile_sha256=digest)


def test_profile_does_not_allow_arbitrary_generation_settings(frozen):
    data, run = frozen
    path, _ = configure_claude(data, run)
    profile = load(path)
    profile["generation_parameters"] = {"max_tokens": 99999, "effort": "low"}
    digest = write(path, profile)
    with pytest.raises(ValueError, match="outside.*supported"):
        compare.audit_and_score(run, data, execution_profile=path, execution_profile_sha256=digest)


def test_cross_provider_comparison_requires_same_observations_mode(frozen):
    data, run = frozen
    previous = compare.audit_and_score(run, data)
    path, digest = configure_claude(data, run)
    claude = compare.audit_and_score(run, data, execution_profile=path, execution_profile_sha256=digest)
    with pytest.raises(ValueError, match="generation parameters"):
        compare.compare_results([previous, claude])
    result = compare.compare_results([previous, claude], comparison_mode="same_observations")
    assert result["generation_parameters_equal"] is False
    assert result["execution_parameters_equal"] is False
    assert "generation_parameters" not in result["input_fingerprint"]
    assert result["models"][0]["generation_parameters"] == {"max_tokens": 4096, "temperature": 0.0}
    assert result["models"][1]["generation_parameters"] == {"max_tokens": 4096, "effort": "high"}
    assert len(result["paired"]) == 4


@pytest.mark.parametrize("field", ["manifest_sha256", "taxonomy_sha256", "gt_sha256", "prompt_sha256"])
def test_same_observations_never_allows_changed_inputs(frozen, field):
    data, run = frozen
    a = compare.audit_and_score(run, data)
    b = deepcopy(a)
    b["model"] = "other-model"
    b["input_fingerprint"][field] = "changed"
    with pytest.raises(ValueError, match="same frozen"):
        compare.compare_results([a, b], comparison_mode="same_observations")
