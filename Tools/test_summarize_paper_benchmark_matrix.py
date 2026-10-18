"""Matrix extension must preserve explicit missing results and exceptions."""

import json

import pytest

import summarize_paper_benchmark_matrix as matrix


@pytest.fixture
def empty_matrix_data(tmp_path):
    for task, (rel, count, _, _) in matrix.TASKS.items():
        data = tmp_path / "Research/Data" / rel
        data.mkdir(parents=True)
        files = {"infer-manifest.json": {"benchmark_id": task, "samples": [{} for _ in range(count)]},
                 "taxonomy.json": {}, "gt.json": {}}
        for name, value in files.items():
            (data / name).write_text(json.dumps(value), encoding="utf-8")
    tools = tmp_path / "Tools"
    tools.mkdir()
    (tools / "satori_paper_benchmark.py").write_text("unit fixture", encoding="utf-8")
    protocol = tmp_path / "protocol.md"
    protocol.write_text("unit protocol", encoding="utf-8")
    return tmp_path, protocol


def test_extra_api_aliases_form_missing_cells_without_fabricated_scores(empty_matrix_data):
    root, protocol = empty_matrix_data
    names = (*matrix.MODELS, "strong-gpt-test-alias", "claude-test-alias")
    result, comparisons = matrix.summarize(root, {}, protocol, model_slots=names)
    assert result["model_slots"] == list(names)
    assert result["audited_runs"] == 0 and result["untested_cells"] == 18
    assert len(result["matrix"]) == 18 and not comparisons
    assert all(row["accuracy"] is None and row["correct"] is None for row in result["matrix"])
    assert result["gt_binding_exceptions"] == []


@pytest.mark.parametrize("names", [[], ["GPT", "GPT"], [" "]])
def test_ambiguous_slots_are_rejected(empty_matrix_data, names):
    root, protocol = empty_matrix_data
    with pytest.raises(ValueError, match="distinct nonempty"):
        matrix.summarize(root, {}, protocol, model_slots=names)


def test_witness_cannot_be_assigned_to_unselected_slot(empty_matrix_data):
    root, protocol = empty_matrix_data
    with pytest.raises(ValueError, match="explicitly selected"):
        matrix.summarize(root, {}, protocol, gt_witnesses={("goal28", "Dispider"): root / "prior"})
