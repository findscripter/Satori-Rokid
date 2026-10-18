import io
import json
import tarfile

import pytest

from prepare_egotaskqa_satori import prepare, read_questions, safe_video_name, select_questions


def questions():
    return [{"question_id": index, "question": f"What happened {index}?",
             "answer": f"secret-label-{index}", "interval": f"scene{index}|P1|0|100",
             "reference": "indirect", "category": ["world"], "type": ["descriptive"]}
            for index in range(20)]


def make_archive(path, items):
    data = json.dumps(items).encode()
    with tarfile.open(path, "w:gz") as stream:
        member = tarfile.TarInfo("Data/qa_ori/indirect/test_qas.json")
        member.size = len(data)
        stream.addfile(member, io.BytesIO(data))


def test_selection_ignores_json_order_answers_and_media():
    original = questions()
    changed = [{**item, "answer": "changed", "media_exists": True} for item in reversed(original)]
    assert [item["question_id"] for item in select_questions(original, 8)] == [
        item["question_id"] for item in select_questions(changed, 8)]


def test_gt_never_enters_inference_and_missing_media_preserves_denominator(tmp_path):
    archive = tmp_path / "data.tar.gz"
    make_archive(archive, questions())
    summary = prepare(archive, tmp_path / "out", 8)
    inference = (tmp_path / "out/pending-media-manifest.json").read_text()
    gt = json.loads((tmp_path / "out/gt.json").read_text())
    assert "secret-label" not in inference
    assert "answer" not in inference
    assert summary["fixed_denominator"] == len(gt["samples"]) == 8
    assert summary["status"] == "not_run_missing_video"
    assert summary["media_ready_questions"] == 0
    assert not (tmp_path / "out/infer-manifest.json").exists()
    assert [item["sample_id"] for item in gt["samples"]] == [f"s{index:03d}" for index in range(8)]


def test_reject_direct_references_and_duplicate_ids(tmp_path):
    archive = tmp_path / "data.tar.gz"
    items = questions()
    items[0]["reference"] = "direct"
    make_archive(archive, items)
    with pytest.raises(ValueError, match="Direct-reference"):
        read_questions(archive)
    make_archive(archive, questions() + questions()[:1])
    with pytest.raises(ValueError, match="unique"):
        read_questions(archive)


def test_frozen_ids_cannot_be_silently_changed(tmp_path):
    archive = tmp_path / "data.tar.gz"
    make_archive(archive, questions())
    prepare(archive, tmp_path / "out", 8, 1)
    with pytest.raises(ValueError, match="frozen selection"):
        prepare(archive, tmp_path / "out", 8, 2)


def test_windows_names_are_stable_and_distinct():
    first = safe_video_name("scene|P1|0|100")
    assert "|" not in first and first.endswith(".mp4")
    assert first == safe_video_name("scene|P1|0|100")
    assert first != safe_video_name("scene|P1|1|100")
