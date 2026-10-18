import io
import json
import shutil
import tarfile

import cv2
import numpy as np
import pytest

from prepare_egotaskqa_frames import extract_clip, file_sha, prepare, verified_video
from prepare_egotaskqa_satori import prepare as freeze_questions, safe_video_name


def clip(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (64, 48))
    assert writer.isOpened()
    for number in range(20):
        writer.write(np.full((48, 64, 3), number * 10, dtype=np.uint8))
    writer.release()


def test_real_decoder_midbins_are_causal_correct_images_and_timestamps(tmp_path):
    video = tmp_path / "clip.mp4"
    clip(video)
    proof = extract_clip(video, tmp_path / "frames", "scene|P1|0|19", file_sha(video))
    assert proof["decoded_frame_count"] == 20
    assert proof["frame_count_equals_inclusive_interval_span"]
    assert [row["source_frame_index_zero_based"] for row in proof["frames"]] == [2, 7, 12, 17]
    assert [row["timestamp_ms"] for row in proof["frames"]] == [133, 467, 800, 1133]
    for row in proof["frames"]:
        image = cv2.imread(str(tmp_path / "frames" / row["filename"]))
        assert abs(float(image.mean()) - row["source_frame_index_zero_based"] * 10) < 6
        assert row["timestamp_ms"] < proof["cutoff_ms"]


def test_changed_frozen_image_is_not_silently_regenerated(tmp_path):
    video = tmp_path / "clip.mp4"
    clip(video)
    extract_clip(video, tmp_path / "frames", "scene|P1|0|19", file_sha(video))
    (tmp_path / "frames/000.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen frame changed"):
        extract_clip(video, tmp_path / "frames", "scene|P1|0|19", file_sha(video))


def test_receipt_rejects_wrong_interval_and_changed_file(tmp_path):
    interval = "scene|P1|0|19"
    video = tmp_path / safe_video_name(interval)
    video.write_bytes(b"media")
    receipt = {"source_interval": interval, "archive_member": f"qa_videos/{interval}.mp4",
               "path": video.name, "bytes": 5, "sha256": file_sha(video)}
    assert verified_video(tmp_path, interval, receipt) == video
    with pytest.raises(ValueError, match="interval differs"):
        verified_video(tmp_path, "other|P1|0|19", receipt)
    video.write_bytes(b"media2")
    with pytest.raises(ValueError, match="SHA256"):
        verified_video(tmp_path, interval, receipt)


def test_complete_manifest_has_only_questions_and_verified_images(tmp_path):
    archive = tmp_path / "qas.tar.gz"
    items = [{"question_id": number, "question": f"What is happening at test {number}?",
              "answer": f"hidden-answer-{number}", "interval": f"scene{number}|P1|0|19",
              "reference": "indirect", "category": ["world"], "type": ["descriptive"]}
             for number in range(4)]
    raw = json.dumps(items).encode()
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo("Data/qa_ori/indirect/test_qas.json")
        member.size = len(raw)
        stream.addfile(member, io.BytesIO(raw))
    frozen = tmp_path / "frozen"
    freeze_questions(archive, frozen, count=4)
    template = tmp_path / "template.mp4"
    clip(template)
    receipts = []
    for item in items:
        target = tmp_path / safe_video_name(item["interval"])
        shutil.copyfile(template, target)
        receipts.append({"source_interval": item["interval"],
                         "archive_member": "qa_videos/" + item["interval"] + ".mp4",
                         "path": target.name, "bytes": target.stat().st_size, "sha256": file_sha(target)})
    receipt = tmp_path / "media-provenance.json"
    source = {"test_fixture": True, "archive_verified": False,
              "lfs_sha256": "a" * 64, "archive_sha256": "a" * 64}
    receipt.write_text(json.dumps({"source": source, "videos": receipts}))
    with pytest.raises(ValueError, match="upstream LFS SHA256"):
        prepare(frozen, receipt)
    source["archive_verified"] = True
    receipt.write_text(json.dumps({"source": source, "videos": receipts}))
    prior = frozen / "train-mode-prior-20261017.json"
    prior.write_text(json.dumps({"selected_answer": "private-training-label"}))
    result = prepare(frozen, receipt)
    assert result["questions"] == 4 and result["frames"] == 16
    manifest_text = (frozen / "infer-manifest.json").read_text()
    assert "hidden-answer" not in manifest_text and "source_interval" not in manifest_text
    manifest = json.loads(manifest_text)
    assert all(len(row["frames"]) == 4 for row in manifest["samples"])
    registry_path = frozen / "native-metadata-registry-20261017.json"
    registry = json.loads(registry_path.read_text())
    assert set(registry["samples"]) == {row["sample_id"] for row in manifest["samples"]}
    assert all(set(row) == {"fps", "total_num_frames", "duration", "width", "height", "frames_indices", "video_backend"}
               for row in registry["samples"].values())
    assert "hidden-answer" not in registry_path.read_text()
    native_protocol = json.loads((frozen / "protocol-videochat3-20261017.json").read_text())
    assert native_protocol["technical_metadata_registry_sha256"] == file_sha(registry_path)
    assert native_protocol["training_answer_prior_sha256"] == file_sha(prior)
    assert "private-training-label" not in manifest_text
    for sample in manifest["samples"]:
        native = registry["samples"][sample["sample_id"]]
        assert all(abs(frame["timestamp_ms"] - index / native["fps"] * 1000) <= 1
                   for frame, index in zip(sample["frames"], native["frames_indices"]))
    with pytest.raises(FileExistsError, match="already frozen"):
        freeze_questions(archive, frozen, count=4)
    with pytest.raises(FileExistsError, match="already frozen"):
        prepare(frozen, receipt)
