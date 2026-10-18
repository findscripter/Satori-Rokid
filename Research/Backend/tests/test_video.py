import json

import cv2
import numpy as np
import pytest

from satori_repro.video import extract_video


@pytest.fixture
def video_path(tmp_path):
    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
    assert writer.isOpened(), "MJPG test encoder is required"
    try:
        for index in range(25):
            writer.write(np.full((24, 32, 3), index * 8, dtype=np.uint8))
    finally:
        writer.release()
    return path


def test_extracts_by_video_time_and_writes_portable_manifest(video_path, tmp_path):
    output = tmp_path / "frames"
    manifest_path = extract_video(video_path, output, fps=2, max_frames=20)
    assert manifest_path == output / "session.json"
    manifest = json.loads(manifest_path.read_text())
    assert set(manifest) == {"session_id", "frames", "events"}
    assert manifest["events"] == []
    assert [frame["timestamp_ms"] for frame in manifest["frames"]] == [0, 500, 1000, 1500, 2000]
    assert len({frame["frame_id"] for frame in manifest["frames"]}) == 5
    for frame in manifest["frames"]:
        image = output / frame["image_path"]
        assert image.is_file() and image.suffix == ".jpg"
        assert cv2.imread(str(image)).shape == (24, 32, 3)
    metadata = json.loads((output / "extraction.json").read_text())
    assert metadata["truncated"] is False
    assert metadata["saved_frames"] == 5 and metadata["sample_fps"] == 2


def test_limit_records_actual_truncation(video_path, tmp_path):
    extract_video(video_path, tmp_path / "limited", fps=2, max_frames=2)
    meta = json.loads((tmp_path / "limited" / "extraction.json").read_text())
    assert meta["truncated"] is True and meta["saved_frames"] == 2
    extract_video(video_path, tmp_path / "exact", fps=2, max_frames=5)
    assert json.loads((tmp_path / "exact" / "extraction.json").read_text())["truncated"] is False


def test_sampling_faster_than_source_has_no_duplicate_frames(video_path, tmp_path):
    manifest = json.loads(extract_video(video_path, tmp_path / "dense", fps=30).read_text())
    times = [frame["timestamp_ms"] for frame in manifest["frames"]]
    assert len(times) == 25 and times == sorted(set(times))


@pytest.mark.parametrize("kwargs", [{"fps": 0}, {"fps": -1}, {"fps": float("nan")},
    {"fps": float("inf")}, {"fps": True}, {"max_frames": 0}, {"max_frames": -1},
    {"max_frames": 1.5}, {"max_frames": True}])
def test_rejects_invalid_sampling_parameters(video_path, tmp_path, kwargs):
    with pytest.raises(ValueError):
        extract_video(video_path, tmp_path / "bad", **kwargs)
    assert not (tmp_path / "bad").exists()


def test_does_not_overwrite_output(video_path, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    keep = output / "keep.txt"
    keep.write_text("keep")
    with pytest.raises(FileExistsError):
        extract_video(video_path, output)
    assert keep.read_text() == "keep"


@pytest.mark.parametrize("source", ["missing.avi", "broken.avi", 0])
def test_rejects_missing_corrupt_and_device_index(tmp_path, source):
    (tmp_path / "broken.avi").write_bytes(b"not a video")
    input_path = tmp_path / source if isinstance(source, str) else source
    with pytest.raises(ValueError):
        extract_video(input_path, tmp_path / "bad")


def test_uses_variable_source_timestamps(monkeypatch, tmp_path):
    source = tmp_path / "vfr.mp4"
    source.write_bytes(b"test source")
    timestamps = [0, 80, 140, 510, 1010, 1700]

    class Capture:
        index = -1
        released = False

        def isOpened(self):
            return True

        def read(self):
            self.index += 1
            return (True, np.zeros((4, 4, 3), dtype=np.uint8)) if self.index < len(timestamps) else (False, None)

        def get(self, prop):
            return timestamps[self.index] if prop == cv2.CAP_PROP_POS_MSEC else 30.0

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)
    manifest = json.loads(extract_video(source, tmp_path / "vfr", fps=2).read_text())
    assert [frame["timestamp_ms"] for frame in manifest["frames"]] == [0, 510, 1010, 1700]
    assert capture.released
