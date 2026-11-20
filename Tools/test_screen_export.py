"""Screen export uses the same immutable-file checks as camera export; no device or credentials."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

import standalone_glasses as tool

NAME = "screen-1794593852000.mp4"
SIDECAR = NAME.removesuffix(".mp4") + ".json"
SERIAL = "SCREEN_FIXTURE_ONLY"


class ScreenAdb:
    def __init__(self, monkeypatch, **metadata):
        data = dict(schema="satori.screen-recording/1", state="saved", finalized=True,
                    path="/data/user/0/com.satori.glasses.capture/files/screen-recordings/" + NAME,
                    with_audio=False, timeline_mapping="manual_sync_required")
        data.update(metadata)
        self.sources = {NAME: b"synthetic-screen-video\r\n\x00" * 100,
                        SIDECAR: json.dumps(data).encode()}
        self.calls, self.streamed = [], []
        self.listing = None
        self.corrupt = False
        monkeypatch.setattr(tool, "find_adb", lambda: Path("fake-adb.exe"))
        monkeypatch.setattr(tool.subprocess, "run", self.run)

    def run(self, words, **kwargs):
        assert words[:8] == ["fake-adb.exe", "-s", SERIAL, "exec-out", "run-as", tool.PACKAGE, "sh", "-c"]
        script = words[8]
        assert "directory='files/screen-recordings'" in script
        assert "first-person-videos" not in script and "shared_prefs" not in script
        assert not kwargs.get("shell") and "input" not in kwargs
        self.calls.append(script)
        if "# satori-video-list-v1" in script:
            listing = self.listing
            if listing is None:
                listing = f"video\t{NAME}\t{len(self.sources[NAME])}\n"
                if SIDECAR in self.sources:
                    listing += f"sidecar\t{SIDECAR}\t{len(self.sources[SIDECAR])}\n"
            return subprocess.CompletedProcess(words, 0, (listing + "done\tvideo-list-v1\n").encode(), b"")
        name = re.search(r"export_name='([^']+)'", script).group(1)
        assert "opened=$(readlink /proc/self/fd/0 <&3)" in script
        value = self.sources[name]
        if "# satori-video-info-v1" in script:
            output = f"{len(value)}\t{hashlib.sha256(value).hexdigest()}\n".encode()
        elif kwargs["stdout"] is subprocess.PIPE:
            assert name.endswith(".json"), "video must stream rather than buffer"
            assert "head -c 65537" in script, "metadata reads must have a device-side cap"
            output = value[:65537]
        else:
            self.streamed.append(name)
            kwargs["stdout"].write(b"corrupt" if self.corrupt and name == NAME else value)
            output = None
        return subprocess.CompletedProcess(words, 0, output, b"")


def invoke(monkeypatch, action, *extra):
    monkeypatch.setattr(sys, "argv", ["standalone_glasses.py", "--serial", SERIAL, action, *map(str, extra)])
    tool.main()


def test_screen_export_preserves_sidecar_and_streams_hash_verified_video(monkeypatch, tmp_path, capsys):
    adb = ScreenAdb(monkeypatch)
    invoke(monkeypatch, "export-screen-recordings", "--output", tmp_path)
    assert adb.streamed == [NAME, SIDECAR]
    assert (tmp_path / NAME).read_bytes() == adb.sources[NAME]
    assert (tmp_path / SIDECAR).read_bytes() == adb.sources[SIDECAR]
    assert "SHA256" in capsys.readouterr().out


@pytest.mark.parametrize("metadata", [dict(state="recording", finalized=False), dict(state="error", finalized=False), dict(state="saved", finalized=False)])
def test_screen_inventory_does_not_offer_unfinished_sidecar(monkeypatch, capsys, metadata):
    adb = ScreenAdb(monkeypatch, **metadata)
    invoke(monkeypatch, "list-screen-recordings")
    assert json.loads(capsys.readouterr().out) == {"screen_recordings": []}
    assert not adb.streamed


def test_screen_inventory_waits_for_committed_sidecar(monkeypatch, capsys):
    adb = ScreenAdb(monkeypatch)
    del adb.sources[SIDECAR]
    invoke(monkeypatch, "list-screen-recordings")
    assert json.loads(capsys.readouterr().out) == {"screen_recordings": []}
    assert not adb.streamed


@pytest.mark.parametrize("name", ["../shared_prefs/standalone_secure.xml", "screen-a.mp4", "screen-1;echo-hi.mp4", "recording-12.mp4"])
def test_screen_inventory_rejects_cross_scope_names(monkeypatch, name):
    adb = ScreenAdb(monkeypatch)
    adb.listing = f"video\t{name}\t100\n"
    with pytest.raises(ValueError):
        invoke(monkeypatch, "list-screen-recordings")
    assert not adb.streamed


def test_screen_export_rejects_overwrite_and_retains_corrupt_partial(monkeypatch, tmp_path):
    adb = ScreenAdb(monkeypatch)
    final = tmp_path / NAME
    final.write_bytes(b"existing-user-file")
    with pytest.raises(FileExistsError):
        invoke(monkeypatch, "export-screen-recordings", "--output", tmp_path)
    assert final.read_bytes() == b"existing-user-file"
    out = tmp_path / "new"
    adb.corrupt = True
    with pytest.raises(RuntimeError):
        invoke(monkeypatch, "export-screen-recordings", "--output", out)
    assert not (out / NAME).exists()
    assert (out / (NAME + ".part")).read_bytes() == b"corrupt"


@pytest.mark.parametrize("metadata", [dict(schema="other"), dict(path="/data/user/0/com.satori.glasses.capture/shared_prefs/other.mp4"), dict(finalized="true")])
def test_screen_metadata_must_describe_the_correct_completed_recording(monkeypatch, metadata):
    adb = ScreenAdb(monkeypatch, **metadata)
    with pytest.raises(ValueError):
        invoke(monkeypatch, "list-screen-recordings")
    assert not adb.streamed


def test_screen_metadata_large_response_is_rejected_before_buffering(monkeypatch):
    adb = ScreenAdb(monkeypatch)
    adb.sources[SIDECAR] = b"x" * 65537
    with pytest.raises(ValueError):
        invoke(monkeypatch, "list-screen-recordings")
    assert len(adb.calls) == 1


def test_screen_export_needs_explicit_destination(monkeypatch, capsys):
    adb = ScreenAdb(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        invoke(monkeypatch, "export-screen-recordings")
    assert exc.value.code == 2
    assert not adb.calls
    assert "--output" in capsys.readouterr().err
