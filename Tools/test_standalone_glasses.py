"""Credential and export boundaries of the glasses-only helper; no device, network or real credentials."""

import io
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace
from urllib import error, request

import pytest

import standalone_glasses as tool


FAKE_KEY = 'unit-test-only-key-"-$literal-no-real-credential'
FAKE_SERIAL = "UNIT_TEST_GLASSES"


def fake_profile():
    return dict(base_url="https://models.example.invalid/v1", api_key=FAKE_KEY,
                planner="test-planner", monitor="test-monitor", asr_model="test-asr",
                tts_model="test-tts", tts_voice="test-voice")


def invoke(monkeypatch, action, *extra):
    monkeypatch.setattr(sys, "argv", ["standalone_glasses.py", "--serial", FAKE_SERIAL, action, *map(str, extra)])
    tool.main()


@pytest.fixture
def adb_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(tool, "find_adb", lambda: Path("fake-adb.exe"))

    def run(words, **kwargs):
        calls.append((words, kwargs))
        # tee echoes the imported JSON, and adb may write arbitrary diagnostics. Neither is safe stdout.
        stdout = kwargs.get("input") if "tee" in words else b"ordinary adb output"
        return subprocess.CompletedProcess(words, 0, stdout or b"", FAKE_KEY.encode())

    monkeypatch.setattr(tool.subprocess, "run", run)
    return calls


def test_provision_uses_stdin_and_app_private_import_never_shell_arguments(monkeypatch, adb_calls, capsys):
    monkeypatch.setattr(tool, "profile", fake_profile)
    invoke(monkeypatch, "provision")
    imported = [(words, kwargs) for words, kwargs in adb_calls if "tee" in words]
    assert len(imported) == 1
    words, kwargs = imported[0]
    assert words == ["fake-adb.exe", "-s", FAKE_SERIAL, "shell", "run-as", tool.PACKAGE,
                     "tee", "files/standalone-import.json"]
    assert json.loads(kwargs["input"]) == fake_profile()
    assert isinstance(kwargs["input"], bytes)
    assert kwargs["stdout"] is subprocess.PIPE and kwargs["stderr"] is subprocess.PIPE
    assert all(FAKE_KEY not in " ".join(command) for command, _ in adb_calls)
    assert all("/sdcard" not in " ".join(command) and "/storage" not in " ".join(command) for command, _ in adb_calls)
    assert all(not options.get("shell") for _, options in adb_calls)
    output = capsys.readouterr()
    assert FAKE_KEY not in output.out + output.err
    assert "Private configuration delivered" in output.out


def test_provision_delivers_configuration_before_restarting_launcher(monkeypatch, adb_calls):
    monkeypatch.setattr(tool, "profile", fake_profile)
    invoke(monkeypatch, "provision")
    commands = [words[3:] for words, _ in adb_calls]
    assert commands[0] == ["shell", "run-as", tool.PACKAGE, "mkdir", "-p", "files"]
    assert commands[1][0:4] == ["shell", "run-as", tool.PACKAGE, "tee"]
    assert commands[2] == ["shell", "am", "force-stop", tool.PACKAGE]
    assert commands[3] == ["shell", "am", "start", "-a", "android.intent.action.MAIN", "-c",
                           "android.intent.category.LAUNCHER", "-n", tool.PACKAGE + "/.MainActivity"]
    assert all(options["timeout"] == 60 for _, options in adb_calls)


def test_failed_import_does_not_start_app_or_print_adb_echo(monkeypatch, adb_calls, capsys):
    monkeypatch.setattr(tool, "profile", fake_profile)

    def broken(words, **kwargs):
        adb_calls.append((words, kwargs))
        return subprocess.CompletedProcess(words, int("tee" in words), FAKE_KEY.encode(), FAKE_KEY.encode())

    monkeypatch.setattr(tool.subprocess, "run", broken)
    with pytest.raises(RuntimeError) as failure:
        invoke(monkeypatch, "provision")
    assert FAKE_KEY not in str(failure.value)
    assert not any("force-stop" in words or "start" in words for words, _ in adb_calls)
    captured = capsys.readouterr()
    assert FAKE_KEY not in captured.out + captured.err


def write_fake_provider(tmp_path, **changes):
    config_dir = tmp_path / "UserSettings" / "SatoriBDI"
    config_dir.mkdir(parents=True)
    key_path = tmp_path / "synthetic-key.txt"
    key_path.write_text("\ufeff" + FAKE_KEY + "\n", encoding="utf-8")
    config = {"base_url": "https://models.example.invalid/v1/", "key_file": str(key_path),
              "planner": "test-plan", "monitor": "test-watch"}
    config.update(changes)
    (config_dir / "provider.json").write_text(json.dumps(config), encoding="utf-8-sig")


def test_profile_reuses_bom_encoded_private_file_without_output(monkeypatch, tmp_path, capsys):
    write_fake_provider(tmp_path)
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    value = tool.profile()
    assert value["api_key"] == FAKE_KEY
    assert value["base_url"] == "https://models.example.invalid/v1"
    assert value["planner"] == "test-plan" and value["monitor"] == "test-watch"
    assert FAKE_KEY not in capsys.readouterr().out


@pytest.mark.parametrize("url", [
    "http://models.example.invalid/v1",
    "https://user:password@models.example.invalid/v1",
    "https://user@models.example.invalid/v1",
    "https://models.example.invalid/v1?token=not-real",
    "https://models.example.invalid/v1#fragment",
    "https:///no-host",
    "file:///models",
])
def test_profile_rejects_urls_that_would_expose_auth_or_bypass_https(monkeypatch, tmp_path, url):
    write_fake_provider(tmp_path, base_url=url)
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="plain HTTPS"):
        tool.profile()


def test_profile_rejects_empty_key_without_including_credentials_in_error(monkeypatch, tmp_path):
    write_fake_provider(tmp_path)
    (tmp_path / "synthetic-key.txt").write_text(" \n", encoding="utf-8")
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="not configured"):
        tool.profile()


def test_models_disables_proxy_and_redirect_and_only_reports_audio_ids(monkeypatch, adb_calls, capsys):
    monkeypatch.setattr(tool, "profile", fake_profile)
    observations = {}

    class Opener:
        def open(self, req, timeout):
            observations["request"] = req
            observations["timeout"] = timeout
            return io.BytesIO(json.dumps({"data": [{"id": "ordinary-vision"}, {"id": "test-asr"},
                                                  {"id": "test-TTS"}]}).encode())

    def build(*handlers):
        observations["handlers"] = handlers
        return Opener()

    monkeypatch.setattr(tool.request, "build_opener", build)
    invoke(monkeypatch, "models")
    handlers = observations["handlers"]
    assert isinstance(handlers[0], request.ProxyHandler) and handlers[0].proxies == {}
    redirect = next(h for h in handlers if isinstance(h, request.HTTPRedirectHandler))
    assert redirect.redirect_request(None, None, 302, "redirect", {}, "https://elsewhere.invalid") is None
    req = observations["request"]
    assert req.full_url == "https://models.example.invalid/v1/models"
    assert req.get_header("Authorization") == "Bearer " + FAKE_KEY
    assert FAKE_KEY not in req.full_url
    assert observations["timeout"] == 25
    output = capsys.readouterr()
    assert FAKE_KEY not in output.out + output.err
    assert json.loads(output.out) == {"count": 3, "audio_models": ["test-asr", "test-TTS"]}
    assert not adb_calls


@pytest.mark.parametrize("status", [302, 401, 429, 500])
def test_models_http_failure_suppresses_provider_body_and_location(monkeypatch, adb_calls, capsys, status):
    monkeypatch.setattr(tool, "profile", fake_profile)

    class Opener:
        def open(self, req, timeout):
            raise error.HTTPError(req.full_url, status, FAKE_KEY,
                                  {"Location": "https://untrusted.invalid/?key=" + FAKE_KEY},
                                  io.BytesIO(("provider echo " + FAKE_KEY).encode()))

    monkeypatch.setattr(tool.request, "build_opener", lambda *unused: Opener())
    invoke(monkeypatch, "models")
    output = capsys.readouterr()
    assert json.loads(output.out) == {"status": status, "models_available": False}
    assert FAKE_KEY not in output.out + output.err and "untrusted" not in output.out
    assert not adb_calls


def test_export_archives_only_run_records_not_preferences_or_import(monkeypatch, adb_calls, tmp_path, capsys):
    archive = records_tar()

    def run(words, **kwargs):
        adb_calls.append((words, kwargs))
        assert kwargs["stdout"] is not subprocess.PIPE, "record archives must stream to a real file handle"
        kwargs["stdout"].write(archive)
        return subprocess.CompletedProcess(words, 0, None, b"")

    monkeypatch.setattr(tool.subprocess, "run", run)
    output_dir = tmp_path / "export"
    invoke(monkeypatch, "export", "--output", output_dir)
    assert (output_dir / "standalone-runs.tar").read_bytes() == archive
    assert len(adb_calls) == 1
    words, options = adb_calls[0]
    assert words[3:] == ["exec-out", "run-as", tool.PACKAGE, "tar", "-C", "files", "-cf", "-", "standalone-runs"]
    assert "shared_prefs" not in " ".join(words) and "standalone-import" not in " ".join(words)
    assert options.get("input") is None
    assert options["timeout"] > 60
    assert FAKE_KEY not in capsys.readouterr().out


def test_export_requires_explicit_destination_before_running_adb(monkeypatch, adb_calls, capsys):
    with pytest.raises(SystemExit) as exit_status:
        invoke(monkeypatch, "export")
    assert exit_status.value.code == 2
    assert not adb_calls
    assert "--output" in capsys.readouterr().err


def test_status_prints_only_import_encryption_and_task_summary(monkeypatch, adb_calls, capsys):
    responses = {
        ("ls", "files"): b"standalone-runs\nstandalone-state.json\n",
        ("cat", "shared_prefs/standalone_secure.xml"): ("<map><string name='envelope'>" + FAKE_KEY + "</string></map>").encode(),
        ("cat", "files/standalone-state.json"): json.dumps({"phase": "executing", "paused_reason": None,
                                                            "goal": {"text": "private task text"}}).encode(),
    }

    def run(words, **kwargs):
        adb_calls.append((words, kwargs))
        value = b"1234\n" if "pidof" in words else responses[tuple(words[-2:])]
        return subprocess.CompletedProcess(words, 0, value, b"")

    monkeypatch.setattr(tool.subprocess, "run", run)
    invoke(monkeypatch, "status")
    output = capsys.readouterr()
    assert json.loads(output.out) == dict(import_deleted=True, encrypted_config=True, phase="executing", paused_reason=None, pid="1234")
    assert FAKE_KEY not in output.out + output.err and "private task text" not in output.out


def write_standalone_provider(tmp_path, **changes):
    directory = tmp_path / "UserSettings" / "SatoriStandalone"
    directory.mkdir(parents=True)
    value = dict(base_url="https://standalone.example.invalid/v1", planner="standalone-plan", monitor="standalone-watch")
    value.update(changes)
    path = directory / "provider.json"
    path.write_text(json.dumps(value), encoding="utf-8-sig")
    return path


def test_standalone_provider_takes_precedence_and_dpapi_store_wins_over_key_file(monkeypatch, tmp_path, capsys):
    write_fake_provider(tmp_path)
    store = tmp_path / "fake-standalone-key.dpapi"
    store.write_bytes(b"synthetic DPAPI fixture; never decrypted")
    write_standalone_provider(tmp_path, key_store=str(store), key_file=str(tmp_path / "must-not-read.txt"))
    loads = []

    def fake_load(**kwargs):
        loads.append(kwargs)
        return FAKE_KEY

    monkeypatch.setitem(sys.modules, "satori_credentials", SimpleNamespace(load=fake_load))
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    value = tool.profile()
    assert value["base_url"] == "https://standalone.example.invalid/v1"
    assert value["planner"] == "standalone-plan" and value["monitor"] == "standalone-watch"
    assert value["api_key"] == FAKE_KEY
    assert loads == [{"store": store}]
    output = capsys.readouterr()
    assert FAKE_KEY not in output.out + output.err


def test_bdi_dpapi_store_is_used_when_no_standalone_configuration_exists(monkeypatch, tmp_path):
    store = tmp_path / "fake-bdi-store.dpapi"
    write_fake_provider(tmp_path, key_store=str(store))
    loads = []
    monkeypatch.setitem(sys.modules, "satori_credentials",
                        SimpleNamespace(load=lambda **kwargs: loads.append(kwargs) or FAKE_KEY))
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    value = tool.profile()
    assert value["planner"] == "test-plan"
    assert loads == [{"store": store}]


def test_provider_without_explicit_key_location_uses_default_dpapi_loader(monkeypatch, tmp_path):
    write_standalone_provider(tmp_path)
    loads = []
    monkeypatch.setitem(sys.modules, "satori_credentials",
                        SimpleNamespace(load=lambda **kwargs: loads.append(kwargs) or FAKE_KEY))
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    assert tool.profile()["api_key"] == FAKE_KEY
    assert loads == [{}]


def test_optional_backup_uses_its_own_dpapi_store_without_output(monkeypatch, tmp_path, capsys):
    primary = tmp_path / "primary-fixture.dpapi"
    backup = tmp_path / "backup-fixture.dpapi"
    backup_key = "test-only-backup-key"
    write_standalone_provider(tmp_path, key_store=str(primary), backup_key_store=str(backup))
    loads = []

    def fake_load(**kwargs):
        loads.append(kwargs)
        return {primary: FAKE_KEY, backup: backup_key}[kwargs["store"]]

    monkeypatch.setitem(sys.modules, "satori_credentials", SimpleNamespace(load=fake_load))
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    value = tool.profile()
    assert value["api_key"] == FAKE_KEY and value["backup_api_key"] == backup_key
    assert loads == [{"store": primary}, {"store": backup}]
    output = capsys.readouterr()
    assert FAKE_KEY not in output.out + output.err and backup_key not in output.out + output.err


@pytest.mark.parametrize("backup_key", [None, "", "has space", "has\nnewline", "非ASCII", "x" * 4097, FAKE_KEY])
def test_configured_invalid_backup_is_not_silently_dropped(monkeypatch, tmp_path, backup_key):
    store = tmp_path / "backup-fixture.dpapi"
    write_fake_provider(tmp_path, backup_key_store=str(store))
    loads = []
    monkeypatch.setitem(sys.modules, "satori_credentials", SimpleNamespace(load=lambda **kw: loads.append(kw) or backup_key))
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="backup API key") as failure:
        tool.profile()
    assert loads == [{"store": store}]
    assert FAKE_KEY not in str(failure.value)


@pytest.mark.parametrize("store", [None, "", " ", 123])
def test_explicit_invalid_backup_store_fails_without_loading_it(monkeypatch, tmp_path, store):
    write_fake_provider(tmp_path, backup_key_store=store)
    monkeypatch.setitem(sys.modules, "satori_credentials", SimpleNamespace(load=lambda **kw: pytest.fail("invalid store read")))
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="backup key store"):
        tool.profile()


def test_absent_backup_retains_the_old_profile_contract(monkeypatch, tmp_path):
    write_fake_provider(tmp_path)
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    assert "backup_api_key" not in tool.profile()


def test_backup_is_provisioned_only_through_private_stdin(monkeypatch, adb_calls, capsys):
    backup = "test-only-backup-key"
    monkeypatch.setattr(tool, "profile", lambda: dict(fake_profile(), backup_api_key=backup))
    invoke(monkeypatch, "provision")
    imports = [(words, kwargs) for words, kwargs in adb_calls if "tee" in words]
    assert len(imports) == 1
    assert json.loads(imports[0][1]["input"])["backup_api_key"] == backup
    assert all(backup not in " ".join(words) for words, _ in adb_calls)
    output = capsys.readouterr()
    assert backup not in output.out + output.err


def test_standalone_key_file_fallback_does_not_accidentally_load_old_bdi_store(monkeypatch, tmp_path):
    write_fake_provider(tmp_path)
    key_file = tmp_path / "synthetic-standalone-key.txt"
    key_file.write_text("test-standalone-only-key\n", encoding="utf-8")
    write_standalone_provider(tmp_path, key_file=str(key_file))

    def unexpected_load(**unused):
        pytest.fail("Explicit standalone key_file must not consult an unrelated DPAPI store")

    monkeypatch.setitem(sys.modules, "satori_credentials", SimpleNamespace(load=unexpected_load))
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    assert tool.profile()["api_key"] == "test-standalone-only-key"


def test_empty_selected_dpapi_key_does_not_fall_back_to_another_account(monkeypatch, tmp_path, capsys):
    write_fake_provider(tmp_path)
    store = tmp_path / "fake-empty-store.dpapi"
    write_standalone_provider(tmp_path, key_store=str(store))
    loads = []
    monkeypatch.setitem(sys.modules, "satori_credentials",
                        SimpleNamespace(load=lambda **kwargs: loads.append(kwargs) or ""))
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="not configured"):
        tool.profile()
    assert loads == [{"store": store}]
    assert FAKE_KEY not in capsys.readouterr().out


def test_corrupt_standalone_config_is_reported_instead_of_silently_switching_provider(monkeypatch, tmp_path):
    write_fake_provider(tmp_path)
    path = write_standalone_provider(tmp_path)
    path.write_text("{broken JSON", encoding="utf-8")
    loads = []
    monkeypatch.setitem(sys.modules, "satori_credentials",
                        SimpleNamespace(load=lambda **kwargs: loads.append(kwargs) or FAKE_KEY))
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    with pytest.raises(json.JSONDecodeError):
        tool.profile()
    assert not loads


VIDEO_NAME = "recording-20261106T010203-001.mp4"
SIDECAR_NAME = VIDEO_NAME.removesuffix(".mp4") + ".json"


class VideoAdb:
    """Fake only the authorized ADB process boundary; payload stdout must be a real writable file."""
    def __init__(self, monkeypatch, video=b"synthetic-mp4-fixture", sidecar=b'{"complete":true}'):
        self.sources = {VIDEO_NAME: video}
        if sidecar is not None:
            self.sources[SIDECAR_NAME] = sidecar
        self.calls = []
        self.streamed = []
        self.info_calls = {}
        self.listing = None
        self.corrupt = {}
        self.fail_stream = {}
        self.changed_after_copy = set()
        monkeypatch.setattr(tool, "find_adb", lambda: Path("fake-adb.exe"))
        monkeypatch.setattr(tool.subprocess, "run", self.run)

    def run(self, words, **kwargs):
        self.calls.append((words, kwargs))
        assert words[:8] == ["fake-adb.exe", "-s", FAKE_SERIAL, "exec-out", "run-as", tool.PACKAGE, "sh", "-c"]
        assert len(words) == 9
        assert "input" not in kwargs, "raw exec-out must receive the whole script via one -c argument"
        assert kwargs.get("shell") is not True
        script = words[8]
        assert isinstance(script, str)
        assert script.startswith("# satori-video-"), "do not add a second shell-quoting layer"
        assert FAKE_KEY not in script and FAKE_KEY not in " ".join(words)
        if "# satori-video-list-v1" in script:
            listing = self.listing
            if listing is None:
                listing = f"video\t{VIDEO_NAME}\t{len(self.sources[VIDEO_NAME])}\n".encode()
                if SIDECAR_NAME in self.sources:
                    listing += f"sidecar\t{SIDECAR_NAME}\t{len(self.sources[SIDECAR_NAME])}\n".encode()
            return subprocess.CompletedProcess(words, 0, listing + b"done\tvideo-list-v1\n", b"")
        import re
        name = re.search(r"export_name='([^']+)'", script).group(1)
        assert "opened=$(readlink /proc/self/fd/0 <&3)" in script
        if "# satori-video-info-v1" in script:
            assert script.count("stat -L -c %s /proc/self/fd/0 <&3") == 2
            assert "digest=$(sha256sum <&3)" in script
            self.info_calls[name] = self.info_calls.get(name, 0) + 1
            value = self.sources[name]
            if name in self.changed_after_copy and self.info_calls[name] > 1:
                value += b"changed"
            data = f"{len(value)}\t{hashlib.sha256(value).hexdigest()}\n".encode()
            return subprocess.CompletedProcess(words, 0, data, b"")
        assert "# satori-video-read-v1" in script
        assert "exec cat <&3" in script
        assert kwargs["stdout"] is not subprocess.PIPE, "the video must never be captured into RAM"
        assert hasattr(kwargs["stdout"], "write")
        self.streamed.append(name)
        value = self.corrupt.get(name, self.sources[name])
        sink = kwargs["stdout"]
        for offset in range(0, len(value), 8192):
            sink.write(value[offset:offset + 8192])
            if name in self.fail_stream:
                raise self.fail_stream[name]
        return subprocess.CompletedProcess(words, 0, None, b"")


def test_list_videos_reports_only_final_video_and_sidecar_metadata(monkeypatch, capsys):
    adb = VideoAdb(monkeypatch)
    invoke(monkeypatch, "list-videos")
    result = json.loads(capsys.readouterr().out)
    assert result == {"videos": [{"file": VIDEO_NAME, "bytes": len(adb.sources[VIDEO_NAME]),
                                  "sidecar": {"file": SIDECAR_NAME, "bytes": len(adb.sources[SIDECAR_NAME])}}]}
    assert not adb.streamed
    assert not adb.info_calls, "listing sizes must not hash or download every video"


def test_video_export_streams_each_file_and_verifies_both_source_and_local_hashes(monkeypatch, tmp_path, capsys):
    adb = VideoAdb(monkeypatch, video=b"synthetic-video-block" * 160_000)
    # Production export is forbidden from reading an entire video through Path.read_bytes().
    monkeypatch.setattr(Path, "read_bytes", lambda *_: pytest.fail("export must hash files in bounded chunks"))
    invoke(monkeypatch, "export-videos", "--output", tmp_path / "videos")
    for name, value in adb.sources.items():
        with (tmp_path / "videos" / name).open("rb") as handle:
            assert hashlib.sha256(handle.read()).digest() == hashlib.sha256(value).digest()
        assert not (tmp_path / "videos" / (name + ".part")).exists()
        assert adb.info_calls[name] == 2
    assert adb.streamed == [VIDEO_NAME, SIDECAR_NAME]
    stream_calls = [options for _, options in adb.calls if options["stdout"] is not subprocess.PIPE]
    assert stream_calls and all(options["timeout"] > 60 for options in stream_calls)
    output = capsys.readouterr()
    assert VIDEO_NAME in output.out and SIDECAR_NAME in output.out
    assert FAKE_KEY not in output.out + output.err
    assert not any("rm" in words or "delete" in words for words, _ in adb.calls)


def test_video_export_requires_explicit_destination_before_any_adb_operation(monkeypatch, capsys):
    adb = VideoAdb(monkeypatch)
    with pytest.raises(SystemExit) as failure:
        invoke(monkeypatch, "export-videos")
    assert failure.value.code == 2
    assert not adb.calls
    assert "--output" in capsys.readouterr().err


@pytest.mark.parametrize("name", [
    "../recording-escape.mp4", "recording-a/../../escape.mp4", "recording-a;echo-secret.mp4",
    "recording-a\\escape.mp4", "recording-a.mp4\x00", "standalone-import.json",
])
def test_video_listing_rejects_unsafe_names_before_any_payload_read(monkeypatch, name):
    adb = VideoAdb(monkeypatch)
    adb.listing = f"video\t{name}\t12\n".encode()
    with pytest.raises((ValueError, RuntimeError)):
        invoke(monkeypatch, "list-videos")
    assert not adb.streamed


def test_video_listing_does_not_offer_pending_or_failed_recordings(monkeypatch, capsys):
    adb = VideoAdb(monkeypatch)
    adb.listing = (f"video\t{VIDEO_NAME}\t21\n"
                   "video\trecording-test.pending.mp4\t5\n"
                   "video\trecording-test.failed.mp4\t8\n").encode()
    invoke(monkeypatch, "list-videos")
    assert [item["file"] for item in json.loads(capsys.readouterr().out)["videos"]] == [VIDEO_NAME]
    assert not adb.streamed


def test_video_listing_rejects_remote_symlink_failure_without_printing_adb_output(monkeypatch, capsys):
    adb = VideoAdb(monkeypatch)

    def unsafe(words, **kwargs):
        adb.calls.append((words, kwargs))
        return subprocess.CompletedProcess(words, 3, FAKE_KEY.encode(), FAKE_KEY.encode())

    monkeypatch.setattr(tool.subprocess, "run", unsafe)
    with pytest.raises(RuntimeError):
        invoke(monkeypatch, "list-videos")
    captured = capsys.readouterr()
    assert FAKE_KEY not in captured.out + captured.err


def test_existing_identical_video_is_skipped_only_after_real_size_and_hash_check(monkeypatch, tmp_path, capsys):
    adb = VideoAdb(monkeypatch, sidecar=None)
    target = tmp_path / VIDEO_NAME
    target.write_bytes(adb.sources[VIDEO_NAME])
    invoke(monkeypatch, "export-videos", "--output", tmp_path)
    assert adb.info_calls[VIDEO_NAME] >= 1
    assert not adb.streamed
    assert target.read_bytes() == adb.sources[VIDEO_NAME]
    assert "skip" in capsys.readouterr().out.lower()


def test_existing_different_video_is_never_overwritten(monkeypatch, tmp_path):
    adb = VideoAdb(monkeypatch, sidecar=None)
    target = tmp_path / VIDEO_NAME
    target.write_bytes(b"original-user-file")
    with pytest.raises(FileExistsError):
        invoke(monkeypatch, "export-videos", "--output", tmp_path)
    assert target.read_bytes() == b"original-user-file"
    assert not adb.streamed


def test_existing_part_file_is_preserved_for_manual_inspection(monkeypatch, tmp_path):
    adb = VideoAdb(monkeypatch, sidecar=None)
    partial = tmp_path / (VIDEO_NAME + ".part")
    partial.write_bytes(b"previous-incomplete-copy")
    with pytest.raises(FileExistsError):
        invoke(monkeypatch, "export-videos", "--output", tmp_path)
    assert partial.read_bytes() == b"previous-incomplete-copy"
    assert not adb.streamed


def test_corrupt_copy_never_becomes_a_final_video_and_keeps_the_partial(monkeypatch, tmp_path):
    adb = VideoAdb(monkeypatch, sidecar=None)
    adb.corrupt[VIDEO_NAME] = b"corrupted-stream"
    with pytest.raises(RuntimeError):
        invoke(monkeypatch, "export-videos", "--output", tmp_path)
    assert not (tmp_path / VIDEO_NAME).exists()
    assert (tmp_path / (VIDEO_NAME + ".part")).read_bytes() == b"corrupted-stream"
    assert adb.sources[VIDEO_NAME] == b"synthetic-mp4-fixture"


def test_source_changed_during_copy_cannot_be_reported_as_verified(monkeypatch, tmp_path):
    adb = VideoAdb(monkeypatch, sidecar=None)
    adb.changed_after_copy.add(VIDEO_NAME)
    with pytest.raises(RuntimeError):
        invoke(monkeypatch, "export-videos", "--output", tmp_path)
    assert not (tmp_path / VIDEO_NAME).exists()
    assert (tmp_path / (VIDEO_NAME + ".part")).exists()
    assert adb.info_calls[VIDEO_NAME] == 2


def test_transfer_timeout_keeps_part_and_does_not_claim_success(monkeypatch, tmp_path, capsys):
    adb = VideoAdb(monkeypatch, sidecar=None)
    adb.fail_stream[VIDEO_NAME] = subprocess.TimeoutExpired(["fake-adb"], 180)
    with pytest.raises((subprocess.TimeoutExpired, RuntimeError)):
        invoke(monkeypatch, "export-videos", "--output", tmp_path)
    assert not (tmp_path / VIDEO_NAME).exists()
    assert (tmp_path / (VIDEO_NAME + ".part")).exists()
    assert "verified" not in capsys.readouterr().out.lower()


def test_empty_video_directory_exports_nothing_without_loading_a_profile(monkeypatch, tmp_path, capsys):
    adb = VideoAdb(monkeypatch)
    adb.listing = b""
    monkeypatch.setattr(tool, "profile", lambda: pytest.fail("video export must never load API credentials"))
    invoke(monkeypatch, "export-videos", "--output", tmp_path)
    result = json.loads(capsys.readouterr().out)
    assert result["videos"] == result["copied"] == result["skipped"] == 0
    assert not adb.info_calls and not adb.streamed
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("listing", [
    f"video\t{VIDEO_NAME}\t21\nvideo\t{VIDEO_NAME}\t21\n",
    f"sidecar\t{SIDECAR_NAME}\t10\n",
    f"symlink\t{VIDEO_NAME}\t21\n",
    f"video\t{VIDEO_NAME}\t-1\n",
    f"video\t{VIDEO_NAME}\t0\n",
])
def test_inconsistent_video_inventory_cannot_trigger_download(monkeypatch, tmp_path, listing):
    adb = VideoAdb(monkeypatch)
    adb.listing = listing.encode()
    with pytest.raises(ValueError):
        invoke(monkeypatch, "export-videos", "--output", tmp_path)
    assert not adb.streamed and not adb.info_calls
    assert not list(tmp_path.iterdir())


def test_video_info_rejects_private_paths_before_issuing_adb(monkeypatch):
    adb = VideoAdb(monkeypatch)
    with pytest.raises(ValueError):
        tool._remote_video_info("fake-adb.exe", FAKE_SERIAL, "../shared_prefs/standalone_secure.xml", timeout=180)
    assert not adb.calls


def test_malformed_remote_digest_is_not_printed_or_used_to_skip_existing_file(monkeypatch, tmp_path, capsys):
    adb = VideoAdb(monkeypatch, sidecar=None)

    def bad_digest(words, **kwargs):
        if "# satori-video-info-v1" in words[-1]:
            return subprocess.CompletedProcess(words, 0, ("21\t" + FAKE_KEY + "\n").encode(), b"")
        return adb.run(words, **kwargs)

    monkeypatch.setattr(tool.subprocess, "run", bad_digest)
    with pytest.raises(ValueError):
        invoke(monkeypatch, "export-videos", "--output", tmp_path)
    assert not adb.streamed and not list(tmp_path.iterdir())
    output = capsys.readouterr()
    assert FAKE_KEY not in output.out + output.err


def test_local_target_reported_as_symlink_is_never_opened_for_export(monkeypatch, tmp_path):
    adb = VideoAdb(monkeypatch, sidecar=None)
    target = tmp_path / VIDEO_NAME
    target.write_bytes(b"unrelated-user-data")
    original = tool._is_link
    monkeypatch.setattr(tool, "_is_link", lambda path: Path(path) == target or original(path))
    with pytest.raises(ValueError, match="symbolic"):
        invoke(monkeypatch, "export-videos", "--output", tmp_path)
    assert target.read_bytes() == b"unrelated-user-data"
    assert not adb.streamed


def test_finalization_cannot_overwrite_a_file_created_during_transfer(monkeypatch, tmp_path, capsys):
    adb = VideoAdb(monkeypatch, sidecar=None)
    target = tmp_path / VIDEO_NAME

    def competing_writer(words, **kwargs):
        result = adb.run(words, **kwargs)
        if kwargs["stdout"] is not subprocess.PIPE:
            target.write_bytes(b"new-file-from-another-writer")
        return result

    monkeypatch.setattr(tool.subprocess, "run", competing_writer)
    with pytest.raises(FileExistsError):
        invoke(monkeypatch, "export-videos", "--output", tmp_path)
    assert target.read_bytes() == b"new-file-from-another-writer"
    assert (tmp_path / (VIDEO_NAME + ".part")).read_bytes() == adb.sources[VIDEO_NAME]
    assert "Verified " + VIDEO_NAME not in capsys.readouterr().out


def test_adb_zero_exit_without_remote_inventory_completion_is_not_an_empty_success(monkeypatch):
    VideoAdb(monkeypatch)
    # Older/raw transports may lose the remote exit status. A failed sh before its footer
    # must not become a successful empty inventory merely because the local adb client returned 0.
    monkeypatch.setattr(tool.subprocess, "run", lambda words, **kwargs:
                        subprocess.CompletedProcess(words, 0, b"", b"remote failure"))
    with pytest.raises(RuntimeError):
        invoke(monkeypatch, "list-videos")


def records_tar(name="standalone-runs/run-123/events.jsonl", data=b'{"synthetic":true}\n', symlink=False):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        root = tarfile.TarInfo("standalone-runs")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        item = tarfile.TarInfo(name)
        if symlink:
            item.type = tarfile.SYMTYPE
            item.linkname = "../../shared_prefs/standalone_secure.xml"
            archive.addfile(item)
        else:
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
    return buffer.getvalue()


@pytest.mark.parametrize("archive", [records_tar("../shared_prefs/secret.xml"),
    records_tar("standalone-runs/run-123/link", symlink=True), records_tar()[:700], records_tar() + b"unexpected-tail"],
    ids=["path-escape", "symlink", "truncated", "trailing-data"])
def test_streamed_records_tar_rejects_unsafe_or_incomplete_archives(monkeypatch, adb_calls, tmp_path, archive):
    def run(words, **kwargs):
        adb_calls.append((words, kwargs))
        kwargs["stdout"].write(archive)
        return subprocess.CompletedProcess(words, 0, None, b"")
    monkeypatch.setattr(tool.subprocess, "run", run)
    with pytest.raises(RuntimeError):
        invoke(monkeypatch, "export", "--output", tmp_path)
    assert not (tmp_path / "standalone-runs.tar").exists()
    assert (tmp_path / "standalone-runs.tar.part").exists()


def test_records_export_never_overwrites_a_different_existing_archive(monkeypatch, adb_calls, tmp_path):
    original = records_tar(data=b"original archive")
    target = tmp_path / "standalone-runs.tar"
    target.write_bytes(original)
    archive = records_tar(data=b"new archive")
    def run(words, **kwargs):
        kwargs["stdout"].write(archive)
        return subprocess.CompletedProcess(words, 0, None, b"")
    monkeypatch.setattr(tool.subprocess, "run", run)
    with pytest.raises(FileExistsError):
        invoke(monkeypatch, "export", "--output", tmp_path)
    assert target.read_bytes() == original
    assert (tmp_path / "standalone-runs.tar.part").read_bytes() == archive


def test_records_export_identical_archive_is_verified_then_skipped(monkeypatch, adb_calls, tmp_path, capsys):
    archive = records_tar()
    target = tmp_path / "standalone-runs.tar"
    target.write_bytes(archive)
    def run(words, **kwargs):
        kwargs["stdout"].write(archive)
        return subprocess.CompletedProcess(words, 0, None, b"")
    monkeypatch.setattr(tool.subprocess, "run", run)
    invoke(monkeypatch, "export", "--output", tmp_path)
    assert target.read_bytes() == archive
    assert not (tmp_path / "standalone-runs.tar.part").exists()
    assert "Skip identical" in capsys.readouterr().out


def test_records_export_failure_keeps_part_and_hides_adb_diagnostics(monkeypatch, adb_calls, tmp_path, capsys):
    def run(words, **kwargs):
        kwargs["stdout"].write(b"partial")
        return subprocess.CompletedProcess(words, 1, None, FAKE_KEY.encode())
    monkeypatch.setattr(tool.subprocess, "run", run)
    with pytest.raises(RuntimeError):
        invoke(monkeypatch, "export", "--output", tmp_path)
    assert (tmp_path / "standalone-runs.tar.part").read_bytes() == b"partial"
    assert not (tmp_path / "standalone-runs.tar").exists()
    captured = capsys.readouterr()
    assert FAKE_KEY not in captured.out + captured.err
