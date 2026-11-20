"""Configuration helper boundary tests: fake ADB, no device or provider secrets."""

import io
import json
from pathlib import Path
import subprocess
import threading

import pytest

import configure_classroom_demo as tool


TOKEN = "test-only-manual-tools-token-0123456789"
SERIAL = "TEST_GLASSES_01"


class AdbFixture:
    def __init__(self):
        self.calls = []
        self.running = True
        self.video = "idle"
        self.screen = "idle"
        self.status_age = 1
        self.fail_tee = False
        self.pending = False
        self.state = {"schema": "satori.standalone/1", "local_run": "run-100", "task_instance_id": "task-old",
                      "goal": {"text": "保留原任务"}, "phase": "executing"}
        self.new_state = dict(self.state, local_run="run-200", paused_reason="user")
        self.launched = False
        self.imported = None
        self.labels = None
        self.reverse = ""

    def run(self, words, **kwargs):
        self.calls.append((words, kwargs))
        command = words[3:] if words[1:3] == ["-s", SERIAL] else words[1:]
        code = 0
        if command == ["devices", "-l"]:
            data = f"List of devices attached\n{SERIAL} device product:glasses model:RG_glasses\n".encode()
        elif command == ["shell", "pm", "path", tool.PACKAGE]:
            data = b"package:/data/app/test/base.apk\n"
        elif command == ["shell", "pidof", tool.PACKAGE]:
            code = 0 if self.running else 1; data = b"1234\n" if self.running else b""
        elif command == ["shell", "cat", "/proc/uptime"]:
            data = b"100.0 90.0\n"
        elif command == ["shell", "run-as", tool.PACKAGE, "ls", "files"]:
            data = b"standalone-state.json\nstandalone-runs\n"
            if self.pending:
                data += b"standalone-tools-import.json\n"
        elif command == ["shell", "run-as", tool.PACKAGE, "cat", "files/standalone-state.json"]:
            data = json.dumps(self.new_state if self.launched else self.state).encode()
        elif command[:5] == ["shell", "run-as", tool.PACKAGE, "tail", "-n"]:
            status = {"schema": "satori.status/1", "elapsed_ns": int((100 - self.status_age) * 1e9),
                      "video": {"state": self.video}, "screen": {"state": self.screen}, "voice_busy": False}
            data = json.dumps(status).encode()
        elif command == ["reverse", "--list"]:
            data = self.reverse.encode()
        elif command[:2] == ["reverse", "--no-rebind"]:
            data = b""
        elif command[:4] == ["shell", "am", "force-stop", tool.PACKAGE]:
            self.running = False; data = b""
        elif command[:4] == ["shell", "run-as", tool.PACKAGE, "tee"]:
            if self.fail_tee:
                return subprocess.CompletedProcess(words, 1, TOKEN.encode(), TOKEN.encode())
            value = json.loads(kwargs["input"])
            if command[4] == "files/standalone-tools-import.json":
                self.imported = value; self.pending = True
            elif command[4] == "files/experiment-context.json":
                self.labels = value
            data = kwargs["input"]
        elif command[:3] == ["shell", "am", "start"]:
            self.launched = True; self.running = True; self.pending = False; data = b"Starting: fixture\n"
        elif command == ["shell", "run-as", tool.PACKAGE, "cat", "files/standalone-runs/run-200/experiment.json"]:
            data = json.dumps(dict(self.labels or {}, schema="satori.experiment/1", metadata_valid=True,
                                   condition="demo-" + self.imported["assistance_mode"], manual_id=self.imported["manual_id"],
                                   manual_source=self.imported["manual_source"])).encode()
        elif command == ["shell", "run-as", tool.PACKAGE, "rm", "-f", "files/standalone-tools-import.json"]:
            self.pending = False; data = b""
        else:
            raise AssertionError(f"Unexpected fake ADB command: {command}")
        return subprocess.CompletedProcess(words, code, data, b"")


@pytest.fixture
def adb(monkeypatch):
    fake = AdbFixture()
    monkeypatch.setenv("SATORI_MANUAL_TOOLS_TOKEN", TOKEN)
    monkeypatch.setattr(tool, "find_adb", lambda: Path("fake-adb.exe"))
    monkeypatch.setattr(tool.subprocess, "run", fake.run)
    monkeypatch.setattr(tool, "probe_service", lambda *args: {"status": "verified", "manual_id": "fixture-device-v1"})
    return fake


def invoke(*extra):
    return tool.main(["--manual-source", "remote_api", "--serial", SERIAL, "--endpoint", "http://127.0.0.1:8766", "--manual-id", "fixture-device-v1", *map(str, extra)])


def test_default_is_read_only_and_redacts_token(adb, tmp_path, capsys):
    output = tmp_path / "not-created"
    assert invoke("--evidence", output, "--usb-reverse") == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "dry_run" and report["pc_dependent_usb_demo"] is True
    assert TOKEN not in json.dumps(report)
    assert not output.exists()
    for words, args in adb.calls:
        assert "tee" not in words and "force-stop" not in words and "start" not in words and "--no-rebind" not in words
        assert "input" not in args


def test_apply_backs_up_original_state_and_only_imports_tools(adb, tmp_path, capsys):
    output = tmp_path / "evidence"
    assert invoke("--apply", "--evidence", output, "--participant-code", "P03", "--trial-label", "pilot-01", "--task-variant", "A") == 0
    assert json.loads((output / "state-before.json").read_text(encoding="utf-8")) == adb.state
    assert adb.imported == {"manual_source": "remote_api", "tools_base_url": "http://127.0.0.1:8766", "tools_token": TOKEN,
                            "manual_id": "fixture-device-v1", "assistance_mode": "guided"}
    assert adb.labels == {"schema": "satori.experiment-context/1", "protocol_id": "runtime-task-v1",
                          "participant_code": "P03", "trial_label": "pilot-01", "task_variant": "A"}
    assert adb.launched
    output_text = capsys.readouterr().out
    assert TOKEN not in output_text
    assert json.loads(output_text)["verified"] is True
    for words, kwargs in adb.calls:
        assert TOKEN not in " ".join(words)
        assert "standalone_secure.xml" not in " ".join(words)
        assert "standalone-import.json" not in " ".join(words)
        assert "clear" not in words
    for path in output.iterdir():
        assert TOKEN not in path.read_text(encoding="utf-8")


def test_web_default_needs_no_endpoint_token_or_pc_service(adb, monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("SATORI_MANUAL_TOOLS_TOKEN")
    monkeypatch.setattr(tool, "probe_service", lambda *args: pytest.fail("web mode must not probe a PC service"))
    monkeypatch.setattr(tool, "_reverse_action", lambda *args: pytest.fail("web mode must not inspect ADB reverse"))
    output = tmp_path / "web-evidence"
    assert tool.main(["--serial", SERIAL, "--apply", "--evidence", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["manual_source"] == "web" and report["pc_dependent_usb_demo"] is False
    assert report["service_check"]["status"] == "not_required" and report["endpoint"] is None
    assert report["service_check"]["reason"] == "web_discovery"
    assert adb.imported == {"manual_source": "web", "manual_id": "web-discovery", "assistance_mode": "guided"}
    assert adb.labels == {"schema": "satori.experiment-context/1", "protocol_id": "runtime-task-v1",
                          "participant_code": None, "trial_label": "runtime", "task_variant": "runtime-input"}
    assert json.loads((output / "state-before.json").read_text(encoding="utf-8")) == adb.state
    assert report["task_preserved"] is True
    assert not any("reverse" in command or "standalone_secure.xml" in command or "standalone-import.json" in command
                   for command, _ in adb.calls)


@pytest.mark.parametrize("extra", [["--usb-reverse"], ["--endpoint", "http://127.0.0.1:8766"], ["--token-stdin"]])
def test_web_rejects_remote_options_before_reading_or_changing_device(adb, extra):
    assert tool.main(["--serial", SERIAL, *extra]) == 2
    assert not adb.calls


def test_remote_mode_still_requires_endpoint_and_token(adb, monkeypatch):
    assert tool.main(["--manual-source", "remote_api", "--serial", SERIAL, "--manual-id", "fixture-device-v1"]) == 2
    monkeypatch.delenv("SATORI_MANUAL_TOOLS_TOKEN")
    assert invoke() == 2
    assert not adb.calls


def test_remote_mode_requires_an_explicit_manual_id_before_device_access(adb):
    assert tool.main(["--manual-source", "remote_api", "--serial", SERIAL, "--endpoint", "http://127.0.0.1:8766"]) == 2
    assert not adb.calls


def test_bundled_device_source_is_not_a_runtime_option(adb):
    with pytest.raises(SystemExit) as failure:
        tool.main(["--serial", SERIAL, "--manual-source", "device"])
    assert failure.value.code == 2 and not adb.calls


def test_import_verifies_web_source_before_claiming_success(adb, monkeypatch, tmp_path, capsys):
    original_run = adb.run

    def old_app(words, **kwargs):
        result = original_run(words, **kwargs)
        if words[-1].endswith("/experiment.json"):
            metadata = json.loads(result.stdout)
            metadata.pop("manual_source")
            return subprocess.CompletedProcess(words, 0, json.dumps(metadata).encode(), b"")
        return result

    monkeypatch.setattr(tool.subprocess, "run", old_app)
    assert tool.main(["--serial", SERIAL, "--apply", "--evidence", str(tmp_path / "evidence")]) == 2
    assert "App did not accept" in capsys.readouterr().err


@pytest.mark.parametrize("track,state", [("video", "starting"), ("video", "recording"), ("video", "finalizing"),
                                         ("screen", "starting"), ("screen", "recording"), ("screen", "finalizing")])
def test_active_capture_refuses_to_interrupt(adb, tmp_path, track, state, capsys):
    setattr(adb, track, state)
    assert invoke("--apply", "--evidence", tmp_path / "evidence") == 2
    assert not any("force-stop" in command or "tee" in command for command, _ in adb.calls)
    assert TOKEN not in capsys.readouterr().err


def test_stale_recording_telemetry_refuses_restart(adb, tmp_path):
    adb.status_age = 60
    assert invoke("--apply", "--evidence", tmp_path / "evidence") == 2
    assert not any("force-stop" in command for command, _ in adb.calls)


def test_pending_import_is_preserved(adb, tmp_path):
    adb.pending = True
    assert invoke("--apply", "--evidence", tmp_path / "evidence") == 2
    assert not any("tee" in command or "rm" in command for command, _ in adb.calls)


def test_apply_requires_new_evidence_directory(adb, tmp_path):
    assert invoke("--apply") == 2
    assert invoke("--apply", "--evidence", tmp_path) == 2
    assert not any("force-stop" in command for command, _ in adb.calls)


def test_usb_reverse_is_explicit_and_never_overwrites_another_mapping(adb, tmp_path):
    assert invoke("--apply", "--usb-reverse", "--evidence", tmp_path / "one") == 0
    mutations = [command for command, _ in adb.calls if "--no-rebind" in command]
    assert mutations == [["fake-adb.exe", "-s", SERIAL, "reverse", "--no-rebind", "tcp:8766", "tcp:8766"]]
    adb.reverse = "UsbFfs tcp:8766 tcp:9999\n"
    before = len(adb.calls)
    assert invoke("--apply", "--usb-reverse", "--evidence", tmp_path / "two") == 2
    assert not any("--no-rebind" in command or "force-stop" in command for command, _ in adb.calls[before:])


def test_failed_private_import_never_echoes_token_or_launches(adb, tmp_path, capsys):
    adb.fail_tee = True
    assert invoke("--apply", "--evidence", tmp_path / "evidence") == 2
    assert not adb.launched
    captured = capsys.readouterr()
    assert TOKEN not in captured.out + captured.err
    assert any("rm" in command for command, _ in adb.calls)


@pytest.mark.parametrize("value", ["http://192.168.1.5:8766", "http://evil.example", "https://u:p@example.com",
                                   "https://example.com?token=secret", "https://example.com#x", "file:///x",
                                   "http://localhost.evil.example:8766", "http://127.0.0.1:99999", "https://example.com/\n"])
def test_endpoint_parser_rejects_unsafe_targets(value):
    with pytest.raises(ValueError):
        tool.normalize_endpoint(value)


def test_secret_from_stdin_is_validated_without_echo(monkeypatch, adb, tmp_path, capsys):
    monkeypatch.delenv("SATORI_MANUAL_TOOLS_TOKEN")
    monkeypatch.setattr(tool.sys, "stdin", io.StringIO(TOKEN + "\n"))
    assert invoke("--token-stdin") == 0
    assert TOKEN not in capsys.readouterr().out
    monkeypatch.setattr(tool.sys, "stdin", io.StringIO("bad\n"))
    assert invoke("--token-stdin", "--apply", "--evidence", tmp_path / "evidence") == 2


@pytest.mark.parametrize("label", ["张三", "P 01", "../secret", "x" * 81])
def test_labels_are_pseudonymous_bounded_identifiers(adb, label):
    assert invoke("--participant-code", label) == 2


def test_service_probe_uses_actual_http_and_checks_selected_manual(tmp_path):
    from satori_repro.manual_tools import ManualStore, create_server
    manual = {"schema_version": 1, "manual_id": "fixture-device-v1", "title": "Test device", "model": "Fixture",
              "version": "1", "source_url": "https://example.invalid/manual", "synthetic": True,
              "sections": [{"section_id": "setup", "title": "Setup", "page": 1, "text": "Connect the test cable."}]}
    (tmp_path / "fixture.json").write_text(json.dumps(manual), encoding="utf-8")
    store = ManualStore(tmp_path)
    server = create_server(store, "127.0.0.1", 0, token=TOKEN)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    try:
        assert tool.probe_service(endpoint, TOKEN, "fixture-device-v1") == {"status": "verified", "manual_id": "fixture-device-v1"}
        assert tool.probe_service(endpoint, TOKEN, "missing")["reason"] == "manual_not_found"
        assert tool.probe_service(endpoint, "wrong-token", "fixture-device-v1")["reason"] == "http_401"
    finally:
        server.shutdown(); server.server_close(); thread.join(2)


def test_web_discovery_preserves_task_without_local_server_or_token(adb, monkeypatch, tmp_path):
    monkeypatch.delenv("SATORI_MANUAL_TOOLS_TOKEN")
    monkeypatch.setattr(tool, "probe_service", lambda *args: pytest.fail("web mode must not require a PC service"))
    output = tmp_path / "web-configuration"
    assert tool.main(["--serial", SERIAL, "--manual-source", "web", "--manual-id", "web-discovery",
                      "--apply", "--evidence", str(output)]) == 0
    assert adb.imported == {"manual_source": "web", "manual_id": "web-discovery", "assistance_mode": "guided"}
    assert not any("reverse" in words for words, _ in adb.calls)
    assert json.loads((output / "state-before.json").read_text(encoding="utf-8")) == adb.state


def test_web_discovery_rejects_a_fixed_manual_before_mutating(adb):
    assert tool.main(["--serial", SERIAL, "--manual-source", "web", "--manual-id", "fixture-device-v1"]) == 2
    assert not adb.calls
