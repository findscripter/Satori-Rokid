"""Preview or apply runtime task tool settings without touching model credentials.

Default is read-only. --apply requires a new evidence directory and restarts the
selected app only after recording checks and a backup of its persisted task.
Web discovery is the default; a remote manual API and its manual ID must be explicit.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib import error, parse, request

from rokid_glasses import find_adb, parse_devices


PACKAGE = "com.satori.glasses.capture"
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
ACTIVE_RECORDING = {"starting", "recording", "finalizing"}
TOOLS_IMPORT = "files/standalone-tools-import.json"


class ConfigurationError(ValueError):
    """Only fixed messages from this helper are safe to show to the operator."""


def normalize_endpoint(raw):
    if not isinstance(raw, str) or not 1 <= len(raw) <= 2048 or any(c.isspace() or ord(c) < 32 or ord(c) == 127 or c == "\\" for c in raw):
        raise ConfigurationError("Invalid endpoint")
    value = raw.rstrip("/")
    uri = parse.urlsplit(value)
    if uri.scheme not in {"https", "http"} or not uri.hostname or uri.username is not None or uri.password is not None or "?" in value or "#" in value:
        raise ConfigurationError("Use a credential-free HTTPS endpoint or explicit HTTP loopback")
    if uri.scheme == "http" and uri.hostname not in {"127.0.0.1", "localhost"}:
        raise ConfigurationError("Unencrypted endpoints are only allowed on loopback")
    if uri.port is not None and not 1 <= uri.port <= 65535:
        raise ConfigurationError("Invalid endpoint port")
    return value


def label(value):
    if value is not None and (not isinstance(value, str) or not ID.fullmatch(value)):
        raise ConfigurationError("Trial labels must be short ASCII identifiers, not names or paths")
    return value


def _json_object(raw, limit=2 * 1024 * 1024):
    if not 1 <= len(raw) <= limit:
        raise ValueError("Metadata size is invalid")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Metadata must be a JSON object")
    return value


class Device:
    def __init__(self, adb, serial):
        self.adb, self.serial = str(adb), serial

    def run(self, *words, data=None, optional=False, global_command=False):
        command = [self.adb] + ([] if global_command else ["-s", self.serial]) + list(words)
        try:
            result = subprocess.run(command, **({"input": data} if data is not None else {}), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.TimeoutExpired):
            raise RuntimeError("ADB operation failed; diagnostics suppressed") from None
        if result.returncode:
            if optional and result.returncode == 1:
                return None
            raise RuntimeError("ADB operation failed; diagnostics suppressed")
        if data is not None:
            return b""  # run-as tee echoes the entire private import, so never return it.
        if len(result.stdout) > 2 * 1024 * 1024:
            raise RuntimeError("Unexpectedly large device metadata")
        return result.stdout

    def files(self):
        raw = self.run("shell", "run-as", PACKAGE, "ls", "files")
        return set(raw.decode("utf-8").splitlines())

    def state(self, files):
        if "standalone-state.json" not in files:
            return None, None
        raw = self.run("shell", "run-as", PACKAGE, "cat", "files/standalone-state.json")
        return raw, _json_object(raw)

    def safe_recording_status(self, state):
        pid = self.run("shell", "pidof", PACKAGE, optional=True)
        if pid is None or not pid.strip():
            return {"app_running": False, "recording_check": "no_app_process"}
        if state is None or not re.fullmatch(r"run-[0-9]+", state.get("local_run", "")):
            raise ConfigurationError("Running app has no verifiable recording status; finish recording and close it first")
        raw = self.run("shell", "run-as", PACKAGE, "tail", "-n", "1", f"files/standalone-runs/{state['local_run']}/status.jsonl")
        status = _json_object(raw)
        uptime = float(self.run("shell", "cat", "/proc/uptime").decode("ascii").split()[0])
        elapsed = status.get("elapsed_ns")
        if type(elapsed) not in (int, float) or not -1 <= uptime - elapsed / 1e9 <= 10:
            raise ConfigurationError("Recording telemetry is stale; refusing to interrupt the app")
        for role in ("video", "screen"):
            recording = status.get(role)
            if not isinstance(recording, dict) or not isinstance(recording.get("state"), str):
                raise ConfigurationError("Recording telemetry is incomplete; close the app after finishing recordings")
            if recording["state"] in ACTIVE_RECORDING:
                raise ConfigurationError("An active recording must finish before configuration")
        if status.get("voice_busy") is True:
            raise ConfigurationError("Manual voice interaction is active; finish it before configuration")
        return status


def probe_service(endpoint, token, manual_id):
    class NoRedirect(request.HTTPRedirectHandler):
        def redirect_request(self, *args):
            return None

    opener = request.build_opener(request.ProxyHandler({}), NoRedirect())
    def fetch(path, body=None):
        req = request.Request(endpoint + path, data=body, headers={
            "Authorization": "Bearer " + token, "Content-Type": "application/json; charset=utf-8"})
        with opener.open(req, timeout=3) as response:
            return _json_object(response.read(1_048_577), 1_048_576)
    try:
        health = fetch("/health")
        if health.get("ok") is not True or health.get("service") != "satori-manual-tools":
            return {"status": "failed", "reason": "unexpected_service"}
        result = fetch("/tools/call", b'{"name":"list_manuals","arguments":{}}')
        rows = result.get("result", {}).get("manuals", [])
        if result.get("ok") is not True or not isinstance(rows, list):
            return {"status": "failed", "reason": "invalid_catalog"}
        if sum(isinstance(row, dict) and row.get("manual_id") == manual_id for row in rows) != 1:
            return {"status": "failed", "reason": "manual_not_found"}
        return {"status": "verified", "manual_id": manual_id}
    except error.HTTPError as failure:
        return {"status": "failed", "reason": f"http_{failure.code}"}
    except (OSError, ValueError, error.URLError, AttributeError):
        return {"status": "failed", "reason": "unavailable_or_invalid_response"}


def _write(path, value):
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")


def _reverse_action(device, endpoint, enabled):
    if not enabled:
        return None, "disabled"
    uri = parse.urlsplit(endpoint)
    if uri.hostname not in {"127.0.0.1", "localhost"} or uri.port is None or ":" in device.serial or device.serial.startswith("emulator-"):
        raise ConfigurationError("USB reverse requires an explicit loopback port and a physical USB serial")
    target = f"tcp:{uri.port}"
    lines = device.run("reverse", "--list").decode("ascii").splitlines()
    matches = [parts for line in lines if len(parts := line.split()) == 3 and parts[1] == target]
    if any(parts[2] != target for parts in matches):
        raise ConfigurationError("The requested reverse port already maps elsewhere; it was left unchanged")
    return target, "already_configured" if matches else "create"


def _verify_import(device, labels, manual_id, mode, previous, manual_source):
    for attempt in range(20):
        files = device.files()
        if "standalone-tools-import.json" not in files:
            _, state = device.state(files)
            run = state.get("local_run", "") if state else ""
            if re.fullmatch(r"run-[0-9]+", run) and (previous is None or run != previous.get("local_run")):
                metadata = _json_object(device.run("shell", "run-as", PACKAGE, "cat", f"files/standalone-runs/{run}/experiment.json"))
                if metadata.get("metadata_valid") is not True or metadata.get("manual_id") != manual_id or metadata.get("condition") != f"demo-{mode}" or metadata.get("manual_source") != manual_source:
                    raise ConfigurationError("App did not accept the requested tool configuration")
                for field in ("protocol_id", "participant_code", "trial_label", "task_variant"):
                    if metadata.get(field) != labels[field]:
                        raise ConfigurationError("App did not accept the requested trial labels")
                if previous is not None and (state.get("task_instance_id") != previous.get("task_instance_id") or state.get("goal") != previous.get("goal")):
                    raise ConfigurationError("Task state changed during configuration; inspect the preserved backup")
                return {"verified": True, "run": run, "task_preserved": previous is not None,
                        "paused_reason": state.get("paused_reason"), "import_deleted": True}
        if attempt < 19:
            time.sleep(0.25)
    raise ConfigurationError("Import completion was not verified; inspect the local evidence before retrying")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--manual-source", choices=("remote_api", "web"), default="web")
    parser.add_argument("--endpoint", help="Required only for remote_api")
    parser.add_argument("--manual-id", help="Defaults to web-discovery for web; required for remote_api")
    parser.add_argument("--mode", choices=("guided", "reactive"), default="guided")
    parser.add_argument("--participant-code")
    parser.add_argument("--trial-label", default="runtime")
    parser.add_argument("--task-variant", default="runtime-input")
    parser.add_argument("--token-stdin", action="store_true")
    parser.add_argument("--evidence", type=Path, help="New local directory required by --apply")
    parser.add_argument("--usb-reverse", action="store_true", help="Explicit PC-dependent USB loopback demo")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    imported = False
    restarted = False
    evidence = None
    device = None
    try:
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", args.serial):
            raise ConfigurationError("Invalid explicit device serial")
        endpoint, token = None, None
        if args.manual_source == "remote_api":
            if args.manual_id is None:
                raise ConfigurationError("Remote API requires an explicit --manual-id")
            endpoint = normalize_endpoint(args.endpoint)
            token = sys.stdin.readline(514).rstrip("\r\n") if args.token_stdin else os.environ.get("SATORI_MANUAL_TOOLS_TOKEN", "")
            if not 32 <= len(token) <= 512 or not all(33 <= ord(c) <= 126 for c in token):
                raise ConfigurationError("Provide a separate 32 to 512 character ASCII tools token through environment or stdin")
        elif args.endpoint is not None or args.token_stdin or args.usb_reverse:
            raise ConfigurationError("Endpoint, token stdin and USB reverse require --manual-source remote_api")
        manual_id = label(args.manual_id if args.manual_id is not None else "web-discovery")
        if args.manual_source == "web" and manual_id != "web-discovery":
            raise ConfigurationError("Web discovery requires --manual-id web-discovery")
        labels = {"schema": "satori.experiment-context/1", "protocol_id": "runtime-task-v1",
                  "participant_code": label(args.participant_code), "trial_label": label(args.trial_label), "task_variant": label(args.task_variant)}
        if args.apply and (args.evidence is None or args.evidence.exists()):
            raise ConfigurationError("--apply requires a new --evidence directory; existing data will not be overwritten")
        device = Device(find_adb(), args.serial)
        devices = parse_devices(device.run("devices", "-l", global_command=True).decode("utf-8"))
        if not any(row["serial"] == args.serial and row["state"] == "device" for row in devices):
            raise ConfigurationError("The explicit device is not connected and authorized")
        package = device.run("shell", "pm", "path", PACKAGE).decode("utf-8").strip()
        if not package.startswith("package:/"):
            raise ConfigurationError("Satori is not installed on the selected device")
        files = device.files()
        if {"standalone-tools-import.json", "standalone-import.json"} & files:
            raise ConfigurationError("A pending private configuration import already exists; it was left untouched")
        state_bytes, state = device.state(files)
        status = device.safe_recording_status(state)
        reverse_port, reverse_action = None, "disabled"
        service = {"status": "not_required", "reason": "web_discovery"}
        if args.manual_source == "remote_api":
            reverse_port, reverse_action = _reverse_action(device, endpoint, args.usb_reverse)
            service = probe_service(endpoint, token, manual_id)
        plan = {"mode": "apply" if args.apply else "dry_run", "serial": args.serial, "endpoint": endpoint,
                "manual_id": manual_id, "manual_source": args.manual_source, "assistance_mode": args.mode, "labels": labels,
                "pc_dependent_usb_demo": args.usb_reverse, "usb_reverse_action": reverse_action,
                "service_check": service, "old_task_present": state is not None and state.get("goal") is not None,
                "configured_at_utc": datetime.now(timezone.utc).isoformat()}
        if not args.apply:
            print(json.dumps(plan, ensure_ascii=False))
            return 0
        if args.manual_source == "remote_api" and service.get("status") != "verified":
            raise ConfigurationError("Tool service and manual could not be verified; no configuration was changed")
        evidence = args.evidence.resolve()
        evidence.mkdir(parents=True, exist_ok=False)
        _write(evidence / "plan.json", plan)
        if state_bytes is not None:
            (evidence / "state-before.json").write_bytes(state_bytes)
        _write(evidence / "status-before.json", status)
        if "experiment-context.json" in files:
            previous_labels = device.run("shell", "run-as", PACKAGE, "cat", "files/experiment-context.json")
            _json_object(previous_labels, 4096)
            (evidence / "experiment-context-before.json").write_bytes(previous_labels)
        # Recheck immediately before the restart; no recording is started or stopped by this helper.
        latest_files = device.files()
        if {"standalone-tools-import.json", "standalone-import.json"} & latest_files:
            raise ConfigurationError("Another private import appeared during preflight; it was left untouched")
        current_bytes, current = device.state(latest_files)
        device.safe_recording_status(current)
        if current is not None and state is not None and current.get("task_instance_id") != state.get("task_instance_id"):
            raise ConfigurationError("The task changed during preflight; rerun with a new evidence directory")
        if current_bytes is not None:
            (evidence / "state-pre-restart.json").write_bytes(current_bytes)
        if reverse_action == "create":
            device.run("reverse", "--no-rebind", reverse_port, reverse_port)
        device.run("shell", "am", "force-stop", PACKAGE)
        private = {"manual_source": args.manual_source, "manual_id": manual_id, "assistance_mode": args.mode}
        if args.manual_source == "remote_api":
            private.update(tools_base_url=endpoint, tools_token=token)
        imported = True
        device.run("shell", "run-as", PACKAGE, "tee", TOOLS_IMPORT, data=json.dumps(private, ensure_ascii=False).encode())
        device.run("shell", "run-as", PACKAGE, "tee", "files/experiment-context.json", data=json.dumps(labels, ensure_ascii=False).encode())
        restarted = True
        device.run("shell", "am", "start", "-a", "android.intent.action.MAIN", "-c", "android.intent.category.LAUNCHER",
                   "-n", PACKAGE + "/.MainActivity")
        verification = _verify_import(device, labels, manual_id, args.mode, current, args.manual_source)
        _write(evidence / "result.json", verification)
        print(json.dumps(dict(plan, **verification, evidence=str(evidence)), ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as failure:
        cleanup = None
        if imported and not restarted and device is not None:
            try:
                device.run("shell", "run-as", PACKAGE, "rm", "-f", TOOLS_IMPORT)
                cleanup = True
            except RuntimeError:
                cleanup = False
        if evidence is not None:
            try:
                _write(evidence / "failure.json", {"configuration_verified": False, "launch_attempted": restarted,
                                                   "pending_import_cleanup": cleanup})
            except OSError:
                pass
        reason = str(failure) if isinstance(failure, ConfigurationError) else "Check device/app state, service access and the preserved evidence. Secret diagnostics were suppressed."
        print("Runtime configuration not completed: " + reason, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
