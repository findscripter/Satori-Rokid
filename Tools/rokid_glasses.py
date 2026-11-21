"""ADB discovery and device inventory for the standalone Rokid tools."""

import os
from pathlib import Path
import shutil
import subprocess


def parse_devices(text):
    """Parse adb devices -l output, including unauthorized and offline devices."""
    devices = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 2 or fields == ["List", "of", "devices", "attached"] or fields[0] == "*":
            continue
        item = {"serial": fields[0], "state": fields[1]}
        item.update(field.split(":", 1) for field in fields[2:] if ":" in field)
        devices.append(item)
    return devices


def find_adb():
    if override := os.environ.get("SATORI_ADB"):
        return Path(override)
    # Use the same client as the live server to avoid resetting wireless connections.
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                 "(Get-Process adb -ErrorAction SilentlyContinue | Select-Object -First 1).Path"],
                capture_output=True, text=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW,
            )
            running = result.stdout.strip()
            if running and Path(running).is_file():
                return Path(running)
        except (OSError, subprocess.TimeoutExpired):
            pass
    if command := shutil.which("adb"):
        return Path(command)
    for variable in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        if sdk := os.environ.get(variable):
            path = Path(sdk) / "platform-tools" / ("adb.exe" if os.name == "nt" else "adb")
            if path.is_file():
                return path
    raise RuntimeError("Install Android Platform Tools or set SATORI_ADB to the adb executable")
