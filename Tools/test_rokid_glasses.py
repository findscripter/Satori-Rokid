import os

import pytest

import rokid_glasses
import satori_credentials


def test_adb_device_list_keeps_state_and_metadata():
    text = ("* daemon started successfully *\nList of devices attached\n"
            "RG-TEST device product:unit model:RG\nSECOND unauthorized\n")
    assert rokid_glasses.parse_devices(text) == [
        {"serial": "RG-TEST", "state": "device", "product": "unit", "model": "RG"},
        {"serial": "SECOND", "state": "unauthorized"},
    ]


def test_adb_sdk_and_explicit_override(monkeypatch, tmp_path):
    monkeypatch.delenv("SATORI_ADB", raising=False)
    monkeypatch.setenv("ANDROID_HOME", str(tmp_path))
    monkeypatch.setattr(rokid_glasses.shutil, "which", lambda _: None)

    def no_running_adb(*args, **kwargs):
        raise OSError("PowerShell unavailable")

    monkeypatch.setattr(rokid_glasses.subprocess, "run", no_running_adb)
    executable = tmp_path / "platform-tools" / ("adb.exe" if os.name == "nt" else "adb")
    executable.parent.mkdir()
    executable.touch()
    assert rokid_glasses.find_adb() == executable
    override = tmp_path / "other-adb"
    monkeypatch.setenv("SATORI_ADB", str(override))
    assert rokid_glasses.find_adb() == override


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI")
def test_configured_store_round_trips_without_plaintext(tmp_path):
    store = tmp_path / "provider.dpapi"
    token = "fixture-only-model-provider-key"
    assert satori_credentials.load(store) is None
    satori_credentials.save(token, store)
    assert token.encode("ascii") not in store.read_bytes()
    assert satori_credentials.load(store) == token
