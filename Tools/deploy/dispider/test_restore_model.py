"""CPU-only restoration safety checks with tiny synthetic artifacts."""
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

if sys.platform != "linux":
    pytest.skip("Dispider RAM restoration requires Linux fcntl and POSIX private-file permissions; see remote Linux evidence", allow_module_level=True)
zstandard = pytest.importorskip("zstandard", reason="Dispider lossless restoration tests require zstandard")

import restore_model as restoration


def tiny_snapshot(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    ram = tmp_path / "ram"
    ram.mkdir(mode=0o700)
    metadata_root = project / "model-metadata"
    archive_root = project / "model-archive"
    archive_root.mkdir()
    entries = []
    for name in sorted(restoration.EXPECTED_FILES):
        data = ("synthetic artifact " + name).encode() * 3
        item = {"rfilename": name, "size": len(data)}
        if name in restoration.WEIGHTS:
            item["lfs"] = {"sha256": hashlib.sha256(data).hexdigest()}
            (archive_root / (name + ".zst")).write_bytes(zstandard.ZstdCompressor().compress(data))
        else:
            target = metadata_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            item["blobId"] = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        entries.append(item)
    manifest = {"sha": restoration.REVISION, "id": restoration.REPOSITORY, "siblings": entries}
    (project / "model-metadata.json").write_text(json.dumps(manifest))
    original_private = restoration.private_root
    monkeypatch.setattr(restoration, "private_root", lambda value: original_private(value, ram))
    return project, ram, manifest


def test_full_restore_then_hot_cache_verification(tmp_path, monkeypatch):
    project, ram, _ = tiny_snapshot(tmp_path, monkeypatch)
    first = restoration.restore(project, ram)
    assert first["status"] == "completed"
    assert len(first["files"]) == 24
    assert all(item["verified"] and not item["reused_hot_cache"] for item in first["files"])
    second = restoration.restore(project, ram)
    assert all(item["reused_hot_cache"] for item in second["files"])
    assert not list(ram.rglob("*.restore-*"))


def test_bad_hot_weight_is_rejected_without_replacing_it(tmp_path, monkeypatch):
    project, ram, _ = tiny_snapshot(tmp_path, monkeypatch)
    restoration.restore(project, ram)
    target = ram / sorted(restoration.WEIGHTS)[0]
    bad = b"x" * target.stat().st_size
    target.write_bytes(bad)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        restoration.restore(project, ram)
    assert target.read_bytes() == bad


@pytest.mark.parametrize("suffix", [".incomplete", ".upload"])
def test_existing_download_partial_is_not_touched(tmp_path, monkeypatch, suffix):
    project, ram, _ = tiny_snapshot(tmp_path, monkeypatch)
    partial = ram / (sorted(restoration.WEIGHTS)[0] + suffix)
    partial.write_bytes(b"still downloading")
    with pytest.raises(ValueError, match="will not interfere"):
        restoration.restore(project, ram)
    assert partial.read_bytes() == b"still downloading"


def test_space_preflight_prevents_writes(tmp_path, monkeypatch):
    project, ram, _ = tiny_snapshot(tmp_path, monkeypatch)
    actual = restoration.shutil.disk_usage(ram)
    monkeypatch.setattr(restoration.shutil, "disk_usage", lambda path: actual._replace(free=1))
    with pytest.raises(OSError, match="capacity"):
        restoration.restore(project, ram)
    assert not any((ram / name).exists() for name in restoration.EXPECTED_FILES)


@pytest.mark.parametrize("mutation", ["extra", "duplicate", "revision", "identity", "missing_sha"])
def test_snapshot_manifest_must_match_allowlist(tmp_path, monkeypatch, mutation):
    _, _, manifest = tiny_snapshot(tmp_path, monkeypatch)
    if mutation == "extra":
        manifest["siblings"][0]["rfilename"] = "../outside"
    elif mutation == "duplicate":
        manifest["siblings"].append(manifest["siblings"][0])
    elif mutation == "revision":
        manifest["sha"] = "wrong"
    elif mutation == "identity":
        manifest["id"] = "wrong"
    else:
        next(item for item in manifest["siblings"] if item["rfilename"] in restoration.WEIGHTS)["lfs"] = {}
    with pytest.raises(ValueError):
        restoration.snapshot_entries(manifest)


def test_symlink_components_cannot_escape_root(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "perception_decision").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        restoration.contained_path(root, "perception_decision/config.json")


@pytest.mark.parametrize("name", ["../outside", "/etc/passwd", "perception_decision/../config.json", "x\\config.json"])
def test_path_aliases_and_traversal_rejected(tmp_path, name):
    with pytest.raises(ValueError, match="allowlisted"):
        restoration.contained_path(tmp_path, name)


def test_private_ram_root_requires_owner_only_permissions(tmp_path):
    root = tmp_path / "ram"
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    with pytest.raises(ValueError, match="0700"):
        restoration.private_root(root, root)
    root.chmod(0o700)
    assert restoration.private_root(root, root) == root
    with pytest.raises(ValueError, match="fixed private"):
        restoration.private_root(tmp_path / "different", root)


@pytest.mark.parametrize("kind", ["wrong_sha", "oversized", "truncated_zstd"])
def test_corrupt_archives_leave_no_partial_model(tmp_path, kind):
    content = b"small original weight" * 20
    target, archive = tmp_path / "weight.safetensors", tmp_path / "weight.zst"
    packed = zstandard.ZstdCompressor().compress(content)
    entry = {"size": len(content), "lfs": {"sha256": hashlib.sha256(content).hexdigest()}}
    if kind == "wrong_sha":
        entry["lfs"]["sha256"] = "0" * 64
    elif kind == "oversized":
        entry["size"] = len(content) - 1
    else:
        packed = packed[:-4]
    archive.write_bytes(packed)
    with pytest.raises((ValueError, zstandard.ZstdError)):
        restoration.restore_weight(entry, target, archive)
    assert not target.exists()
    assert not list(tmp_path.glob("*.restore-*"))


def test_metadata_source_symlink_is_rejected(tmp_path, monkeypatch):
    project, ram, _ = tiny_snapshot(tmp_path, monkeypatch)
    source = project / "model-metadata/config.json"
    original = source.read_bytes()
    outside = tmp_path / "outside.json"
    outside.write_bytes(original)
    source.unlink()
    source.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        restoration.restore(project, ram)
