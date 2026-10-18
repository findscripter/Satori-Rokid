"""Offline CPU-only restoration of the pinned Dispider snapshot into private RAM.

Compressed files are lossless archives, not quantized weights. Every existing or
restored file is checked against the fixed HF revision metadata before success.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
from pathlib import Path

REVISION = "7c5517eaaa4dc3788602af40e077063792c75f6c"
REPOSITORY = "Mar2Ding/Dispider"
RAM_ROOT = Path("/dev/shm/gpu-host-satori-dispider-7c5517e")
WEIGHTS = {f"model-{index:05d}-of-00004.safetensors" for index in range(1, 5)}
METADATA_FILES = {
    ".gitattributes", "README.md", "added_tokens.json", "config.json",
    "dispider_checkpoint_manifest.json", "generation_config.json", "merges.txt",
    "model.safetensors.index.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.json",
    "perception_decision/added_tokens.json", "perception_decision/config.json",
    "perception_decision/generation_config.json", "perception_decision/merges.txt",
    "perception_decision/special_tokens_map.json", "perception_decision/tokenizer_config.json",
    "perception_decision/vocab.json", "perception_decision/vision_tower/config.json",
    "perception_decision/vision_tower/preprocessor_config.json",
}
EXPECTED_FILES = WEIGHTS | METADATA_FILES


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def contained_path(root: Path, relative: str) -> Path:
    """Reject traversal, aliases and symlink components before opening a file."""
    if not isinstance(relative, str) or "\\" in relative or relative not in EXPECTED_FILES:
        raise ValueError("Snapshot contains a non-allowlisted path")
    candidate = root / relative
    current = root
    for component in Path(relative).parts:
        if component in {".", ".."}:
            raise ValueError("Invalid snapshot path component")
        current = current / component
        if current.is_symlink():
            raise ValueError("Snapshot path contains a symlink")
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError("Snapshot path escaped its private root")
    return candidate


def private_root(root: Path, required_root: Path = RAM_ROOT) -> Path:
    if root.absolute() != required_root.absolute() or root.resolve() != required_root.absolute():
        raise ValueError("RAM model path must be the fixed private deployment directory")
    if root.is_symlink():
        raise ValueError("RAM model directory cannot be a symlink")
    root.mkdir(mode=0o700, parents=False, exist_ok=True)
    info = root.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError("RAM model directory must belong to the current user")
    if info.st_mode & 0o077:
        raise ValueError("RAM model directory must have private 0700 permissions")
    return root


def snapshot_entries(metadata: dict) -> list[dict]:
    if metadata.get("sha") != REVISION or metadata.get("id") != REPOSITORY:
        raise ValueError("Snapshot metadata does not identify the pinned Dispider revision")
    entries = metadata.get("siblings")
    if not isinstance(entries, list) or {item.get("rfilename") for item in entries} != EXPECTED_FILES:
        raise ValueError("Snapshot must contain exactly the 24 allowlisted files")
    if len(entries) != len(EXPECTED_FILES):
        raise ValueError("Duplicate snapshot file entries")
    for entry in entries:
        if type(entry.get("size")) is not int or not 0 < entry["size"] < 5 * 1024**3:
            raise ValueError("Invalid pinned file size")
        if entry["rfilename"] in WEIGHTS:
            expected = (entry.get("lfs") or {}).get("sha256")
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError("Weight entry lacks the official LFS SHA256")
        elif not re.fullmatch(r"[0-9a-f]{40}", entry.get("blobId", "")):
            raise ValueError("Metadata entry lacks its official Git blob SHA1")
    return entries


def validate_file(path: Path, entry: dict) -> str:
    if not path.is_file() or path.is_symlink() or path.stat().st_size != entry["size"]:
        raise ValueError(f"Pinned file is missing or has the wrong size: {entry['rfilename']}")
    actual = sha256_file(path)
    if entry["rfilename"] in WEIGHTS:
        if actual != entry["lfs"]["sha256"]:
            raise ValueError(f"Official weight SHA256 mismatch: {entry['rfilename']}")
    else:
        data = path.read_bytes()
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if blob != entry["blobId"]:
            raise ValueError(f"Official metadata Git blob mismatch: {entry['rfilename']}")
    return actual


def restore_weight(entry: dict, target: Path, source: Path) -> str:
    import zstandard
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Verified lossless archive is missing: {source.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    h, total = hashlib.sha256(), 0
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=target.name + ".restore-",
                                         dir=target.parent, delete=False) as output:
            temporary = Path(output.name)
            os.chmod(temporary, 0o600)
            with source.open("rb") as packed, zstandard.ZstdDecompressor().stream_reader(packed) as stream:
                for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                    total += len(chunk)
                    if total > entry["size"]:
                        raise ValueError("Archive expands beyond its pinned file size")
                    h.update(chunk)
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if total != entry["size"] or h.hexdigest() != entry["lfs"]["sha256"]:
            raise ValueError("Restored weight does not match its official size/SHA256")
        if target.exists():
            raise ValueError("A target appeared during restoration; refusing to replace it")
        temporary.replace(target)
        return h.hexdigest()
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def restore(project: Path, root: Path) -> dict:
    project = project.resolve(strict=True)
    metadata_file = project / "model-metadata.json"
    if metadata_file.is_symlink():
        raise ValueError("Pinned snapshot metadata cannot be an external symlink")
    entries = snapshot_entries(json.loads(metadata_file.read_text(encoding="utf-8-sig")))
    root = private_root(root)
    for filename in WEIGHTS:
        for suffix in (".incomplete", ".upload", ".uploading", ".upload-part", ".part"):
            if (root / (filename + suffix)).exists():
                raise ValueError("A model download/upload is incomplete; restore will not interfere")
    lock_path = root / ".restore.lock"
    if lock_path.is_symlink():
        raise ValueError("Restore lock cannot be a symlink")
    with lock_path.open("a") as lock:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        needed = sum(entry["size"] for entry in entries if not contained_path(root, entry["rfilename"]).exists())
        free = shutil.disk_usage(root).free
        if free < needed + 512 * 1024**2:
            raise OSError("Insufficient private RAM filesystem capacity, including 512MiB margin")
        archive_root = project / "model-archive"
        metadata_root = project / "model-metadata"
        if archive_root.is_symlink() or metadata_root.is_symlink():
            raise ValueError("Persistent deployment directories cannot be external symlinks")
        records = []
        for entry in entries:
            name = entry["rfilename"]
            target = contained_path(root, name)
            reused = target.exists()
            if reused:
                actual = validate_file(target, entry)
            elif name in WEIGHTS:
                source = archive_root / (name + ".zst")
                if source.resolve().parent != archive_root.resolve():
                    raise ValueError("Archive escaped its persistent directory")
                actual = restore_weight(entry, target, source)
            else:
                source = contained_path(metadata_root, name)
                actual = validate_file(source, entry)
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(mode="wb", prefix=target.name + ".restore-",
                                                 dir=target.parent, delete=False) as out:
                    temporary = Path(out.name)
                    try:
                        out.write(source.read_bytes())
                        out.flush()
                        os.fsync(out.fileno())
                        os.chmod(temporary, 0o600)
                        temporary.replace(target)
                    finally:
                        temporary.unlink(missing_ok=True)
                validate_file(target, entry)
            records.append({"filename": name, "bytes": entry["size"], "sha256": actual,
                            "verified": True, "reused_hot_cache": reused})
        result = {"status": "completed", "repository": REPOSITORY, "revision": REVISION,
                  "model_root": str(root), "files": records,
                  "all_24_files_verified": len(records) == 24,
                  "weight_storage": "original_weights_restored_from_lossless_zstd",
                  "quantized": False, "network_used": False, "device": "cpu",
                  "ram_bytes_needed_before_restore": needed, "ram_free_bytes_before_restore": free}
        if (root / "restored-integrity.json").is_symlink():
            raise ValueError("Integrity manifest cannot be a symlink")
        (root / "restored-integrity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        os.chmod(root / "restored-integrity.json", 0o600)
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--model-root", type=Path, default=RAM_ROOT)
    args = parser.parse_args()
    started = time.perf_counter()
    evidence = args.project_root.resolve() / "evidence"
    if evidence.is_symlink():
        raise ValueError("Restore audit directory cannot be an external symlink")
    evidence.mkdir(exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    output = evidence / f"restore-{stamp}-{os.getpid()}.json"
    try:
        result = restore(args.project_root, args.model_root)
    except Exception as error:
        result = {"status": "failed", "error_type": type(error).__name__, "device": "cpu",
                  "revision": REVISION, "network_used": False}
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"status": result["status"], "evidence": str(output),
                          "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
