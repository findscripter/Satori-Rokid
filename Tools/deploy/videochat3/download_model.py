"""Download a fixed public VideoChat3 snapshot without a second HF cache copy.

Only CPU/network work. Every file is verified against the frozen HF manifest.
Partial files survive interrupted invocations. Never deletes unrelated files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

REPOSITORY = "MCG-NJU/VideoChat3-4B"
REVISION = "9f6107bdfee9686dfe493444022f5cf79a1a6acf"
RESERVE_BYTES = 3 * 1024**3


def verify(path: Path, info: dict) -> bool:
    if not path.is_file() or path.stat().st_size != info["size"]:
        return False
    lfs = info.get("lfs")
    digest = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        digest.update(f"blob {info['size']}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest() == (lfs["sha256"] if lfs else info["blobId"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--weight-source", choices=("huggingface", "modelscope"), default="huggingface")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    if manifest["sha"] != REVISION:
        raise ValueError("snapshot_revision_mismatch")
    args.target.mkdir(parents=True, exist_ok=True)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for info in sorted(manifest["siblings"], key=lambda x: x["size"]):
        name = info["rfilename"]
        destination = args.target / name
        if not destination.resolve().is_relative_to(args.target.resolve()):
            raise ValueError("unsafe_manifest_path")
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".partial")
        was_present = verify(destination, info)
        if not was_present:
            remaining = info["size"] - (partial.stat().st_size if partial.exists() else 0)
            if remaining < 0:
                raise ValueError("partial_file_larger_than_expected")
            if shutil.disk_usage(args.target).free - remaining < RESERVE_BYTES:
                raise RuntimeError("insufficient_space_with_3GiB_reserve")
            url = f"{args.endpoint}/{REPOSITORY}/resolve/{REVISION}/{name}"
            if args.weight_source == "modelscope" and name.endswith(".safetensors"):
                # All 3 LFS SHA256/size pairs also match this frozen MS mirror.
                mirror_revision = "9fc3c4a1c5f850399f621332719e0aebea334170"
                url = f"https://modelscope.cn/models/{REPOSITORY}/resolve/{mirror_revision}/{name}"
            print(json.dumps({"event": "download", "file": name, "size": info["size"]}), flush=True)
            # curl's continuation preserves a complete partial download on network failure.
            subprocess.run([
                "curl", "--fail", "--location", "--retry", "4", "--retry-delay", "3",
                "--connect-timeout", "20", "--speed-limit", "1024", "--speed-time", "90",
                "--continue-at", "-", "--output", str(partial), "--silent", "--show-error", url,
            ], check=True)
            if not verify(partial, info):
                raise ValueError(f"snapshot_integrity_failed:{name}")
            os.replace(partial, destination)
        rows.append({"file": name, "bytes": info["size"], "verified": True, "already_present": was_present})
        record = {"repository": REPOSITORY, "revision": REVISION, "complete": len(rows) == len(manifest["siblings"]),
                  "verified_files": rows, "verified_bytes": sum(x["bytes"] for x in rows), "updated_unix": time.time()}
        args.result.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps({"event": "verified", "file": name}), flush=True)


if __name__ == "__main__":
    main()
