#!/usr/bin/env python3
"""Download only Mage-VL base inference files from its public ModelScope mirror.

Standard library only. No installation, GPU access, remote command execution, or
deletion of other files. Existing verified files are reused; partial transfers
resume only when the server confirms the requested byte range.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen


MODEL_ID = "microsoft/Mage-VL"
HF_REFERENCE_REVISION = "d88b153285f1633a61b2f693c59c8576693af185"
API = "https://modelscope.cn/api/v1/models/" + MODEL_ID
HF_REFERENCE_SHA256 = {
    "model-00001-of-00002.safetensors": "98fa203652a843650343732d2597f08c9b491cf4e310cd269eed34ca27ebce58",
    "model-00002-of-00002.safetensors": "719360069004f6b7b59303bd21bc1111bcd353bb0ccdebe1fe3971b87cd7b30f",
    "modeling_mage_vl.py": "3fe8bb1e40fa38db4ae08da097ee1238aa4258c4d454e66fe93a44d08ab641ec",
    "config.json": "fd4621212569e893ce2caa110431fde73f3cc1a1d92e1b8320bb4e0cfbcad34a",
    "configuration_mage_vl.py": "7e3da0c90726ef33da4f0362b3c71dc28650003f1cc11e061d3e99404a8bfed8",
    "processing_mage_vl.py": "c755ac93a161377cbe15f4d6dc30fae6460ef90e6d156d1bff7df82f3dd828cf",
    "video_processing_mage_vl.py": "67f28ea772c260b131983e8a90ad15948134a1baa769bfe6adbf0dd383541acb",
    "codec_video_processing_mage_vl.py": "f3bff7fe00f84a91d542e08561d5d819dd6a4b4192cf075858eb8c495f9e0b02",
    "streammind_gate.py": "5d9a9d7525aeecc0360ffd43f891ce9d2220297df5092a293a98e2fd31e9d99d",
    "model.safetensors.index.json": "2cad26f4364750d9b0551dff11e776837b9ef822fe29e068f68d935a8e65cc54",
    "preprocessor_config.json": "86401f541a46dee7689e84444b87cfa556e1575f1994a446582dc58d919cc69e",
    "video_preprocessor_config.json": "9d8345e0c7be09da56d8ea535fa6a5fa1a35728ccec9e7dbffa030ba34ce4a72",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_files(index):
    if not isinstance(index, dict) or index.get("Success") is not True:
        raise ValueError("ModelScope did not return a successful file index")
    files = index.get("Data", {}).get("Files")
    if not isinstance(files, list):
        raise ValueError("ModelScope file index is malformed")
    selected, seen = [], set()
    for item in files:
        name = item.get("Path", "")
        if item.get("Type") != "blob" or not isinstance(name, str):
            continue
        if not name or name in {".", ".."} or any(character in name for character in ("/", "\\", ":", "\x00")):
            continue
        if Path(name).suffix not in {".py", ".json", ".txt", ".jinja"} and not re.fullmatch(r"model-\d+-of-\d+\.safetensors", name):
            continue
        size, digest = item.get("Size"), item.get("Sha256")
        if (type(size) is not int or size < 0 or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-fA-F]{64}", digest) or name in seen):
            raise ValueError("Selected file has invalid size, SHA256, or duplicate name")
        selected.append({"Path": name, "Size": size, "Sha256": digest.lower(), "Revision": item.get("Revision") or "master"})
        seen.add(name)
    if not selected:
        raise ValueError("No base inference files were selected")
    return sorted(selected, key=lambda item: item["Path"])


def verify_reference(files):
    actual = {item["Path"]: item["Sha256"] for item in files}
    for name, digest in HF_REFERENCE_SHA256.items():
        if actual.get(name) != digest:
            raise RuntimeError(f"Mirror does not match pinned HF reference for {name}")


def _paths(model_dir, item):
    root = Path(model_dir).resolve()
    name = item["Path"]
    if Path(name).name != name or any(char in name for char in ("/", "\\", ":")):
        raise ValueError("Downloads must stay at the model directory root")
    target, partial = root / name, root / (name + ".mage-part")
    if target.is_symlink() or partial.is_symlink():
        raise RuntimeError("Refusing a symbolic-link download target")
    return target, partial


def _existing(target, item):
    if target.exists():
        if not target.is_file() or target.stat().st_size != item["Size"] or sha256(target) != item["Sha256"]:
            raise RuntimeError(f"Unverified existing file preserved: {target.name}")
        return True
    return False


def check_space(model_dir, files, *, reserve_bytes=2 * 1024**3):
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("reserve_bytes must be a nonnegative integer")
    remaining = 0
    for item in files:
        target, partial = _paths(model_dir, item)
        if _existing(target, item):
            continue
        partial_size = partial.stat().st_size if partial.is_file() else 0
        if partial.exists() and not partial.is_file() or partial_size > item["Size"]:
            raise RuntimeError(f"Invalid partial file preserved: {partial.name}")
        remaining += item["Size"] - partial_size
    free = shutil.disk_usage(model_dir).free
    if free < remaining + reserve_bytes:
        raise RuntimeError(f"Insufficient disk space: {free} free, {remaining} remaining plus {reserve_bytes} reserve required")
    return {"free_bytes": free, "remaining_bytes": remaining, "reserve_bytes": reserve_bytes}


def download_file(model_dir, item, *, timeout=90, opener=urlopen):
    target, partial = _paths(model_dir, item)
    if _existing(target, item):
        return {"status": "reused", "sha256": item["Sha256"], "size": item["Size"], "resumed_from": 0}
    offset = partial.stat().st_size if partial.is_file() else 0
    if partial.exists() and not partial.is_file() or offset > item["Size"]:
        raise RuntimeError(f"Invalid partial file preserved: {partial.name}")
    if offset < item["Size"]:
        url = API + "/repo?" + urlencode({"Revision": item["Revision"], "FilePath": item["Path"]})
        headers = {"Accept-Encoding": "identity", "User-Agent": "satori-mage-download/1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        req = Request(url, headers=headers)
        with opener(req, timeout=timeout) as response:
            status = response.status
            if offset and status != 206:
                raise RuntimeError("Server ignored Range request; partial preserved without truncation")
            if status not in {200, 206}:
                raise RuntimeError(f"Unexpected download HTTP status {status}")
            if status == 206:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if not match or int(match[1]) != offset or int(match[3]) != item["Size"] or int(match[2]) < offset:
                    raise RuntimeError("Invalid Content-Range; partial preserved")
            mode = "ab" if offset else "xb"
            if not offset and partial.exists():
                mode = "ab"  # Existing zero-byte partial is not truncated.
            written = offset
            with partial.open(mode) as output:
                while chunk := response.read(8 * 1024 * 1024):
                    if written + len(chunk) > item["Size"]:
                        raise RuntimeError("Response exceeds indexed file size; partial preserved")
                    output.write(chunk)
                    written += len(chunk)
                output.flush()
                os.fsync(output.fileno())
    elif item["Size"] == 0 and not partial.exists():
        partial.touch(exist_ok=False)
    if partial.stat().st_size != item["Size"]:
        raise RuntimeError("Download ended before indexed size; partial retained for resume")
    digest = sha256(partial)
    if digest != item["Sha256"]:
        raise RuntimeError("Downloaded SHA256 does not match mirror metadata; partial preserved")
    # Hard-link publication is atomic and refuses to overwrite a final file that
    # another process may have created. Both paths are in the same directory.
    os.link(partial, target)
    partial.unlink()  # Only this downloader's verified temporary file is removed.
    return {"status": "downloaded", "sha256": digest, "size": item["Size"], "resumed_from": offset}


def _save_manifest(path, value):
    temporary = path.with_name(path.name + ".tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise RuntimeError("Refusing a symbolic-link manifest target")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--reserve-gib", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.reserve_gib) or args.reserve_gib < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("reserve-gib must be nonnegative and timeout positive; both finite")
    root = args.model_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    index_url = API + "/repo/files?" + urlencode({"Revision": args.revision, "Recursive": "true"})
    with urlopen(Request(index_url, headers={"User-Agent": "satori-mage-download/1"}), timeout=args.timeout) as response:
        raw = response.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise RuntimeError("File index exceeds size limit")
    files = select_files(json.loads(raw))
    verify_reference(files)
    space = check_space(root, files, reserve_bytes=int(args.reserve_gib * 1024**3))
    print(json.dumps({"selected_files": len(files), "total_bytes": sum(item["Size"] for item in files), **space}), flush=True)
    if args.dry_run:
        return 0
    manifest_path = root / "mage-download-manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("downloader") != "satori-mage-modelscope-v1":
            raise RuntimeError("An unrelated manifest already exists; it was preserved")
    manifest = {"downloader": "satori-mage-modelscope-v1", "model_id": MODEL_ID,
                "hf_reference_revision": HF_REFERENCE_REVISION, "index_url": index_url,
                "status": "running", "files": {item["Path"]: {**item, "status": "pending"} for item in files}}
    _save_manifest(manifest_path, manifest)
    try:
        # Fetch the small processor/code files before the weight shards so CPU
        # compatibility can be inspected while the large downloads are running.
        for item in sorted(files, key=lambda entry: (entry["Path"].endswith(".safetensors"), entry["Path"])):
            print(f"Downloading/verifying {item['Path']} ({item['Size']} bytes)", flush=True)
            result = download_file(root, item, timeout=args.timeout)
            manifest["files"][item["Path"]].update(result)
            _save_manifest(manifest_path, manifest)
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error_type"] = type(exc).__name__
        raise
    finally:
        _save_manifest(manifest_path, manifest)
    print("All selected files match indexed SHA256 values.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        raise SystemExit(2)
