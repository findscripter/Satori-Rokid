"""Pinned, checksum-verified Dispider download without an HF duplicate cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path

REVISION = "7c5517eaaa4dc3788602af40e077063792c75f6c"
REPOSITORY = "Mar2Ding/Dispider"
BASE_URL = "https://huggingface.co"


def fetch_prefix(filename: str, size: int) -> bytes:
    request = urllib.request.Request(
        f"{BASE_URL}/{REPOSITORY}/resolve/{REVISION}/{filename}",
        headers={"Range": f"bytes=0-{size - 1}", "User-Agent": "satori-research/1.0"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        value = response.read(size)
    if len(value) != size:
        raise ValueError("Incomplete compression sample")
    return value


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(entry: dict, root: Path) -> dict:
    filename, size = entry["rfilename"], entry["size"]
    target = root / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = (entry.get("lfs") or {}).get("sha256")
    if target.exists():
        actual = digest(target)
        if target.stat().st_size != size or (expected and actual != expected):
            raise ValueError(f"Existing file failed integrity validation: {filename}")
        return {"filename": filename, "bytes": size, "sha256": actual, "reused": True}
    temporary = target.with_name(target.name + ".incomplete")
    for attempt in range(4):
        start = temporary.stat().st_size if temporary.exists() else 0
        if start > size:
            raise ValueError("Partial download exceeds expected file length")
        if start == size:
            break
        headers = {"User-Agent": "satori-research/1.0"}
        if start:
            headers["Range"] = f"bytes={start}-"
        request = urllib.request.Request(
            f"{BASE_URL}/{REPOSITORY}/resolve/{REVISION}/{filename}", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                if start and (response.status != 206 or
                              not response.headers.get("Content-Range", "").startswith(f"bytes {start}-")):
                    raise ValueError("Server did not honor resumable byte range")
                mode = "ab" if start else "wb"
                last_report = start
                with temporary.open(mode) as stream:
                    while block := response.read(8 * 1024 * 1024):
                        stream.write(block)
                        current = stream.tell()
                        if current > size:
                            raise ValueError("Download exceeds pinned file size")
                        if current - last_report >= 256 * 1024 * 1024:
                            print(json.dumps({"event": "download_progress", "file": filename,
                                              "bytes": current, "total": size}), flush=True)
                            last_report = current
            if temporary.stat().st_size == size:
                break
        except Exception as error:
            print(json.dumps({"event": "download_transport_error", "file": filename,
                              "attempt": attempt + 1, "error_type": type(error).__name__}), flush=True)
            if attempt == 3:
                raise RuntimeError(f"Download failed: {filename}") from None
            time.sleep(2)
    if temporary.stat().st_size != size:
        raise ValueError(f"Incomplete download: {filename}")
    actual = digest(temporary)
    if expected and actual != expected:
        raise ValueError(f"SHA256 mismatch: {filename}")
    temporary.replace(target)
    return {"filename": filename, "bytes": size, "sha256": actual, "hf_lfs_sha256": expected,
            "reused": False}


def archive(entry: dict, root: Path, archive_root: Path, reserve_bytes: int) -> dict:
    import zstandard
    filename = entry["rfilename"]
    original = root / filename
    archive_root.mkdir(parents=True, exist_ok=True)
    target = archive_root / (filename + ".zst")
    if target.exists():
        h = hashlib.sha256()
        decompressed_bytes = 0
        with target.open("rb") as src, zstandard.ZstdDecompressor().stream_reader(src) as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(chunk)
                decompressed_bytes += len(chunk)
        expected = (entry.get("lfs") or {}).get("sha256")
        if h.hexdigest() != expected or decompressed_bytes != entry["size"]:
            raise ValueError(f"Existing archive failed lossless validation: {target.name}")
        return {"filename": filename, "status": "verified_lossless", "bytes": target.stat().st_size,
                "decompressed_sha256": h.hexdigest(), "reused": True}
    temporary = target.with_name(target.name + ".incomplete")
    budget = shutil.disk_usage(archive_root).free - reserve_bytes
    if budget <= 0:
        return {"filename": filename, "status": "skipped_for_disk_budget"}
    total = 0
    try:
        with original.open("rb") as source, temporary.open("wb") as dest:
            compressor = zstandard.ZstdCompressor(level=3).compressobj()
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                compressed = compressor.compress(chunk)
                total += len(compressed)
                if total > budget:
                    raise OSError("Archive would consume reserved disk space")
                dest.write(compressed)
            tail = compressor.flush()
            total += len(tail)
            if total > budget:
                raise OSError("Archive tail would consume reserved disk space")
            dest.write(tail)
        h = hashlib.sha256()
        with temporary.open("rb") as src, zstandard.ZstdDecompressor().stream_reader(src) as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        expected = (entry.get("lfs") or {}).get("sha256")
        if h.hexdigest() != expected:
            raise ValueError("Lossless archive verification failed")
        temporary.replace(target)
        return {"filename": filename, "status": "verified_lossless", "bytes": total,
                "decompressed_sha256": h.hexdigest()}
    except OSError:
        temporary.unlink(missing_ok=True)
        return {"filename": filename, "status": "skipped_for_disk_budget"}


def main() -> None:
    global BASE_URL
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", choices=("https://huggingface.co", "https://hf-mirror.com"),
                        default="https://huggingface.co")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--probe-compression", action="store_true")
    parser.add_argument("--download-weights", action="store_true")
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--reserve-bytes", type=int, default=12_171_225_472)
    args = parser.parse_args()
    BASE_URL = args.base_url
    metadata = json.loads(args.metadata.read_text(encoding="utf-8-sig"))
    if metadata["sha"] != REVISION:
        raise ValueError("Model revision does not match pinned metadata")
    args.root.mkdir(parents=True, exist_ok=True)
    os.chmod(args.root, 0o700)
    args.evidence.mkdir(parents=True, exist_ok=True)
    entries = metadata["siblings"]
    weights = [entry for entry in entries if entry["rfilename"].endswith(".safetensors")]
    if args.probe_compression:
        import zstandard
        samples = []
        for entry in weights:
            value = fetch_prefix(entry["rfilename"], 4 * 1024 * 1024)
            compressed = zstandard.ZstdCompressor(level=3).compress(value)
            samples.append({"filename": entry["rfilename"], "sample_bytes": len(value),
                            "compressed_bytes": len(compressed), "ratio": len(compressed) / len(value),
                            "weight_bytes": entry["size"]})
        estimated = sum(sample["ratio"] * sample["weight_bytes"] for sample in samples)
        output = {"status": "sampled", "revision": REVISION, "zstd_level": 3,
                  "samples": samples, "estimated_archive_bytes": int(estimated),
                  "conservative_plus_5_percent_bytes": int(estimated * 1.05),
                  "limitation": "Only each shard's first 4MiB; real compression and budget checked per archive."}
        (args.evidence / "compression-probe.json").write_text(json.dumps(output, indent=2))
        print(json.dumps(output, indent=2), flush=True)
        return
    records, archives = [], []
    output_path = args.evidence / "download-manifest.json"
    for entry in entries:
        is_weight = entry["rfilename"].endswith(".safetensors")
        if is_weight and not args.download_weights:
            continue
        records.append(download(entry, args.root))
        if is_weight and args.archive_root:
            archives.append(archive(entry, args.root, args.archive_root, args.reserve_bytes))
        output = {"repository": REPOSITORY, "revision": REVISION,
                  "status": "completed" if len(records) == len(entries) else "partial",
                  "root": str(args.root), "files": records, "archives": archives,
                  "quantized": False, "reserve_bytes": args.reserve_bytes}
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(json.dumps({"event": "file_complete", **records[-1]}), flush=True)


if __name__ == "__main__":
    main()
