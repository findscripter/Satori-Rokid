"""Eight parallel strict HTTP ranges, one contiguous append writer, pinned SHA."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from download_snapshot import REVISION, REPOSITORY, archive, digest

DIRECT_URLS = {}


def fetch(filename, start, end, size, base_url):
    began = time.perf_counter()
    for attempt in range(4):
        try:
            request = urllib.request.Request(
                DIRECT_URLS.get(filename, f"{base_url}/{REPOSITORY}/resolve/{REVISION}/{filename}"),
                headers={"Range": f"bytes={start}-{end}", "User-Agent": "satori-research/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                expected = f"bytes {start}-{end}/{size}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError("Server returned a wrong HTTP byte range")
                value = response.read(end - start + 2)
                if len(value) != end - start + 1:
                    raise ValueError("HTTP range has a wrong payload size")
                print(json.dumps({"event": "range_received", "file": filename,
                                  "start": start, "end": end,
                                  "seconds": time.perf_counter() - began}), flush=True)
                return value
        except Exception as error:
            print(json.dumps({"event": "range_error", "file": filename, "start": start,
                              "attempt": attempt + 1, "error_type": type(error).__name__}), flush=True)
            if attempt == 3:
                raise RuntimeError(f"Failed exact range {filename}:{start}-{end}") from None
            time.sleep(1)


def main():
    global DIRECT_URLS
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--reserve-bytes", type=int, default=12_171_225_472)
    parser.add_argument("--workers", type=int, default=8, choices=(8, 12, 16))
    parser.add_argument("--chunk-mib", type=int, default=32)
    parser.add_argument("--only-shards", type=int, nargs="*", choices=(1, 2, 3, 4))
    parser.add_argument("--url-map", type=Path)
    parser.add_argument("--base-url", default="https://hf-mirror.com",
                        choices=("https://hf-mirror.com", "https://huggingface.co"))
    args = parser.parse_args()
    if args.url_map:
        import urllib.parse
        DIRECT_URLS = json.loads(args.url_map.read_text())
        for url in DIRECT_URLS.values():
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme != "https" or not parsed.hostname.endswith((".cdn.hf.co", ".xethub.hf.co")):
                raise ValueError("Direct URLs must be public Hugging Face CDN HTTPS endpoints")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8-sig"))
    assert metadata["sha"] == REVISION
    args.root.mkdir(parents=True, exist_ok=True)
    args.evidence.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records, archives = [], []
    externally_pending = []
    for entry in metadata["siblings"]:
        name, size = entry["rfilename"], entry["size"]
        target = args.root / name
        if not name.endswith(".safetensors"):
            if not target.is_file() or target.stat().st_size != size:
                raise ValueError(f"Copy the pinned small metadata first: {name}")
            records.append({"filename": name, "bytes": size, "sha256": digest(target)})
            continue
        expected_sha = entry["lfs"]["sha256"]
        shard_number = int(name.split("-")[1])
        if args.only_shards and shard_number not in args.only_shards and not target.exists():
            externally_pending.append(name)
            continue
        if target.exists():
            if target.stat().st_size != size or digest(target) != expected_sha:
                raise ValueError("Existing completed shard failed SHA256")
        else:
            partial = target.with_name(target.name + ".incomplete")
            ledger_path = args.evidence / (name + ".ranges.json")
            prefix = partial.stat().st_size if partial.exists() else 0
            if prefix > size:
                raise ValueError("Partial file exceeds pinned file size")
            if ledger_path.exists():
                ledger = json.loads(ledger_path.read_text())
                offset = 0
                with partial.open("rb") as source:
                    for item in ledger["completed_ranges"]:
                        if item["start"] != offset:
                            raise ValueError("Range ledger is not contiguous")
                        value = source.read(item["bytes"])
                        if hashlib.sha256(value).hexdigest() != item["sha256"]:
                            raise ValueError("Partial range changed since its atomic ledger commit")
                        offset += item["bytes"]
                if prefix < offset:
                    raise ValueError("Partial file is shorter than committed ledger")
                if prefix > offset:
                    with partial.open("r+b") as output:
                        output.truncate(offset)
                prefix = offset
            else:
                ledger = {"filename": name, "size": size, "hf_lfs_sha256": expected_sha,
                          "workers": args.workers, "completed_ranges": []}
                if prefix:
                    ledger["completed_ranges"].append({"start": 0, "bytes": prefix,
                        "sha256": digest(partial), "origin": "preserved_contiguous_prior_download_prefix"})
            chunk_size = args.chunk_mib * 1024 * 1024
            ranges = [(start, min(size - 1, start + chunk_size - 1))
                      for start in range(prefix, size, chunk_size)]
            with concurrent.futures.ThreadPoolExecutor(args.workers) as pool, partial.open("ab") as output:
                pending = {}
                submitted = 0
                for index, (start, end) in enumerate(ranges):
                    while submitted < min(len(ranges), index + args.workers):
                        left, right = ranges[submitted]
                        pending[submitted] = pool.submit(fetch, name, left, right, size, args.base_url)
                        submitted += 1
                    value = pending.pop(index).result()
                    if output.tell() != start:
                        raise ValueError("Single writer append offset did not match HTTP range")
                    output.write(value)
                    output.flush()
                    ledger["completed_ranges"].append({"start": start, "bytes": len(value),
                                                       "sha256": hashlib.sha256(value).hexdigest()})
                    ledger_tmp = ledger_path.with_suffix(".json.tmp")
                    ledger_tmp.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
                    ledger_tmp.replace(ledger_path)
                    print(json.dumps({"event": "range_appended", "file": name,
                                      "bytes": end + 1, "total": size,
                                      "elapsed_seconds": time.perf_counter() - started}), flush=True)
                    del value
            if partial.stat().st_size != size or digest(partial) != expected_sha:
                raise ValueError("Completed shard does not match pinned HF LFS SHA256")
            partial.replace(target)
        record = {"filename": name, "bytes": size, "sha256": expected_sha,
                  "hf_lfs_sha256": expected_sha, "verified": True}
        records.append(record)
        if args.archive_root:
            archives.append(archive(entry, args.root, args.archive_root, args.reserve_bytes))
        state = {"repository": REPOSITORY, "revision": REVISION,
                 "status": "completed" if len(records) == len(metadata["siblings"]) else "partial",
                 "root": str(args.root), "files": records, "archives": archives,
                 "quantized": False, "reserve_bytes": args.reserve_bytes,
                 "download_method": "strict_range_parallel_fetch_single_append_writer"}
        (args.evidence / "download-manifest.json").write_text(json.dumps(state, indent=2))
        print(json.dumps({"event": "shard_verified", **record}), flush=True)
    state["status"] = "completed" if len(records) == len(metadata["siblings"]) else "partial"
    state["files"] = records
    state["externally_pending"] = externally_pending
    (args.evidence / "download-manifest.json").write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
