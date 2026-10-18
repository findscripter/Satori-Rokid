"""CPU-only public checkpoint transfer with strict ranged reads and SHA256.

This task's coordinator owns shards 3/4 locally; the server owns shards 1/2.
No API credentials, model execution, quantization, or source-file deletion.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen


REVISION = "7c5517eaaa4dc3788602af40e077063792c75f6c"
REPOSITORY = "Mar2Ding/Dispider"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def block(url, start, end, total):
    for attempt in range(4):
        try:
            req = Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
                                        "User-Agent": "satori-public-model-transfer/1"})
            with urlopen(req, timeout=90) as response:
                content_range = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                if response.status != 206 or not match or tuple(map(int, match.groups())) != (start, end, total):
                    raise ValueError("range response differs from requested checkpoint bytes")
                payload = response.read(end - start + 2)
            if len(payload) != end - start + 1:
                raise ValueError("range payload has an incorrect length")
            return payload
        except Exception:
            if attempt == 3:
                raise RuntimeError("public checkpoint block transfer failed") from None
            time.sleep(min(attempt + 1, 3))


def staged_block(url, start, end, total, directory):
    """Stage completed ranges on disk so one slow range cannot idle workers."""
    target = directory / f"{start:012d}-{end:012d}.chunk"
    metadata = target.with_suffix(".json")
    if target.exists() and metadata.exists():
        record = json.loads(metadata.read_text(encoding="utf-8"))
        if (record.get("start") == start and record.get("end") == end
                and record.get("total") == total and target.stat().st_size == end - start + 1
                and digest(target) == record.get("sha256")):
            return target
        raise RuntimeError("existing staged range differs; preserving it")
    payload = block(url, start, end, total)
    temporary = target.with_suffix(".writing")
    temporary.write_bytes(payload)
    temporary.replace(target)
    record = {"start": start, "end": end, "total": total, "sha256": hashlib.sha256(payload).hexdigest()}
    metadata.write_text(json.dumps(record), encoding="utf-8")
    return target


def transfer(item, root, *, workers, block_bytes, endpoint):
    name, total = item["rfilename"], item["size"]
    expected = item["lfs"]["sha256"]
    target, partial = root / name, root / (name + ".part")
    if target.exists():
        if target.stat().st_size != total or digest(target) != expected:
            raise RuntimeError("existing target differs; preserving it")
        return {"file": name, "sha256": expected, "size": total, "verified": True, "reused": True}
    offset = partial.stat().st_size if partial.exists() else 0
    if not 0 <= offset <= total:
        raise RuntimeError("invalid existing partial size")
    url = f"{endpoint.rstrip('/')}/{REPOSITORY}/resolve/{REVISION}/{name}?download=true"
    began = time.monotonic()
    staging = root / (name + ".ranges")
    staging.mkdir(exist_ok=True)
    ranges = [(begin, min(total, begin + block_bytes) - 1) for begin in range(offset, total, block_bytes)]
    with partial.open("ab") as output, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(staged_block, url, begin, end, total, staging) for begin, end in ranges]
        for (begin, end), future in zip(ranges, futures):
            staged = future.result()
            if begin != offset:
                raise RuntimeError("non-contiguous checkpoint write refused")
            with staged.open("rb") as source:
                for payload in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(payload)
            output.flush()
            offset = end + 1
            print(json.dumps({"file": name, "downloaded_bytes": offset, "total_bytes": total,
                              "elapsed_seconds": round(time.monotonic() - began, 3)}), flush=True)
    actual = digest(partial)
    if actual != expected:
        raise RuntimeError("final checkpoint SHA256 differs; partial preserved")
    partial.rename(target)
    # Only this shard's exact staged files, after the complete official hash passes.
    for begin, end in ranges:
        staged = staging / f"{begin:012d}-{end:012d}.chunk"
        staged.unlink(missing_ok=True)
        staged.with_suffix(".json").unlink(missing_ok=True)
    return {"file": name, "sha256": actual, "size": total, "verified": True, "reused": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards", nargs="+", type=int, default=[3, 4])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--block-mib", type=int, default=8)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if metadata["sha"] != REVISION or metadata["id"] != REPOSITORY:
        raise ValueError("metadata is not the fixed public checkpoint")
    if not 1 <= args.workers <= 24 or not 1 <= args.block_mib <= 32 or any(n not in (1, 2, 3, 4) for n in args.shards):
        raise ValueError("invalid bounded transfer configuration")
    args.output.mkdir(parents=True, exist_ok=True)
    selected = [item for item in metadata["siblings"]
                if item["rfilename"] in {f"model-{n:05d}-of-00004.safetensors" for n in args.shards}]
    if len(selected) != len(set(args.shards)):
        raise ValueError("missing checkpoint shards")
    results = [transfer(item, args.output, workers=args.workers, block_bytes=args.block_mib * 1024**2,
                        endpoint=args.endpoint) for item in selected]
    report = {"model_id": REPOSITORY, "revision": REVISION, "verified_files": results}
    suffix = "-".join(str(n) for n in sorted(args.shards))
    (args.output / f"local-transfer-integrity-{suffix}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "verified", "files": len(results)}), flush=True)


if __name__ == "__main__":
    main()
