"""Finish lossless archival after the other model has finished using disk."""
import argparse
import json
from pathlib import Path

from download_snapshot import REVISION, archive, digest

parser = argparse.ArgumentParser()
parser.add_argument("--metadata", type=Path, required=True)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--archive-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--reserve-bytes", type=int, default=3 * 1024**3)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
metadata = json.loads(args.metadata.read_text(encoding="utf-8-sig"))
assert metadata["sha"] == REVISION
records = []
for entry in metadata["siblings"]:
    if not entry["rfilename"].endswith(".safetensors"):
        continue
    original = args.root / entry["rfilename"]
    if original.stat().st_size != entry["size"] or digest(original) != entry["lfs"]["sha256"]:
        raise ValueError("Source shard has not passed pinned size and SHA256")
    records.append(archive(entry, args.root, args.archive_root, args.reserve_bytes))
    complete = len(records) == 4 and all(item["status"] == "verified_lossless" for item in records)
    args.output.write_text(json.dumps({"revision": REVISION, "archives": records,
                                     "status": "completed" if complete else "partial",
                                     "all_four_archives_verified_lossless": complete,
                                     "compressed_bytes": sum(item.get("bytes", 0) for item in records),
                                     "reserve_bytes": args.reserve_bytes}, indent=2))
    print(json.dumps(records[-1]), flush=True)
