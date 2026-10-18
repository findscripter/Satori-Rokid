"""Resolve public temporary URLs without printing signed capabilities."""
import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

from download_snapshot import REVISION, REPOSITORY

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--shards", nargs="+", type=int, default=[1, 2])
args = parser.parse_args()
mapping = {}
for number in args.shards:
    filename = f"model-{number:05d}-of-00004.safetensors"
    request = urllib.request.Request(
        f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{filename}", method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        mapping[filename] = response.url
        print(json.dumps({"filename": filename, "host": urllib.parse.urlparse(response.url).hostname,
                          "status": response.status}))
args.output.write_text(json.dumps(mapping), encoding="utf-8")
