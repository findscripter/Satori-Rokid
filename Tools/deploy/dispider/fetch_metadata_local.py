"""Acquire only the small pinned model metadata through the local network."""
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path

project = Path(__file__).resolve().parent
metadata = json.loads((project / "model-metadata.json").read_text(encoding="utf-8-sig"))
root = project / "local-metadata"
entries = [entry for entry in metadata["siblings"] if not entry["rfilename"].endswith(".safetensors")]
base = "https://huggingface.co/Mar2Ding/Dispider/resolve/" + metadata["sha"] + "/"


def one(entry):
    target = root / entry["rfilename"]
    target.parent.mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlopen(base + entry["rfilename"], timeout=30).read()
    if len(data) != entry["size"]:
        raise ValueError("Metadata size mismatch")
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if blob != entry["blobId"]:
        raise ValueError("Metadata Git blob hash mismatch")
    target.write_bytes(data)
    return {"filename": entry["rfilename"], "bytes": len(data), "git_blob": blob,
            "sha256": hashlib.sha256(data).hexdigest()}


if __name__ == "__main__":
    records = list(concurrent.futures.ThreadPoolExecutor(4).map(one, entries))
    (root / "metadata-integrity.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(records), "bytes": sum(item["bytes"] for item in records)}))
