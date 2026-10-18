"""Adopt only the two explicitly handed-off, independently verified SCP prefixes."""
import hashlib
import json
import os
from pathlib import Path

RAM_ROOT = Path("/dev/shm/gpu-host-satori-dispider-7c5517e")
PREFIXES = (
    ("model-00003-of-00004.safetensors", 479723520,
     "f31e9afefb4f1001a024386a9837f59fbd707233aceb9576beaba5ff448225a7"),
    ("model-00004-of-00004.safetensors", 93716480,
     "b95ee0ec358dfb9e8fbef2dde4f24f383d9892a3f6ec6d7c948f6734da3991af"),
)
if RAM_ROOT.resolve() != RAM_ROOT or RAM_ROOT.stat().st_uid != os.getuid():
    raise ValueError("Prefix directory identity/ownership mismatch")
records = []
for name, length, expected in PREFIXES:
    source, destination = RAM_ROOT / (name + ".upload"), RAM_ROOT / (name + ".incomplete")
    if source.is_symlink() or source.resolve().parent != RAM_ROOT or destination.exists():
        raise ValueError("Unsafe source or pre-existing competing writer destination")
    if source.stat().st_uid != os.getuid() or source.stat().st_size != length:
        raise ValueError("Handed-off source identity or size changed")
    h = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    if h.hexdigest() != expected:
        raise ValueError("Handed-off prefix changed from independently verified PC prefix")
    source.rename(destination)
    records.append({"source": str(source), "destination": str(destination), "bytes": length,
                    "sha256": expected, "status": "reused_by_exact_atomic_rename",
                    "pc_same_length_prefix_verified": True,
                    "origin": "SCP from local full shard independently matching official HF LFS SHA256"})
result = {"status": "completed", "files": records, "duplicate_prefix_files_remaining": False}
output = Path(__file__).parent / "evidence/scp-prefix-handoff.json"
output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
