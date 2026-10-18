"""Compare existing pinned and SDPA runtime source bytes with historical proof."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

source = Path(sys.argv[1])
old = json.loads(source.read_text())
counts = {}
for group, metadata in old["sources"].items():
    root = Path(metadata["root"])
    current = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in root.rglob("*.py")}
    if current != metadata["files"]:
        raise ValueError("Pinned/runtime Python sources differ from historical identity")
    counts[group] = len(current)
result = {"passed": True, "captured_utc": datetime.now(timezone.utc).isoformat(), "files_verified": counts,
          "historical_source_identity_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
          "scope": "Post-start byte identity with the previously pinned source record; no claim this file was captured before current inference."}
Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
