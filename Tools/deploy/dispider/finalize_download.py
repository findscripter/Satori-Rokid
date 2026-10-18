"""CPU verification and bounded cleanup of this deployment's small diagnostics."""
import json
import os
from pathlib import Path

from restore_model import RAM_ROOT, WEIGHTS, restore, sha256_file

project = Path(__file__).resolve().parent
result = restore(project, RAM_ROOT)
if result["status"] != "completed" or not result["all_24_files_verified"]:
    raise ValueError("Final snapshot integrity did not pass")
if any((RAM_ROOT / (name + suffix)).exists()
       for name in WEIGHTS for suffix in (".upload", ".incomplete")):
    raise ValueError("A redundant/incomplete model prefix remains")
records = []
for name in (
    "probe-first4MiB.bin", "local-probe-2.bin", "local-probe-3.bin", "local-probe-4.bin",
    "mirror-probe.bin", "temporary-direct-url.json", "temporary-direct-urls.json",
    "temporary-direct-urls34.json",
):
    path = project / name
    if not path.exists():
        continue
    if path.is_symlink() or path.resolve().parent != project.resolve():
        raise ValueError("Diagnostic cleanup escaped the deployment root")
    info = path.stat()
    if info.st_uid != os.getuid() or not path.is_file() or info.st_size > 8 * 1024**2:
        raise ValueError("Unexpected diagnostic identity/size; refuse cleanup")
    records.append({"filename": name, "bytes": info.st_size,
                    "sha256": sha256_file(path), "action": "removed_exact_owned_diagnostic"})
    path.unlink()
result["cleanup"] = records
result["redundant_upload_or_download_prefixes"] = False
output = project / "evidence/final-download-integrity.json"
output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({"status": result["status"], "verified_files": len(result["files"]),
                  "verified_bytes": sum(item["bytes"] for item in result["files"]),
                  "removed_owned_diagnostics": len(records), "evidence": str(output)}), flush=True)
