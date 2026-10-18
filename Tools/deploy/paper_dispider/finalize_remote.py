"""Score after all inference completed, bind final sources, package evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
from datetime import datetime, timezone

TOOLS = Path(__file__).resolve().parents[2]
ROOT = TOOLS.parent
PREVIOUS = Path("/home/gpu-user/work/satori-paper-eval-20261017")
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(ROOT / "Research/Backend"))
import satori_paper_benchmark as benchmark
from audit_native import audit

jobs = [("gtea-goal28", "gtea-satori/goal28"),
        ("gtea-forecast96-raw73-v2", "gtea-satori/forecast96-raw73-v2"),
        ("egotaskqa-indirect200", "egotaskqa-satori/frozen-original-indirect-200")]
for name, _ in jobs:
    metadata, _ = benchmark._read(ROOT / "runs" / ("dispider-" + name + "-20261018") / "run.json")
    if metadata["status"] != "completed" or metadata["not_run_samples"]:
        raise RuntimeError("All formal inference must finish before labels are introduced")

binding = json.loads((ROOT / "source-binding-before.json").read_text())
after = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in binding}
if after != binding:
    raise ValueError("Bound inference code changed")
benchmark._write(ROOT / "source-binding-after.json", {"passed": True, "files": after,
                                                        "checked_utc": datetime.now(timezone.utc).isoformat()})
labels = ROOT / "scoring-labels"
labels.mkdir(exist_ok=True)
summaries = []
for name, relative in jobs:
    target = labels / (name + ".json")
    if target.exists():
        raise FileExistsError("Refusing to replace previously introduced labels")
    shutil.copyfile(PREVIOUS / "Research/Data" / relative / "gt.json", target)
    run_dir = ROOT / "runs" / ("dispider-" + name + "-20261018")
    scores = benchmark.score(run_dir, target)
    benchmark._write(run_dir / "scores.json", scores)
    native_audit = audit(run_dir)
    benchmark._write(run_dir / "native-audit.json", native_audit)
    summaries.append({"dataset": name, "scores": scores, "native_audit": native_audit})
benchmark._write(ROOT / "final-summary.json", summaries)
archive = ROOT / "dispider-paper-results-20261018.tar.gz"
with tarfile.open(archive, "w:gz") as tar:
    for name in ("runs", "dispider-protocols-20261018", "source-binding-before.json", "source-binding-after.json",
                 "inference-summary.json", "final-summary.json", "comparison-binding-sidecar.json"):
        if (ROOT / name).exists():
            tar.add(ROOT / name, arcname=name)
print(json.dumps({"completed": True, "archive": str(archive),
                  "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}))
