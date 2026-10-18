"""Load pinned Dispider once and execute the three frozen datasets sequentially."""
from pathlib import Path
import json
import os
import sys

TOOLS = Path(__file__).resolve().parents[2]
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(ROOT / "Research/Backend"))

import satori_paper_benchmark as benchmark
from provider import MODEL_REVISION, PaperDispiderProvider, native

jobs = [
    ("gtea-goal28", "gtea-satori/goal28"),
    ("gtea-forecast96-raw73-v2", "gtea-satori/forecast96-raw73-v2"),
    ("egotaskqa-indirect200", "egotaskqa-satori/frozen-original-indirect-200"),
]
if list((ROOT / "Research/Data").rglob("gt.json")):
    raise ValueError("GT files present before inference")
if not os.environ.get("GPURUN_JOB") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
    raise ValueError("Use gpurun allocation")
for name, relative in jobs:
    data = ROOT / "Research/Data" / relative
    manifest, mh = benchmark._read(data / "infer-manifest.json")
    taxonomy, th = benchmark._read(data / "taxonomy.json")
    protocol, _ = benchmark._read(ROOT / "dispider-protocols-20261018" / (name + ".json"))
    benchmark.validate_manifest(manifest, taxonomy)
    benchmark._protocol_input_hashes(protocol, {"source_manifest_sha256": mh, "source_taxonomy_sha256": th})
    if protocol["model_revision"] != MODEL_REVISION or protocol["generation_parameters"] != {"max_tokens": 4096, "temperature": 0.0}:
        raise ValueError("Frozen parameters mismatch")
    if (ROOT / "runs" / ("dispider-" + name + "-20261018")).exists():
        raise FileExistsError("Refusing to rerun previous samples")

backend = native.Backend("/dev/shm/gpu-host-satori-dispider-7c5517e")
summaries = []
for name, relative in jobs:
    data = ROOT / "Research/Data" / relative
    output = ROOT / "runs" / ("dispider-" + name + "-20261018")
    provider = PaperDispiderProvider(backend)
    benchmark.infer(data / "infer-manifest.json", data / "taxonomy.json", output, provider,
                    protocol_path=ROOT / "dispider-protocols-20261018" / (name + ".json"),
                    progress=lambda row: print(json.dumps(row), flush=True))
    audit = benchmark.audit_run(output)
    benchmark._write(output / "audit.json", audit)
    summaries.append({"dataset": name, "audit": audit})
    print(json.dumps(summaries[-1]), flush=True)
benchmark._write(ROOT / "inference-summary.json", {"gpu_job": os.environ["GPURUN_JOB"], "datasets": summaries})
