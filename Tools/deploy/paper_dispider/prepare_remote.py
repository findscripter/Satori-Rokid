"""Prepare a fresh no-GT workspace from already-verified anonymous archives."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path("/home/gpu-user/work/satori-dispider-paper-20261018").resolve()
PREVIOUS = Path("/home/gpu-user/work/satori-paper-eval-20261017")
MODEL_REVISION = "7c5517eaaa4dc3788602af40e077063792c75f6c"
ROOT.mkdir(exist_ok=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for filename, expected in (
    ("gtea-native-infer-package.tar.gz", "8ac519ecf77120d20145cb077887267548f47599dcc545ffd30a93db46c5bfb5"),
    ("ego-native-infer-package.tar.gz", "541de1be76bf651f7bfbaca54f4b6e0cbedbd8d0a7a12b7c885419c7e53a542a"),
):
    archive_path = PREVIOUS / filename
    if digest(archive_path) != expected:
        raise ValueError("Frozen inference archive hash mismatch")
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (ROOT / member.name).resolve()
            if not target.is_relative_to(ROOT) or member.issym() or member.islnk():
                raise ValueError("Unsafe archive entry")
            if Path(member.name).name in ("gt.json", "selection.json", "source-audit.json"):
                raise ValueError("GT metadata not allowed in inference package")
        archive.extractall(ROOT, filter="data")

preflight = ROOT / "Research/Data/dispider-paper-preflight-20261018"
preflight.mkdir(parents=True, exist_ok=True)
for filename in ("infer-manifest.json", "taxonomy.json", "frame00.jpg", "frame01.jpg", "frame02.jpg", "frame03.jpg"):
    shutil.copyfile(PREVIOUS / "Research/Data/mage-paper-preflight-20261017" / filename, preflight / filename)

binding_files = [ROOT / "Tools/satori_paper_benchmark.py"]
binding_files += list((ROOT / "Tools/deploy/paper_dispider").glob("*.py"))
binding = {str(path.relative_to(ROOT)): digest(path) for path in sorted(binding_files)}
protocols = ROOT / "dispider-protocols-20261018"
protocols.mkdir(exist_ok=True)
for name, relative in (
    ("preflight", "dispider-paper-preflight-20261018"),
    ("gtea-goal28", "gtea-satori/goal28"),
    ("gtea-forecast96-raw73-v2", "gtea-satori/forecast96-raw73-v2"),
    ("egotaskqa-indirect200", "egotaskqa-satori/frozen-original-indirect-200"),
):
    data = ROOT / "Research/Data" / relative
    protocol = json.loads((data / "protocol.json").read_text()) if (data / "protocol.json").exists() else {}
    protocol.update({
        "model": "Dispider", "model_revision": MODEL_REVISION,
        "source_manifest_sha256": digest(data / "infer-manifest.json"),
        "source_taxonomy_sha256": digest(data / "taxonomy.json"),
        "input_unique_frames": 4, "model_frame_slots": 16,
        "slot_to_unique_image": [i for i in range(4) for _ in range(4)],
        "generation_parameters": {"max_tokens": 4096, "temperature": 0.0},
        "timeout_seconds": None,
        "preprocessing": "Four frozen original JPEGs, each repeated four times in adjacent slots; upstream nearest square resize and CLIP processor; timestamp endpoints from supplied original frames.",
        "native_path": "Direct offline Reaction once, with Perception/Decision memory; native streaming trigger not evaluated; keep_silent preserved.",
        "attempt_policy": "One attempt per sample; no retries, response repair, prompt or gate tuning, and no replacement samples. All outcomes retained.",
        "gt_isolation": "Inference workspace includes anonymous inputs, full shared taxonomy, technical protocol, frames and code only; GT added only after inference completes.",
        "source_code_sha256": binding,
        "preflight": name == "preflight",
    })
    target = protocols / (name + ".json")
    if target.exists():
        raise FileExistsError("Refusing to replace frozen protocol")
    target.write_text(json.dumps(protocol, indent=2) + "\n")
(ROOT / "source-binding-before.json").write_text(json.dumps(binding, indent=2) + "\n")
print(json.dumps({"root": str(ROOT), "protocols": str(protocols), "gt_files": list(map(str, ROOT.rglob("gt.json")))}))
