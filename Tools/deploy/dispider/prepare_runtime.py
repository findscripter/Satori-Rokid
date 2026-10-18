"""Create a documented attention-backend adaptation of pinned Dispider code."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import shutil
from pathlib import Path

SOURCE_REVISION = "803f6b4634b3232293e89a38bcce9f5815c33ed4"
MODEL_REVISION = "7c5517eaaa4dc3788602af40e077063792c75f6c"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if args.runtime.exists():
        raise SystemExit(f"Refusing to overwrite {args.runtime}")
    shutil.copytree(args.source, args.runtime)
    relative = Path("dispider/model/decision_backbone/adapter.py")
    original = (args.source / relative).read_text(encoding="utf-8")
    expected = '        config._attn_implementation = "flash_attention_2"\n'
    if original.count(expected) != 1:
        raise SystemExit("Pinned attention-backend line did not match exactly")
    patched = original.replace("import torch\n", "import os\n\nimport torch\n", 1)
    patched = patched.replace(
        expected,
        '        backend = os.environ.get("DISPIDER_ATTN_IMPLEMENTATION", "flash_attention_2")\n'
        '        if backend not in {"flash_attention_2", "sdpa", "eager"}:\n'
        '            raise ValueError(f"Unsupported Dispider attention backend: {backend}")\n'
        '        config._attn_implementation = backend\n',
        1,
    )
    (args.runtime / relative).write_text(patched, encoding="utf-8")
    args.evidence.mkdir(parents=True, exist_ok=True)
    (args.evidence / "sdpa-compatibility.patch").write_text(
        "".join(difflib.unified_diff(original.splitlines(True), patched.splitlines(True),
                                     fromfile=f"upstream/{relative.as_posix()}",
                                     tofile=f"runtime/{relative.as_posix()}")), encoding="utf-8")
    record = {
        "source_repository": "https://github.com/Mark12Ding/Dispider",
        "source_revision": SOURCE_REVISION,
        "model_repository": "Mar2Ding/Dispider",
        "model_revision": MODEL_REVISION,
        "changed_file": relative.as_posix(),
        "source_file_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "runtime_file_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "reason": "Reuse installed Torch/CUDA; official Decision forces unavailable FlashAttention2.",
        "weight_changes": False,
        "backend_selected_at_launch": "DISPIDER_ATTN_IMPLEMENTATION=sdpa",
        "official_validated_stack": "Python3.10 / Torch2.2.0+cu121 / FlashAttention2.5.9.post1",
        "scope": "Backend compatibility; random CPU equivalence is not end-to-end model equivalence.",
    }
    (args.evidence / "runtime-adaptation.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
