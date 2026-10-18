"""Freeze native-model protocol before any classification call; no GT is read."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys

from server import compact, sha, strict_json


def main():
    parser = argparse.ArgumentParser()
    for flag in ("project-root", "manifest", "taxonomy", "registry", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--model", nargs=3, action="append", metavar=("ID", "HF_REVISION", "BACKEND_FILE"), required=True)
    parser.add_argument("--gt-sha256", help="Optional already frozen digest only; GT contents are never read")
    parser.add_argument("--smoke", action="store_true", help="Non-evaluation format smoke, explicitly separate from GT run")
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    root = args.project_root.resolve()
    sys.path[:0] = [str(root / "Tools"), str(root / "Research/Backend"), str(root)]
    from gt_benchmark import PROMPT, validate_manifest, validate_taxonomy
    PROMPT = PROMPT.replace("eight", "thirty-two")
    manifest = validate_manifest(strict_json(args.manifest.read_bytes()), frame_count=32)
    validate_taxonomy(strict_json(args.taxonomy.read_bytes()))
    if args.gt_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", args.gt_sha256):
        raise ValueError("invalid GT digest")
    if not args.smoke and len(manifest["samples"]) != 32:
        raise ValueError("formal comparison requires all 32 frozen samples")
    models = {}
    for model_id, revision, path in args.model:
        if not re.fullmatch(r"[0-9a-f]{40}", revision) or model_id in models:
            raise ValueError("exact unique model IDs/revisions required")
        models[model_id] = {"hf_revision": revision, "backend_file": str(Path(path).resolve()),
                            "backend_sha256": sha(Path(path).read_bytes())}
    files = [root / "Tools/gt_benchmark.py", *Path(__file__).parent.glob("*.py")]
    value = {
        "schema_version": 1, "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "non-evaluation native32 chunked-SDPA schema smoke" if args.smoke else "32-frame plus block-equivalent SDPA compatibility exploration; not a pure single-variable ablation",
        "vision_attention_execution": "Per native q/k_cu_seqlens independent block SDPA; same q/k/v, frames, pixels and weights",
        "floating_point_limit": "Kernel choice and rounding may differ from the original dense masked SDPA; no bitwise GPU equivalence claim",
        "prior_dense32_smoke": "vc3-native32-schema-smoke-001 failed with recorded OutOfMemoryError; preserved, not retried in place",
        "manifest_sha256": sha(args.manifest.read_bytes()), "taxonomy_sha256": sha(args.taxonomy.read_bytes()),
        "metadata_registry_sha256": sha(args.registry.read_bytes()), "prompt_sha256": sha(PROMPT.encode()),
        "metadata_registry_path": str(args.registry.resolve()), "timeout_seconds": args.timeout,
        "gt_sha256": args.gt_sha256, "planned_samples": len(manifest["samples"]), "frame_count": 32,
        "max_new_tokens": 4096, "temperature": 0, "automatic_retry": False, "json_repair": False,
        "valid_prediction": "Natural complete EOS and exact four-key JSON; no Markdown removal or completion repair",
        "model_input": "Same frozen thirty-two independent JPEGs; no GT/source IDs/scripts/objects/history/IMU/audio",
        "native_preprocessing": {"VideoChat3-4B": "Native 32-frame video tensor, do_sample_frames=False, temporal_merge_size=4, true source metadata",
                                 "controlled_ablation": "Nested 32 frames include all previous 8 unchanged JPEGs; additional temporal coverage and compute are not isolated"},
        "failures": "Every planned sample remains in denominator; failures and abstentions are not retried, removed or backfilled",
        "models": models,
        "code_files": {str(path.resolve()): sha(path.read_bytes()) for path in sorted(files)},
    }
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(compact({"status": "frozen", "sha256": sha(args.output.read_bytes()), "models": list(models),
                   "planned_samples": value["planned_samples"], "scope": value["scope"]}))


if __name__ == "__main__":
    main()
