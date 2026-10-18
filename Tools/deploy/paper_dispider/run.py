"""One-attempt frozen Dispider inference; no GT argument or scoring in this process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent / "Research" / "Backend"))

import satori_paper_benchmark as benchmark
from provider import MODEL_REVISION, PaperDispiderProvider, native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "taxonomy", "protocol", "model-path", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    manifest, manifest_sha = benchmark._read(args.manifest)
    taxonomy, taxonomy_sha = benchmark._read(args.taxonomy)
    protocol, _ = benchmark._read(args.protocol)
    benchmark.validate_manifest(manifest, taxonomy)
    benchmark._protocol_input_hashes(protocol, {"source_manifest_sha256": manifest_sha,
                                              "source_taxonomy_sha256": taxonomy_sha})
    if protocol.get("model_revision") != MODEL_REVISION or protocol.get("input_unique_frames") != 4:
        raise ValueError("Freeze pinned model and four-frame protocol before inference")
    if protocol.get("generation_parameters") != {"max_tokens": 4096, "temperature": 0.0}:
        raise ValueError("Frozen generation parameters disagree")
    if Path(args.output).exists():
        raise FileExistsError("Refusing to overwrite a previous run")
    provider = PaperDispiderProvider(native.Backend(args.model_path))
    benchmark.infer(args.manifest, args.taxonomy, args.output, provider, protocol_path=args.protocol,
                    progress=lambda row: print(json.dumps(row), flush=True))
    audit = benchmark.audit_run(args.output)
    benchmark._write(Path(args.output) / "audit.json", audit)
    print(json.dumps({"audit": audit}), flush=True)


if __name__ == "__main__":
    main()
