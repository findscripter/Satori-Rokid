"""Run only frozen unlabeled module inputs, inside an owner-allocated GPU job."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent / "Research" / "Backend"))

import satori_paper_benchmark as benchmark
from provider import PaperBackend, PaperVideoChat3Provider, audit_native_metadata, native, validate_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "taxonomy", "protocol", "metadata-registry", "model-path", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    # Validate frozen sources and metadata before potentially loading the model.
    manifest, manifest_sha = benchmark._read(args.manifest)
    taxonomy, taxonomy_sha = benchmark._read(args.taxonomy)
    protocol, _ = benchmark._read(args.protocol)
    manifest = benchmark.validate_manifest(manifest, taxonomy)
    registry, registry_sha = benchmark._read(args.metadata_registry)
    registry_raw = Path(args.metadata_registry).read_bytes()
    if protocol.get("technical_metadata_registry_sha256") != registry_sha:
        raise ValueError("Freeze technical_metadata_registry_sha256 in protocol before inference")
    validate_registry(registry, manifest)
    benchmark._protocol_input_hashes(protocol, {"source_manifest_sha256": manifest_sha,
                                              "source_taxonomy_sha256": taxonomy_sha})
    if Path(args.output).exists():
        raise FileExistsError("Refusing to overwrite a previous run")
    native.require_gpu_allocation()
    backend = PaperBackend(args.model_path)
    provider = PaperVideoChat3Provider(backend, registry=registry, manifest=manifest,
                                      registry_sha256=registry_sha)
    try:
        benchmark.infer(args.manifest, args.taxonomy, args.output, provider,
                        protocol_path=args.protocol,
                        progress=lambda row: print(json.dumps(row), flush=True))
    finally:
        if Path(args.output).is_dir():
            (Path(args.output) / "metadata-registry.json").write_bytes(registry_raw)
    audit = benchmark.audit_run(args.output)
    metadata_audit = audit_native_metadata(args.output, registry)
    benchmark._write(Path(args.output) / "audit.json", audit)
    benchmark._write(Path(args.output) / "metadata-audit.json", metadata_audit)
    print(json.dumps({"audit": audit, "metadata_audit": metadata_audit}), flush=True)


if __name__ == "__main__":
    main()
