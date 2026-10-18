"""CPU HTTP client; run beside the GPU service to avoid SSH/tunnel resets.

Credentials stay in process memory, never argv, run metadata, or stdout. GT is
deliberately not accepted: score separately after all planned calls complete.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid
from urllib import request


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_freeze(args, freeze, prompt):
    for name, expected in (("manifest_sha256", digest(args.manifest)), ("taxonomy_sha256", digest(args.taxonomy)),
                           ("prompt_sha256", hashlib.sha256(prompt.encode()).hexdigest()),
                           ("frame_count", 8), ("max_new_tokens", 4096), ("temperature", 0),
                           ("timeout_seconds", args.timeout), ("automatic_retry", False)):
        if freeze.get(name) != expected:
            raise ValueError("frozen protocol differs: " + name)
    if args.model not in freeze["models"]:
        raise ValueError("model not preregistered")
    if digest(freeze["metadata_registry_path"]) != freeze["metadata_registry_sha256"]:
        raise ValueError("metadata registry changed after freeze")
    for path, expected in freeze.get("code_files", {}).items():
        if digest(path) != expected:
            raise ValueError("evaluation code changed after freeze")
    backend = freeze["models"][args.model]
    if digest(backend["backend_file"]) != backend["backend_sha256"]:
        raise ValueError("native backend changed after freeze")


def probe_health(*, port, token, model, freeze, opener):
    req = request.Request(f"http://127.0.0.1:{port}/health", headers={"Authorization": "Bearer " + token})
    with opener.open(req, timeout=10) as response:
        raw = response.read(65537)
    if len(raw) > 65536:
        raise ValueError("health response too large")
    health = json.loads(raw)
    expected = {"status": "ready", "busy": False, "model": model,
                "revision": freeze["models"][model]["hf_revision"],
                "metadata_registry_sha256": freeze["metadata_registry_sha256"],
                "source_unique_frames": 8, "generation_retry": False, "json_repair": False}
    if any(health.get(name) != value for name, value in expected.items()):
        raise ValueError("active service differs from frozen model/metadata or is not ready")
    return health


def main():
    parser = argparse.ArgumentParser()
    for flag in ("project-root", "manifest", "taxonomy", "output", "freeze", "token-file"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--timeout", default=240, type=float)
    args = parser.parse_args()
    root = args.project_root.resolve()
    sys.path[:0] = [str(root / "Tools"), str(root / "Research/Backend"), str(root)]
    from gt_benchmark import BenchmarkProvider, PROMPT, infer
    from satori_repro.providers import _NoRedirect
    freeze = json.loads(args.freeze.read_text("utf-8"))
    validate_freeze(args, freeze, PROMPT)
    name = "SATORI_NATIVE_EVAL_" + uuid.uuid4().hex.upper()
    try:
        token = args.token_file.read_text("ascii").strip()
        if len(token) < 16 or any(c.isspace() for c in token):
            raise ValueError("invalid token file")
        health = probe_health(port=args.port, token=token, model=args.model, freeze=freeze,
                              opener=request.build_opener(_NoRedirect()))
        os.environ[name] = token
        provider = BenchmarkProvider(model=args.model, base_url=f"http://127.0.0.1:{args.port}/v1",
                                     api_key_env=name, timeout=args.timeout, frame_count=8,
                                     max_output_tokens=4096, temperature=0)
        result = infer(args.manifest, args.taxonomy, args.output, provider, frame_count=8,
                       progress=lambda row: print(json.dumps(row), flush=True))
        # This extra evidence binds the unchanged generic benchmark output to
        # the preregistration. Do not mutate old run metadata or predictions.
        (args.output / "native-protocol-binding.json").write_text(json.dumps({
            "freeze_sha256": digest(args.freeze), "model": args.model,
            "frame_count": 8, "gt_read": False, "automatic_retry": False,
            "client_location": "server CPU loopback", "timeout_seconds": args.timeout,
            "ready_health": health,
        }, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: result[key] for key in ("status", "successful_samples", "failed_samples", "api_calls")}))
        return 0 if not result["failed_samples"] else 2
    finally:
        os.environ.pop(name, None)


if __name__ == "__main__":
    sys.exit(main())
