"""CPU freeze and readiness gates must run before any formal output is created."""

from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location("native_cpu_runner", Path(__file__).with_name("run_cpu_benchmark.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def fixture(tmp_path):
    paths = {}
    for name in ("manifest", "taxonomy", "registry", "backend", "code"):
        paths[name] = tmp_path / name
        paths[name].write_text(name)
    args = SimpleNamespace(manifest=paths["manifest"], taxonomy=paths["taxonomy"], model="VideoChat3-4B", timeout=240)
    freeze = {"manifest_sha256": runner.digest(paths["manifest"]), "taxonomy_sha256": runner.digest(paths["taxonomy"]),
              "metadata_registry_path": str(paths["registry"]), "metadata_registry_sha256": runner.digest(paths["registry"]),
              "prompt_sha256": hashlib.sha256(b"fixed prompt").hexdigest(), "frame_count": 8,
              "max_new_tokens": 4096, "temperature": 0, "timeout_seconds": 240, "automatic_retry": False,
              "code_files": {str(paths["code"]): runner.digest(paths["code"])},
              "models": {"VideoChat3-4B": {"hf_revision": "a"*40, "backend_file": str(paths["backend"]),
                                            "backend_sha256": runner.digest(paths["backend"])}}}
    return args, freeze, paths


def test_exact_freeze_accepted_and_changed_timeout_rejected(tmp_path):
    args, freeze, _ = fixture(tmp_path)
    runner.validate_freeze(args, freeze, "fixed prompt")
    args.timeout = 180
    with pytest.raises(ValueError, match="timeout_seconds"):
        runner.validate_freeze(args, freeze, "fixed prompt")


@pytest.mark.parametrize("target", ["manifest", "taxonomy", "registry", "backend", "code"])
def test_any_frozen_input_or_code_change_rejected(tmp_path, target):
    args, freeze, paths = fixture(tmp_path)
    paths[target].write_text("modified")
    with pytest.raises(ValueError):
        runner.validate_freeze(args, freeze, "fixed prompt")


def health(freeze):
    return {"status": "ready", "busy": False, "model": "VideoChat3-4B", "revision": "a"*40,
            "metadata_registry_sha256": freeze["metadata_registry_sha256"], "source_unique_frames": 8,
            "generation_retry": False, "json_repair": False}


def test_exact_live_model_revision_registry_and_readiness_required(tmp_path):
    _, freeze, _ = fixture(tmp_path)
    value = health(freeze)
    class Opener:
        def open(self, req, timeout):
            assert timeout == 10 and req.full_url == "http://127.0.0.1:18867/health"
            return io.BytesIO(json.dumps(value).encode())
    runner.probe_health(port=18867, token="test-token", model="VideoChat3-4B", freeze=freeze, opener=Opener())
    for name, wrong in (("model", "other"), ("revision", "b"*40), ("busy", True),
                         ("metadata_registry_sha256", "wrong"), ("json_repair", True), ("generation_retry", True)):
        value = health(freeze)
        value[name] = wrong
        with pytest.raises(ValueError, match="active service differs"):
            runner.probe_health(port=18867, token="test-token", model="VideoChat3-4B", freeze=freeze, opener=Opener())
