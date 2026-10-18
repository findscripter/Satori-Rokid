import hashlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location("mage_modelscope_download", Path(__file__).with_name("download_modelscope.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def entry(name, content=b"weights"):
    return {"Path": name, "Type": "blob", "Size": len(content), "Sha256": hashlib.sha256(content).hexdigest(), "Revision": "fixed"}


class Response(io.BytesIO):
    def __init__(self, content, status=200, headers=None):
        super().__init__(content)
        self.status = status
        self.headers = headers or {}


def test_selection_is_root_only_and_excludes_optional_weight_packages():
    selected = MODULE.select_files({"Success": True, "Data": {"Files": [entry(name) for name in [
        "config.json", "modeling_mage_vl.py", "tokenizer.json", "chat_template.jinja", "merges.txt",
        "model-00001-of-00002.safetensors", "streammind_gate.safetensors", "neural_codec/dcvc_rt_intra.tar",
        "neural_codec/helper.py", "../escape.py", "C:\\escape.py", "README.md"]]}})
    assert [item["Path"] for item in selected] == ["chat_template.jinja", "config.json", "merges.txt",
        "model-00001-of-00002.safetensors", "modeling_mage_vl.py", "tokenizer.json"]


def test_verified_existing_file_is_reused_without_network(tmp_path):
    meta = entry("model-00001-of-00002.safetensors")
    (tmp_path / meta["Path"]).write_bytes(b"weights")
    def forbidden(*args, **kwargs):
        raise AssertionError("No request should be made")
    result = MODULE.download_file(tmp_path, meta, opener=forbidden)
    assert result["status"] == "reused"


def test_range_resume_validates_hash_and_does_not_duplicate_bytes(tmp_path):
    data = b"complete-test-model"
    meta = entry("model-00001-of-00002.safetensors", data)
    (tmp_path / (meta["Path"] + ".mage-part")).write_bytes(data[:8])
    calls = []
    def serve(req, **kwargs):
        calls.append(req)
        assert req.get_header("Range") == "bytes=8-"
        return Response(data[8:], 206, {"Content-Range": f"bytes 8-{len(data)-1}/{len(data)}"})
    result = MODULE.download_file(tmp_path, meta, opener=serve)
    assert result["status"] == "downloaded" and result["resumed_from"] == 8
    assert (tmp_path / meta["Path"]).read_bytes() == data
    assert not (tmp_path / (meta["Path"] + ".mage-part")).exists()


def test_server_ignoring_range_leaves_existing_partial_intact(tmp_path):
    meta = entry("config.json", b"abcdef")
    partial = tmp_path / "config.json.mage-part"
    partial.write_bytes(b"abc")
    with pytest.raises(RuntimeError, match="Range"):
        MODULE.download_file(tmp_path, meta, opener=lambda *a, **kw: Response(b"abcdef", 200))
    assert partial.read_bytes() == b"abc" and not (tmp_path / "config.json").exists()


def test_invalid_existing_final_is_never_overwritten(tmp_path):
    target = tmp_path / "config.json"
    target.write_bytes(b"keep")
    with pytest.raises(RuntimeError, match="existing"):
        MODULE.download_file(tmp_path, entry("config.json", b"good"))
    assert target.read_bytes() == b"keep"


def test_hash_mismatch_preserves_partial_and_does_not_publish_final(tmp_path):
    with pytest.raises(RuntimeError, match="SHA256"):
        MODULE.download_file(tmp_path, entry("config.json", b"good"), opener=lambda *a, **kw: Response(b"bad!"))
    assert not (tmp_path / "config.json").exists()
    assert (tmp_path / "config.json.mage-part").read_bytes() == b"bad!"


def test_space_check_includes_reserve_and_existing_partial(tmp_path, monkeypatch):
    meta = entry("model-00001-of-00002.safetensors", b"12345678")
    (tmp_path / (meta["Path"] + ".mage-part")).write_bytes(b"123")
    monkeypatch.setattr(MODULE.shutil, "disk_usage", lambda path: SimpleNamespace(free=7))
    with pytest.raises(RuntimeError, match="space"):
        MODULE.check_space(tmp_path, [meta], reserve_bytes=3)
    monkeypatch.setattr(MODULE.shutil, "disk_usage", lambda path: SimpleNamespace(free=8))
    assert MODULE.check_space(tmp_path, [meta], reserve_bytes=3)["remaining_bytes"] == 5
