"""One-shot Unity file bridge to the project's authenticated Mage-VL endpoint.

Unity receives no credential. This process validates local input first, checks the
existing managed tunnel, then captures only the requested SSH token into memory.
No automatic retries or previous-answer fallback are performed.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

from satori_repro.context_assistant import _validate_answer
from satori_repro.context_runner import run_context_question
from satori_repro.models import validate_session
from satori_repro.providers import _image_data_url, _json_loads
from satori_repro.stage_provider import SatoriStageProvider, _redacted_trace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL = "Mage-VL"
BASE_URL = "http://127.0.0.1:18866/v1"
_MAX_REQUEST_BYTES = 65536
_GUID = re.compile(r"(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})")


class BridgeError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def _hidden_flags():
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _local_process(command, *, cwd=None, timeout=60):
    try:
        return subprocess.run(command, cwd=cwd, stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout, check=False, creationflags=_hidden_flags())
    except (OSError, subprocess.TimeoutExpired):
        raise BridgeError("service_unavailable", "无法完成 Mage-VL 服务连接检查。") from None


def ensure_service(project_root):
    """Use the existing ownership-checked tunnel script; discard all captured text."""
    shell = shutil.which("pwsh.exe") or shutil.which("powershell.exe") or shutil.which("pwsh")
    script = Path(project_root) / "Tools/Ask-MageVL.ps1"
    if not shell or not script.is_file():
        raise BridgeError("service_unavailable", "找不到本项目的 Mage-VL 连接检查脚本。")
    result = _local_process([shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                             "-File", str(script), "-CheckOnly"], cwd=project_root, timeout=75)
    if result.returncode:
        raise BridgeError("service_unavailable", "Mage-VL 服务或项目隧道不可用；本次未发送问答。")
    try:
        health = _json_loads(result.stdout.decode("utf-8-sig").strip())
        if health.get("status") != "ready" or health.get("served_model") != MODEL:
            raise ValueError
    except (ValueError, UnicodeError, AttributeError):
        raise BridgeError("service_unavailable", "服务未返回可验证的 Mage-VL 就绪状态。") from None


def fetch_service_token():
    """Capture the single authorized credential over SSH, never to console or disk."""
    executable = shutil.which("ssh.exe") or shutil.which("ssh")
    if not executable:
        raise BridgeError("credential_unavailable", "找不到 SSH 客户端，无法取得服务凭证。")
    result = _local_process(
        [executable, "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "gpu-host",
         "cat /home/gpu-user/work/mage-vl-satori/.secrets/api.token"], timeout=30)
    try:
        token = result.stdout.decode("ascii").strip()
    except UnicodeError:
        token = ""
    if (result.returncode or not 16 <= len(token) <= 512
            or any(char.isspace() for char in token)):
        raise BridgeError("credential_unavailable", "无法通过 SSH 取得有效的 Mage-VL 服务凭证。")
    return token


def _contains_secret(value, token):
    if isinstance(value, str):
        return token in value
    if isinstance(value, dict):
        return any(_contains_secret(key, token) or _contains_secret(item, token) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_secret(item, token) for item in value)
    return False


class _CredentialGuard:
    """Block a credential echo before the core runner writes any answer artifact."""
    synthetic = False
    def __init__(self, provider, token):
        self._provider, self._token = provider, token
        self.name = provider.name
        self.model = provider.model
        self.base_url = provider.base_url

    @property
    def traces(self):
        return [_redacted_trace(trace, self._token) for trace in self._provider.traces]

    def complete_json(self, stage, frames, payload):
        result = self._provider.complete_json(stage, frames, payload)
        if _contains_secret(result, self._token):
            raise RuntimeError("Credential-like content was rejected before answer persistence")
        return result


def _io_paths(request_path, response_path, root):
    allowed = (root / "UserSettings/SatoriMage").resolve()
    def resolve(path):
        path = Path(path)
        return (path if path.is_absolute() else root / path).resolve()
    request_path, response_path = resolve(request_path), resolve(response_path)
    if not request_path.is_relative_to(allowed) or not response_path.is_relative_to(allowed):
        raise BridgeError("path_not_allowed", "请求和响应文件必须位于本项目 UserSettings/SatoriMage。")
    if request_path == response_path:
        raise BridgeError("path_not_allowed", "响应文件不能覆盖请求文件。")
    if response_path.exists():
        raise BridgeError("response_exists", "响应文件已经存在；请使用新的请求 ID。")
    return request_path, response_path


def _read_request(path):
    try:
        with path.open("rb") as source:
            raw = source.read(_MAX_REQUEST_BYTES + 1)
        if len(raw) > _MAX_REQUEST_BYTES:
            raise ValueError
        data = _json_loads(raw.decode("utf-8-sig"))
        if not isinstance(data, dict) or set(data) != {"request_id", "question", "session_path", "at_ms", "window"}:
            raise ValueError
        if not isinstance(data["request_id"], str) or not _GUID.fullmatch(data["request_id"]):
            raise ValueError
        uuid.UUID(data["request_id"])
        if not isinstance(data["question"], str) or not data["question"].strip() or len(data["question"]) > 2000:
            raise ValueError
        if not isinstance(data["session_path"], str) or not data["session_path"].strip() or len(data["session_path"]) > 4096:
            raise ValueError
        if type(data["window"]) is not int or not 1 <= data["window"] <= 32:
            raise ValueError
        if type(data["at_ms"]) is not int or data["at_ms"] < 0:
            raise ValueError
    except (OSError, ValueError, UnicodeError):
        raise BridgeError("invalid_request", "请求参数无效：请检查请求 ID、问题、时间和帧数。") from None
    return data


def _preflight(data, root):
    """Validate exactly the images that can be sent, before SSH or health traffic."""
    try:
        session_path = Path(data["session_path"])
        session_path = (session_path if session_path.is_absolute() else root / session_path).resolve()
        session = validate_session(_json_loads(session_path.read_text(encoding="utf-8-sig")))
        selected = [(index, frame) for index, frame in enumerate(session["frames"])
                    if frame["timestamp_ms"] <= data["at_ms"]][-data["window"]:]
        if not selected:
            raise ValueError
        expected = []
        for index, frame in selected:
            if not frame.get("image_path"):
                raise ValueError
            image = (session_path.parent / frame["image_path"]).resolve()
            _image_data_url(image)
            expected.append({"frame_id": f"v{index:03d}", "timestamp_ms": frame["timestamp_ms"],
                             "image_path": str(image), "sha256": hashlib.sha256(image.read_bytes()).hexdigest()})
    except (OSError, ValueError, UnicodeError):
        raise BridgeError("invalid_session", "选定的会话或图片不可用；请检查时间范围和图片文件。") from None
    return session_path, expected


def _atomic_response(path, value):
    """Publish a complete file without overwriting an existing destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
    descriptor, temp_name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        # A hard link creates the destination atomically and refuses replacement.
        os.link(temporary, path)
    except FileExistsError:
        raise BridgeError("response_exists", "响应文件已经存在；未覆盖已有内容。") from None
    except OSError:
        raise BridgeError("response_write_failed", "无法安全写入本次响应文件。") from None
    finally:
        temporary.unlink(missing_ok=True)


def _response():
    return {"status": "failed", "request_id": "", "question": "", "run_directory": None,
            "model": MODEL, "answer_payload": None, "raw_answer_json": None, "input_frames": [],
            "latency_ms": 0.0, "usage": [], "missing_sensors": ["imu"],
            "error_code": None, "message": ""}


def handle_request(request_path, response_path, *, project_root=None):
    root = Path(project_root or PROJECT_ROOT).resolve()
    request_path, response_path = _io_paths(request_path, response_path, root)
    started = time.perf_counter()
    response, token, environment_name = _response(), None, None
    run_directory = None
    try:
        data = _read_request(request_path)
        response.update(request_id=data["request_id"], question=data["question"])
        session_path, expected_frames = _preflight(data, root)
        ensure_service(root)
        token = fetch_service_token()
        if _contains_secret(data, token) or _contains_secret(expected_frames, token):
            raise BridgeError("sensitive_input", "请求包含不适合提交或显示的敏感内容。")
        environment_name = "SATORI_UNITY_MAGE_" + uuid.uuid4().hex.upper()
        if environment_name in os.environ:
            raise BridgeError("internal_error", "无法建立独立的服务凭证环境。")
        os.environ[environment_name] = token
        provider = _CredentialGuard(SatoriStageProvider(
            model=MODEL, base_url=BASE_URL, api_key_env=environment_name, timeout=180), token)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_directory = root / "Research/Backend/runs" / ("unity-mage-" + stamp + "-" + uuid.UUID(data["request_id"]).hex)
        response["run_directory"] = str(run_directory)
        run_context_question(session_path, data["question"], provider, run_directory,
                             window_size=data["window"], at_ms=data["at_ms"])
        run = _json_loads((run_directory / "run.json").read_text(encoding="utf-8-sig"))
        context = _json_loads((run_directory / "input-context.json").read_text(encoding="utf-8-sig"))
        raw_answer = (run_directory / "answer.json").read_bytes().decode("utf-8")
        answer = _json_loads(raw_answer)
        traces = provider.traces
        expected_ids = [frame["frame_id"] for frame in expected_frames]
        expected_hashes = {frame["frame_id"]: frame["sha256"] for frame in expected_frames}
        if len(traces) != 1 or run.get("api_calls") != 1:
            raise BridgeError("invalid_artifacts", "本次结果缺少唯一成功的模型调用记录，未显示答案。")
        trace = traces[0]
        if (trace.get("status") != "success" or trace.get("stage") != "context_question"
                or trace.get("requested_model") != MODEL or trace.get("returned_model") != MODEL
                or trace.get("image_ids") != expected_ids or trace.get("image_sha256") != expected_hashes
                or trace.get("raw_json") != answer):
            raise BridgeError("invalid_artifacts", "模型调用记录与本次图片或原始答案不匹配，未显示答案。")
        if (run.get("status") != "completed" or run.get("model") != MODEL
                or context.get("question") != data["question"] or context.get("frames") != expected_frames):
            raise BridgeError("invalid_artifacts", "本次模型结果与输入证据不匹配，未显示答案。")
        _validate_answer(answer, [frame["frame_id"] for frame in expected_frames], [])
        if _contains_secret((answer, raw_answer), token):
            raise BridgeError("sensitive_output", "模型返回了不适合显示的敏感内容，未显示答案。")
        response.update(status="completed", answer_payload=answer, raw_answer_json=raw_answer,
                        input_frames=expected_frames, usage=run.get("usage", []),
                        missing_sensors=run.get("missing_sensors", ["imu"]),
                        message="本次 Mage-VL 环境问答已完成。")
    except BridgeError as error:
        response.update(status="failed", error_code=error.code, message=error.message)
    except KeyboardInterrupt:
        response.update(status="failed", error_code="cancelled", message="本次问答已取消；未复用历史答案。")
    except Exception:
        response.update(status="failed", error_code="inference_failed",
                        message="本次 Mage-VL 问答失败；请查看本次运行记录。未自动重试。")
    finally:
        if environment_name is not None:
            os.environ.pop(environment_name, None)
        if response["status"] != "completed":
            response.update(answer_payload=None, raw_answer_json=None, input_frames=[])
            if run_directory is not None and not run_directory.exists():
                response["run_directory"] = None
            if token:
                response = _redacted_trace(response, token)
        response["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        token = None
    _atomic_response(response_path, response)
    return response


def main(argv=None):
    parser = argparse.ArgumentParser(description="One-shot Unity to Mage-VL context bridge")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        response = handle_request(args.request, args.response)
        print(json.dumps({"status": response["status"], "request_id": response["request_id"],
                          "error_code": response["error_code"]}, ensure_ascii=True), flush=True)
        return 0 if response["status"] == "completed" else 2
    except BridgeError as error:
        print(json.dumps({"status": "failed", "error_code": error.code}), flush=True)
        return 2
    except Exception:
        print(json.dumps({"status": "failed", "error_code": "internal_error"}), flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
