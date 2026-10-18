"""Replay an exact prefix of recorded real API successes, then continue once.

This is request memoization for failed experiments, not a synthetic response
provider. Any cached input mismatch stops execution before a network request.
"""

from copy import deepcopy
import base64
import hashlib
from pathlib import Path
import re
import time

from .providers import _image_data_url, _json_loads
from .satori_method import build_stage_prompt
from .stage_provider import _CHINESE_FIELDS, _json_text


_FIELDS = ("requested_model", "stage", "image_ids", "image_sha256",
           "system_prompt_sha256", "payload_sha256")
_TRACE_FIELDS = set(_FIELDS) | {
    "status", "returned_model", "usage", "latency_ms", "raw_json",
    "http_status", "error",
}


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _fingerprint(model, stage, frames, payload):
    prompt = build_stage_prompt(stage, payload) + _CHINESE_FIELDS
    if not isinstance(payload, dict) or not isinstance(frames, list) or not frames:
        raise ValueError("Resume matching requires an object payload and nonempty frames")
    hashes, ids = {}, []
    previous = -1
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("Invalid resume frame")
        fid, stamp = frame.get("frame_id"), frame.get("timestamp_ms")
        if (not isinstance(fid, str) or not fid or fid in hashes
                or type(stamp) is not int or stamp < 0 or stamp <= previous):
            raise ValueError("Resume frames must have unique IDs and ordered timestamps")
        image = _image_data_url(frame.get("image_path"))
        hashes[fid] = _hash(base64.b64decode(image.split(",", 1)[1]))
        ids.append(fid)
        previous = stamp
    return {"requested_model": model, "stage": stage, "image_ids": ids, "image_sha256": hashes,
            "system_prompt_sha256": _hash(prompt.encode("utf-8")),
            "payload_sha256": _hash(_json_text(payload).encode("utf-8"))}


def _valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


class ResumableSatoriStageProvider:
    name = "satori-stage-resumable"
    synthetic = False

    def __init__(self, provider, *, resume_from=None, trace_path=None):
        if getattr(provider, "trace_path", None) is not None:
            raise ValueError("Wrapped provider must leave trace persistence to the resume wrapper")
        self.provider = provider
        self.model = provider.model
        self.base_url = provider.base_url
        self.trace_path = Path(trace_path).resolve() if trace_path is not None else None
        self.traces = []
        self.cached_calls = 0
        self.real_api_calls = 0
        self.resume_metadata = None
        self._prefix = []
        self._position = 0
        self._source_inputs = None
        if resume_from is not None:
            self._load_source(Path(resume_from))

    def _load_source(self, source):
        try:
            source = source.resolve(strict=True)
            run_path, trace_path = source / "run.json", source / "api-calls.jsonl"
            if self.trace_path == trace_path.resolve():
                raise ValueError("Resume output must not overwrite its source trace")
            run_bytes, trace_bytes = run_path.read_bytes(), trace_path.read_bytes()
            run = _json_loads(run_bytes.decode("utf-8-sig"))
            if (not isinstance(run, dict) or run.get("synthetic") is not False
                    or run.get("status") not in {"failed", "aborted"}
                    or run.get("provider") != "satori-staged-api"
                    or run.get("model") != self.model
                    or run.get("endpoint", "").rstrip("/") != self.base_url.rstrip("/")):
                raise ValueError("Resume source must be a compatible failed real API experiment")
            lines = trace_bytes.decode("utf-8-sig").splitlines()
            for line_number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                record = _json_loads(line)
                if not isinstance(record, dict) or record.get("status") not in {"success", "error"}:
                    raise ValueError("Invalid resume trace record")
                if record["status"] != "success":
                    break
                if (not isinstance(record.get("raw_json"), dict)
                        or not isinstance(record.get("requested_model"), str)
                        or not isinstance(record.get("stage"), str)
                        or not isinstance(record.get("image_ids"), list)
                        or not record["image_ids"]
                        or any(not isinstance(fid, str) or not fid for fid in record["image_ids"])
                        or len(set(record["image_ids"])) != len(record["image_ids"])
                        or not isinstance(record.get("image_sha256"), dict)
                        or set(record["image_sha256"]) != set(record["image_ids"])
                        or any(not _valid_hash(value) for value in record["image_sha256"].values())
                        or not _valid_hash(record.get("system_prompt_sha256"))
                        or not _valid_hash(record.get("payload_sha256"))):
                    raise ValueError("Successful resume trace lacks complete response fingerprints")
                if record.get("call_source") not in (None, "real", "cached"):
                    raise ValueError("Unknown response source in resume trace")
                if record.get("response_origin") not in (None, "real_api"):
                    raise ValueError("Resume responses must originate from the real API")
                if record.get("call_source") == "cached" and (
                    record.get("response_origin") != "real_api" or not isinstance(record.get("cache_provenance"), dict)
                ):
                    raise ValueError("Cached response has no real API provenance")
                result_hash = _hash(_json_text(record["raw_json"]).encode("utf-8"))
                if "raw_json_sha256" in record and record["raw_json_sha256"] != result_hash:
                    raise ValueError("Cached response content hash does not match")
                self._prefix.append((line_number, deepcopy(record), _hash(line.encode("utf-8"))))
            self._source_inputs = run.get("inputs")
            self.resume_metadata = {
                "source_run": str(source), "source_run_sha256": _hash(run_bytes),
                "source_trace": str(trace_path), "source_trace_sha256": _hash(trace_bytes),
                "success_prefix_calls": len(self._prefix),
            }
        except (OSError, UnicodeError, TypeError, AttributeError):
            raise ValueError("Cannot read compatible resume source metadata and trace") from None

    def validate_run_inputs(self, inputs):
        if self.resume_metadata is None:
            return
        for field in ("session_sha256", "task_catalog_sha256"):
            if (not isinstance(inputs, dict) or not isinstance(self._source_inputs, dict)
                    or not _valid_hash(inputs.get(field))
                    or inputs[field] != self._source_inputs.get(field)):
                raise ValueError(f"Resume input hash does not match: {field}")

    def assert_cache_consumed(self):
        if self._position != len(self._prefix):
            raise ValueError("Resume source has unconsumed successful calls")

    def _record(self, trace):
        clean = deepcopy(trace)
        self.traces.append(clean)
        if self.trace_path is not None:
            try:
                self.trace_path.parent.mkdir(parents=True, exist_ok=True)
                with self.trace_path.open("a", encoding="utf-8") as output:
                    output.write(_json_text(clean) + "\n")
            except OSError:
                raise RuntimeError("Could not append resumed stage trace") from None

    def _record_real(self, trace):
        record = {key: deepcopy(value) for key, value in trace.items() if key in _TRACE_FIELDS}
        record.update(call_source="real", response_origin="real_api", request_sent_this_run=True)
        if isinstance(record.get("raw_json"), dict):
            record["raw_json_sha256"] = _hash(_json_text(record["raw_json"]).encode("utf-8"))
        self.real_api_calls += 1
        self._record(record)

    def complete_json(self, stage, frames, payload):
        started = time.perf_counter()
        fingerprint = _fingerprint(self.model, stage, frames, payload)
        if self._position < len(self._prefix):
            line_number, source, record_hash = self._prefix[self._position]
            for field in _FIELDS:
                if source.get(field) != fingerprint[field]:
                    raise ValueError(f"Resume fingerprint does not match: {field}")
            record = {key: deepcopy(value) for key, value in source.items() if key in _TRACE_FIELDS}
            record.update(call_source="cached", response_origin="real_api", request_sent_this_run=False,
                          replay_latency_ms=round((time.perf_counter() - started) * 1000, 3),
                          raw_json_sha256=_hash(_json_text(source["raw_json"]).encode("utf-8")),
                          cache_provenance={**self.resume_metadata, "source_line": line_number,
                                            "source_record_sha256": record_hash})
            if source.get("cache_provenance"):
                record["upstream_cache_provenance"] = deepcopy(source["cache_provenance"])
            self._record(record)
            self._position += 1
            self.cached_calls += 1
            return deepcopy(source["raw_json"])

        before = len(self.provider.traces)
        try:
            result = self.provider.complete_json(stage, frames, payload)
        except Exception:
            emitted = self.provider.traces[before:]
            if len(emitted) == 1:
                self._record_real(emitted[0])
            elif emitted:
                raise RuntimeError("Wrapped stage provider emitted an unexpected number of traces") from None
            raise
        emitted = self.provider.traces[before:]
        if len(emitted) != 1 or emitted[0].get("status") != "success":
            raise RuntimeError("Real stage response is missing its successful API trace")
        self._record_real(emitted[0])
        return result
