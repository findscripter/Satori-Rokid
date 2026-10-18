"""Bounded, redacted HTTP error evidence for the unchanged paper provider.

This wrapper never retries, follows redirects, changes payloads or schedules a
request. Use the caller's frozen scheduler. No credential is persisted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from urllib import error

import satori_paper_benchmark as bench


_OPENER_LOCK = threading.RLock()
BODY_LIMIT = 8192
HEADER_WHITELIST = frozenset({
    "retry-after", "ratelimit-limit", "ratelimit-remaining", "ratelimit-reset", "ratelimit-policy",
    "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset",
    "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests",
    "x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens",
    "anthropic-ratelimit-requests-limit", "anthropic-ratelimit-requests-remaining", "anthropic-ratelimit-requests-reset",
    "anthropic-ratelimit-tokens-limit", "anthropic-ratelimit-tokens-remaining", "anthropic-ratelimit-tokens-reset",
})


def _redact(value, credential):
    text = str(value)
    if credential:
        text = text.replace(credential, "[REDACTED]")
    text = re.sub(r"(?i)\bBearer\s+[^\s,;]+", "Bearer [REDACTED]", text)
    text = re.sub(r"\b(?:sk-|rck_)[A-Za-z0-9_.-]{6,}", "[REDACTED]", text)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", text)
    return text


def numeric_retry_after(value):
    """HTTP delta-seconds only; date forms are retained as headers, not guessed."""
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,10}", value.strip()):
        return int(value.strip())
    return None


def _capture_http_error(exc, credential):
    raw = exc.read(BODY_LIMIT)
    headers = {str(key).lower(): _redact(value, credential)[:512]
               for key, value in (exc.headers.items() if exc.headers else [])
               if str(key).lower() in HEADER_WHITELIST}
    result = {"http_status": exc.code, "headers": headers, "captured_body_bytes": len(raw),
              "body_capture_limit_bytes": BODY_LIMIT, "body_may_be_truncated": len(raw) == BODY_LIMIT,
              "captured_body_sha256": hashlib.sha256(raw).hexdigest(),
              "body_hash_scope": "captured prefix only; no raw body persisted"}
    delay = numeric_retry_after(headers.get("retry-after"))
    if delay is not None:
        result["retry_after_seconds"] = delay
    try:
        body = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError):
        result["body_format"] = "non_json_or_truncated"
    else:
        result["body_format"] = "json"
        detail = body.get("error", body) if isinstance(body, dict) else None
        if isinstance(detail, dict):
            result["error"] = {key: _redact(detail[key], credential)[:2048] for key in ("type", "code", "message")
                               if key in detail and isinstance(detail[key], (str, int, float, bool))}
    return result


class DiagnosticPaperProvider(bench.PaperBenchmarkProvider):
    """Use the original provider's exact one-attempt request and prediction logic."""

    def predict(self, sample, taxonomy):
        with _OPENER_LOCK:
            original = bench.request.build_opener
            owner_thread = threading.get_ident()
            before = len(self.traces)
            observed = []
            credential = os.environ.get(self.api_key_env)

            class CaptureOpener:
                def __init__(self, inner):
                    self.inner = inner

                def open(self, *args, **kwargs):
                    try:
                        return self.inner.open(*args, **kwargs)
                    except error.HTTPError as exc:
                        try:
                            observed.append(_capture_http_error(exc, credential))
                        except Exception as capture_error:
                            observed.append({"http_status": exc.code,
                                             "capture_error_type": type(capture_error).__name__})
                        raise

            def build_opener(*args, **kwargs):
                opener = original(*args, **kwargs)
                # Unrelated threads retain their original behavior and cannot
                # contribute requests to this sample's diagnostic record.
                return CaptureOpener(opener) if threading.get_ident() == owner_thread else opener

            bench.request.build_opener = build_opener
            try:
                return super().predict(sample, taxonomy)
            finally:
                bench.request.build_opener = original
                emitted = self.traces[before:]
                if observed and len(emitted) == 1:
                    emitted[0]["http_error_diagnostics"] = observed[0]
