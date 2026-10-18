"""Run one experiment while recording a sanitized HTTP failure category.

No request bodies, credentials or successful response bodies are written here.
This diagnostic wrapper does not retry requests.
"""
import json
import os
from pathlib import Path
import re
import runpy
import sys
from urllib import error, request


original_build_opener = request.build_opener
destination = Path(__file__).resolve().parents[1] / "Evidence/api-failure-diagnostic.json"


class DiagnosticOpener:
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def open(self, req, **kwargs):
        try:
            return self.wrapped.open(req, **kwargs)
        except error.HTTPError as failure:
            diagnostic = {"http_status": failure.code,
                          "retry_after": failure.headers.get("Retry-After")}
            try:
                body = json.loads(failure.read(16000).decode("utf-8"))
                detail = body.get("error", {})
                if isinstance(detail, dict):
                    for field in ("type", "code", "message"):
                        value = detail.get(field)
                        if isinstance(value, str):
                            key = os.environ.get("SATORI_API_KEY", "")
                            if key:
                                value = value.replace(key, "[REDACTED]")
                            diagnostic[field] = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", value)[:1000]
            except (ValueError, UnicodeError, AttributeError):
                diagnostic["body"] = "No parseable error object; raw body discarded."
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(diagnostic, ensure_ascii=True), flush=True)
            raise


def diagnostic_build_opener(*handlers):
    return DiagnosticOpener(original_build_opener(*handlers))


request.build_opener = diagnostic_build_opener
sys.argv[0] = str(Path(__file__).with_name("run_satori_api_experiment.py"))
runpy.run_path(sys.argv[0], run_name="__main__")
