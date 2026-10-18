"""Measure accepted prompt sizes through the real authenticated vision endpoint.

Uses one public research image and deterministic filler. This is a capacity and
simple marker-retrieval smoke, not a benchmark of long-video understanding.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
from urllib import error, request

from gt_benchmark import _capture_http_error_metadata
from unity_mage_request import PROJECT_ROOT, ensure_service, fetch_service_token


MARKER = "SATORI_CONTEXT_START_7F3A9B2C"


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_payload(repeats, image_url, *, max_tokens=64):
    text = ("The unique start marker is " + MARKER + ".\n"
            + " neutral" * repeats
            + "\nReturn only the unique start marker from the beginning. Do not describe the image.")
    return {"model": "Mage-VL", "stream": False, "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": text}]}]}


def call(payload, token, *, timeout=900):
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    trace = {"request_body_sha256": hashlib.sha256(raw).hexdigest(), "request_bytes": len(raw)}
    req = request.Request("http://127.0.0.1:18866/v1/chat/completions", data=raw,
                          headers={"Content-Type": "application/json", "Authorization": "Bearer " + token})
    started = time.monotonic()
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read(2 * 1024 * 1024 + 1)
            if len(body) > 2 * 1024 * 1024:
                raise ValueError("Response exceeds the diagnostic limit")
        result = json.loads(body)
        if result.get("model") != "Mage-VL":
            raise ValueError("Unexpected returned model")
        choice = result["choices"][0]
        text = choice["message"]["content"]
        if not isinstance(text, str):
            raise ValueError("Invalid generated text type")
        trace.update(status="completed", model=result["model"], usage=result.get("usage"),
                     finish_reason=choice["finish_reason"], marker_exact_match=text.strip()==MARKER,
                     answer=text if isinstance(text,str) and token not in text else "[withheld]")
    except error.HTTPError as exc:
        trace.update(status="failed", http_status=exc.code)
        try:
            _capture_http_error_metadata(exc, trace)
        finally:
            exc.close()
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        trace.update(status="failed", error_type=type(exc).__name__)
    trace["latency_ms"] = round((time.monotonic()-started)*1000, 3)
    return trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=PROJECT_ROOT / "Research/Data/real-task/clean_surface_0_dense_2fps/frame_000018.jpg")
    parser.add_argument("--context-budget", type=int, default=96000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    ensure_service(PROJECT_ROOT)
    token = fetch_service_token()
    image_bytes = args.image.read_bytes()
    image_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")
    req = request.Request("http://127.0.0.1:18866/health", headers={"Authorization": "Bearer " + token})
    with request.urlopen(req, timeout=30) as response:
        health = json.loads(response.read())
    report = {"health": health, "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
              "context_budget": args.context_budget, "capacity_probes": [],
              "scope": "Real input-capacity and simple marker-retrieval smoke; not evidence of long-video accuracy"}
    save(args.output / "probe.json", report)
    overhead = None
    for target in (32768, 65536, args.context_budget):
        repeats = target - 2048 if overhead is None else target - 320 - overhead
        trace = call(build_payload(repeats, image_url), token)
        trace.update(target_total_budget=target, filler_repeats=repeats)
        report["capacity_probes"].append(trace)
        save(args.output / "probe.json", report)
        print(json.dumps({key: trace.get(key) for key in ("status", "target_total_budget", "usage", "marker_exact_match", "latency_ms", "http_status", "server_error_code")}), flush=True)
        if trace["status"] != "completed":
            return 2
        actual = trace["usage"]["prompt_tokens"]
        observed = actual - repeats
        if overhead is not None and observed != overhead:
            raise ValueError("Tokenizer overhead changed; do not guess the maximum prompt size")
        overhead = observed
    # Both requests should be inexpensive: one checks accepted output allowance,
    # the other exceeds the token budget and must be rejected before GPU generation.
    report["max_output_parameter_probe"] = call(build_payload(0, image_url, max_tokens=8192), token)
    report["over_context_rejection_probe"] = call(build_payload(args.context_budget, image_url), token)
    save(args.output / "probe.json", report)
    print(json.dumps({"completed_capacity_probes": len(report["capacity_probes"]),
                      "largest_actual_prompt": report["capacity_probes"][-1]["usage"]["prompt_tokens"],
                      "output_8192_accepted": report["max_output_parameter_probe"]["status"] == "completed",
                      "over_context_result": report["over_context_rejection_probe"]}), flush=True)
    token = None
    rejected = report["over_context_rejection_probe"]
    return 0 if (report["max_output_parameter_probe"]["status"] == "completed"
                 and rejected["status"] == "failed" and rejected.get("http_status") in (400, 413)
                 and rejected.get("server_error_code") == "context_length_exceeded") else 2


if __name__ == "__main__":
    raise SystemExit(main())
