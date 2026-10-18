"""CPU HTTP client for long-context capacity probes on the model server host.

All GPU work remains in the existing gpurun-managed service. Large filler text
is constructed on the server, avoiding bulk transfer through an SSH tunnel.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
from urllib import error, request


MARKER = "SATORI_CONTEXT_START_7F3A9B2C"
BASE = "http://127.0.0.1:18866"


def write(path, value):
    temporary = path.with_suffix(".writing")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def payload(repeats, image_url, *, max_tokens=64):
    text = ("The unique start marker is " + MARKER + ".\n" + " neutral" * repeats
            + "\nReturn only the unique start marker from the beginning. Do not describe the image.")
    return {"model": "Mage-VL", "stream": False, "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": text}]}]}


def safe_error(exc):
    result = {"status": "failed", "http_status": exc.code}
    raw = exc.read(65537)
    if len(raw) > 65536:
        return {**result, "error_body_status": "too_large"}
    try:
        body = json.loads(raw)
        detail = body.get("error", {})
        for name in ("code", "finish_reason"):
            value = detail.get(name)
            if isinstance(value, str) and len(value) <= 64 and all(c.islower() or c=="_" for c in value):
                result["server_" + name] = value
        budget = detail.get("budget")
        if isinstance(budget, dict):
            result["budget"] = {k: v for k, v in budget.items()
                                if k in {"input_tokens", "reserved_output", "max_context", "operational_context",
                                         "model_context_window", "tokenizer_context_window"}
                                and type(v) is int and v >= 0}
        usage = body.get("usage")
        if isinstance(usage, dict):
            result["server_usage"] = {k: v for k, v in usage.items()
                                      if k in {"prompt_tokens", "completion_tokens", "total_tokens"}
                                      and type(v) is int and v >= 0}
    except (ValueError, TypeError, AttributeError):
        result["error_body_status"] = "invalid"
    return result


def call(body, token):
    raw = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    result = {"request_body_sha256": hashlib.sha256(raw).hexdigest(), "request_bytes": len(raw)}
    req = request.Request(BASE+"/v1/chat/completions", data=raw,
                          headers={"Content-Type": "application/json", "Authorization": "Bearer " + token})
    started = time.monotonic()
    try:
        with request.urlopen(req, timeout=900) as response:
            data = response.read(2*1024*1024+1)
        if len(data) > 2*1024*1024:
            raise ValueError("Oversized response")
        response = json.loads(data)
        choice = response["choices"][0]
        text = choice["message"]["content"]
        if response.get("model") != "Mage-VL" or not isinstance(text, str):
            raise ValueError("Invalid response")
        result.update(status="completed", model=response["model"], usage=response.get("usage"),
                      finish_reason=choice["finish_reason"], marker_exact_match=text.strip()==MARKER,
                      answer=text[:2048] if token not in text else "[withheld]")
    except error.HTTPError as exc:
        try:
            result.update(safe_error(exc))
        finally:
            exc.close()
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        result.update(status="failed", error_type=type(exc).__name__)
    result["latency_ms"] = round((time.monotonic()-started)*1000, 3)
    return result


def health(token):
    req = request.Request(BASE+"/health", headers={"Authorization": "Bearer " + token})
    with request.urlopen(req, timeout=30) as response:
        return json.loads(response.read(65536))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", type=int, nargs="+", default=[32768,65536,96000])
    parser.add_argument("--operational-budget", type=int, default=96000)
    parser.add_argument("--known-token-overhead", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    token = (args.project/".secrets/api.token").read_text().strip()
    if not 16 <= len(token) <= 512 or not token.isascii() or any(c.isspace() for c in token):
        raise ValueError("Invalid private credential")
    image = args.image.read_bytes()
    image_url = "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")
    report = {"transport": "server_loopback_cpu_client", "gpu_owner": "existing gpurun service only",
              "health_before": health(token), "image_sha256": hashlib.sha256(image).hexdigest(),
              "capacity_probes": [], "scope": "Capacity and synthetic repeated-filler marker smoke; not long-video accuracy"}
    output = args.output/"probe.json"
    write(output, report)
    overhead = args.known_token_overhead
    for target in args.targets:
        repeats = target-2048 if overhead is None else target-320-overhead
        trace = call(payload(repeats, image_url), token)
        trace.update(target_total_budget=target, filler_repeats=repeats)
        report["capacity_probes"].append(trace)
        write(output, report)
        print(json.dumps(trace), flush=True)
        if trace["status"] != "completed":
            report["status"] = "capacity_probe_failed"
            successful = [p["usage"]["prompt_tokens"] for p in report["capacity_probes"] if p["status"]=="completed"]
            report["largest_actual_input_tokens"] = max(successful) if successful else None
            report["health_after"] = health(token)
            write(output, report)
            return 2
        actual = trace["usage"]["prompt_tokens"]
        observed = actual-repeats
        if overhead is not None and observed != overhead:
            raise ValueError("Tokenizer calibration changed")
        overhead = observed
    report["short_marker_control"] = call(payload(0, image_url, max_tokens=64), token)
    report["max_output_parameter_probe"] = call(payload(0, image_url, max_tokens=8192), token)
    report["over_context_rejection_probe"] = call(payload(args.operational_budget, image_url), token)
    report["health_after"] = health(token)
    rejected = report["over_context_rejection_probe"]
    report["status"] = ("passed_capacity_checks" if report["max_output_parameter_probe"]["status"]=="completed"
                        and rejected["status"]=="failed" and rejected.get("http_status") in (400,413)
                        and rejected.get("server_code")=="context_length_exceeded" else "boundary_check_failed")
    report["largest_actual_input_tokens"] = report["capacity_probes"][-1]["usage"]["prompt_tokens"]
    report["marker_matches"] = sum(p["marker_exact_match"] for p in report["capacity_probes"])
    write(output, report)
    print(json.dumps({"status":report["status"],"largest_actual_input_tokens":report["largest_actual_input_tokens"],
                      "marker_matches":report["marker_matches"]}), flush=True)
    return 0 if report["status"]=="passed_capacity_checks" else 2


if __name__ == "__main__":
    raise SystemExit(main())
