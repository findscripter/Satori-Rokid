"""One small explicitly configured route probe, with credential redaction."""
import argparse
import json
import os
from pathlib import Path
import time
from urllib import request, error
from satori_repro.providers import _NoRedirect, _image_data_url

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint, e.g. https://your-endpoint.example/v1")
parser.add_argument("--image", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
key = os.environ["SATORI_API_KEY"]
content = [{"type": "text", "text": "Describe the visible hand action and objects in one short sentence. If no image is supplied, return the word ready."}]
if args.image:
    content.append({"type": "image_url", "image_url": {"url": _image_data_url(args.image)}})
payload = {"model": args.model, "stream": False, "messages": [{"role": "user", "content": content}]}
req = request.Request(args.base_url.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
                      headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
start = time.perf_counter()
try:
    with request.build_opener(_NoRedirect()).open(req, timeout=120) as response:
        body = json.loads(response.read(2 * 1024 * 1024))
        result = {"status": response.status, "requested_model": args.model, "response_model": body.get("model"),
                  "usage": body.get("usage"), "content": body.get("choices", [{}])[0].get("message", {}).get("content")}
except error.HTTPError as exc:
    raw = exc.read(16000).decode("utf-8", "replace").replace(key, "[REDACTED]")
    result = {"status": exc.code, "requested_model": args.model, "retry_after": exc.headers.get("Retry-After"), "error": raw}
except (OSError, error.URLError, ValueError) as exc:
    result = {"status": "transport-error", "requested_model": args.model, "error_type": type(exc).__name__}
result["elapsed_seconds"] = round(time.perf_counter() - start, 3)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=True))
