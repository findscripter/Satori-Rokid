"""Query only model IDs from an explicitly provided API; never log credentials."""
import argparse
import json
import os
from pathlib import Path
from urllib import request, error


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    key = os.environ.get("SATORI_API_KEY")
    if not key:
        raise SystemExit("SATORI_API_KEY is not configured in this process")
    endpoint = args.base_url.rstrip("/") + "/models"
    req = request.Request(endpoint, headers={"Authorization": "Bearer " + key, "Accept": "application/json"})
    try:
        with request.build_opener(NoRedirect()).open(req, timeout=30) as response:
            data = json.loads(response.read(8 * 1024 * 1024))
    except error.HTTPError as exc:
        raise SystemExit(f"Model listing returned HTTP {exc.code}") from None
    except (error.URLError, OSError, ValueError):
        raise SystemExit("Model listing connection or JSON parsing failed") from None
    models = [item["id"] for item in data.get("data", []) if isinstance(item, dict) and isinstance(item.get("id"), str)]
    result = {"endpoint": endpoint, "model_ids": models}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    candidates = [m for m in models if any(t in m.lower() for t in ("gpt-4o", "gpt-4.1", "gemini", "qwen", "claude", "gpt-5"))]
    print(json.dumps({"model_count": len(models), "candidate_ids": candidates[:70]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
