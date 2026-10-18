"""Launch unlabeled GT-benchmark inference using the existing private service."""

import argparse
import json
import os
from pathlib import Path
import sys
import uuid

from gt_benchmark import BenchmarkProvider, infer
from unity_mage_request import PROJECT_ROOT, ensure_service, fetch_service_token


def main():
    parser = argparse.ArgumentParser(description="Run Mage-VL on frozen unlabeled images; no GT argument")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, choices=(8, 32), default=8)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--output-token-field", choices=("max_tokens", "max_completion_tokens"), default="max_tokens")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--expected-response-model")
    args = parser.parse_args()
    name = "SATORI_GT_MAGE_" + uuid.uuid4().hex.upper()
    try:
        ensure_service(PROJECT_ROOT)
        os.environ[name] = fetch_service_token()
        provider = BenchmarkProvider(model="Mage-VL", base_url="http://127.0.0.1:18866/v1", api_key_env=name,
                                     frame_count=args.frames, timeout=args.timeout,
                                     max_output_tokens=args.max_output_tokens, output_token_field=args.output_token_field,
                                     temperature=args.temperature, expected_response_model=args.expected_response_model)
        result = infer(args.manifest, args.taxonomy, args.output, provider,
                       progress=lambda row: print(json.dumps(row), flush=True), frame_count=args.frames)
        print(json.dumps({key: result[key] for key in (
            "status", "planned_samples", "successful_samples", "failed_samples", "api_calls"
        )}), flush=True)
        return 0 if result["failed_samples"] == 0 else 2
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}), flush=True)
        return 2
    finally:
        os.environ.pop(name, None)


if __name__ == "__main__":
    sys.exit(main())
