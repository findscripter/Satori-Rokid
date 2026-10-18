"""Run a frozen unlabeled benchmark with a process-private API credential.

Use --prompt-for-key to enter a credential through a non-echoing console.
Credentials never enter argv, files, Unity, or a global environment setting.
"""

import argparse
import getpass
import json
import os
from pathlib import Path
import sys
import uuid

from gt_benchmark import BenchmarkProvider, infer


def main():
    parser = argparse.ArgumentParser(description="One API attempt per frozen unlabeled sample; no GT input")
    for name in ("manifest", "taxonomy", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--expected-response-model")
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--output-token-field", choices=("max_tokens", "max_completion_tokens"), default="max_tokens")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--api-key-env", default="SATORI_API_KEY")
    parser.add_argument("--prompt-for-key", action="store_true")
    args = parser.parse_args()
    environment_name = "SATORI_GT_API_" + uuid.uuid4().hex.upper()
    try:
        key = getpass.getpass("API credential: ") if args.prompt_for_key else os.environ.get(args.api_key_env)
        if not key or not key.isascii() or any(char.isspace() for char in key):
            raise ValueError("A process credential is required")
        os.environ[environment_name] = key
        provider = BenchmarkProvider(model=args.model, base_url=args.base_url, api_key_env=environment_name,
                                     expected_response_model=args.expected_response_model,
                                     max_output_tokens=args.max_output_tokens,
                                     output_token_field=args.output_token_field,
                                     temperature=args.temperature, timeout=args.timeout)
        result = infer(args.manifest, args.taxonomy, args.output, provider,
                       progress=lambda row: print(json.dumps(row), flush=True))
        print(json.dumps({field: result[field] for field in
                          ("status", "planned_samples", "successful_samples", "failed_samples", "api_calls")}), flush=True)
        return 0 if result["failed_samples"] == 0 else 2
    except Exception as exc:
        # Never expose arbitrary upstream errors or credentials through stderr.
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}), flush=True)
        return 2
    finally:
        os.environ.pop(environment_name, None)


if __name__ == "__main__":
    sys.exit(main())
