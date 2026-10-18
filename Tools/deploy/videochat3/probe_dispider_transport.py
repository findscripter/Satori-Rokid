"""Small public-CDN probes only; never logs signed URLs or writes weight data."""
import argparse
import json
from pathlib import Path
import subprocess
import urllib.parse

parser = argparse.ArgumentParser()
parser.add_argument("--url-map", required=True)
parser.add_argument("--resolve-ip", action="append")
args = parser.parse_args()
mapping = json.loads(Path(args.url_map).read_text())
url = next(iter(mapping.values()))
if urllib.parse.urlparse(url).hostname != "us.aws.cdn.hf.co":
    raise ValueError("unexpected_public_cdn")
variants = [("--http1.1", ip) for ip in args.resolve_ip] if args.resolve_ip else [("--http1.1", None), ("--http2", None)]
for protocol, ip in variants:
    config = 'url = ' + json.dumps(url) + '\nrange = "4194304-8388607"\n'
    command = ["curl", "--config", "-", protocol, "--silent", "--connect-timeout", "10", "--max-time", "30",
               "--output", "/dev/null", "--write-out",
               '{"status":%{http_code},"http":"%{http_version}","bytes":%{size_download},"seconds":%{time_total},"connect":%{time_connect},"first_byte":%{time_starttransfer}}']
    if ip:
        command.extend(["--resolve", f"us.aws.cdn.hf.co:443:{ip}"])
    result = subprocess.run(command, input=config, text=True, capture_output=True)
    try:
        output = json.loads(result.stdout)
    except ValueError:
        output = {"invalid_curl_metadata": True}
    output.update({"protocol_option": protocol, "return_code": result.returncode,
                   "host": "us.aws.cdn.hf.co", "resolve_ip": ip})
    print(json.dumps(output), flush=True)
