"""Small local MCP client for this Unity project when native tools are not loaded."""

import argparse
import json
from pathlib import Path
import urllib.request


URL = "http://127.0.0.1:8080/mcp"


def decode(body):
    text = body.decode("utf-8")
    if text.lstrip().startswith("{"):
        return json.loads(text)
    events = [line[5:].strip() for line in text.splitlines() if line.startswith("data:")]
    if not events:
        return None
    return json.loads(events[-1])


def request(payload, session=None):
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if session:
        headers["Mcp-Session-Id"] = session
        headers["MCP-Protocol-Version"] = "2025-03-26"
    req = urllib.request.Request(URL, json.dumps(payload).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=55) as response:
        return decode(response.read()), response.headers.get("Mcp-Session-Id", session)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool")
    parser.add_argument("--args", type=Path)
    parser.add_argument("--resource")
    parser.add_argument("--list-resources", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    _, session = request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-03-26", "capabilities": {},
        "clientInfo": {"name": "satori-local-development", "version": "0.1"}}})
    request({"jsonrpc": "2.0", "method": "notifications/initialized"}, session)
    if args.tool:
        method, params = "tools/call", {"name": args.tool, "arguments": json.loads(args.args.read_text(encoding="utf-8-sig")) if args.args else {}}
    elif args.resource:
        method, params = "resources/read", {"uri": args.resource}
    elif args.list_resources:
        method, params = "resources/list", {}
    else:
        method, params = "tools/list", {}
    result, _ = request({"jsonrpc": "2.0", "id": 2, "method": method, "params": params}, session)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(str(args.output))
    elif method == "tools/list":
        print(json.dumps([{k: t.get(k) for k in ("name", "description")} for t in result["result"]["tools"]], ensure_ascii=True))
    else:
        print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
