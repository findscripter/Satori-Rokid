import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

entry = json.loads((Path(__file__).parent / "temporary-direct-url.json").read_text())
started = time.perf_counter()
try:
    request = urllib.request.Request(entry["url"], headers={"Range": "bytes=0-4194303"})
    with urllib.request.urlopen(request, timeout=20) as response:
        data = response.read(4 * 1024 * 1024)
        output = {"status": response.status, "host": urllib.parse.urlparse(response.url).hostname,
                  "bytes": len(data), "seconds": time.perf_counter() - started,
                  "content_range": response.headers.get("Content-Range")}
except Exception as error:
    output = {"error_type": type(error).__name__, "seconds": time.perf_counter() - started}
print(json.dumps(output))
