import json
import os
from pathlib import Path
import time

from satori_repro.models import validate_observation
from satori_repro.providers import OpenAICompatibleProvider


root = Path(__file__).resolve().parents[1]
sample = root / "Research/Data/real-task/clean_surface_0"
session = json.loads((sample / "session.json").read_text(encoding="utf-8"))
task = json.loads((sample / "task.json").read_text(encoding="utf-8"))
frame = dict(session["frames"][0])
frame["image_path"] = str((sample / frame["image_path"]).resolve())
# Endpoint and key come from the environment, e.g. SATORI_BASE_URL=https://your-endpoint.example/v1.
provider = OpenAICompatibleProvider(model="gpt-5.4-mini", base_url=os.environ["SATORI_BASE_URL"], api_key_env="SATORI_API_KEY", timeout=90)
start = time.perf_counter()
observation = provider.infer([frame], task=task, history=[], state=None)
validate_observation(observation, task, [frame["frame_id"]])
result = {"requested_model": provider.model, "endpoint": provider.base_url, "synthetic": False,
          "source_kind": "public_dataset_video", "elapsed_seconds": round(time.perf_counter() - start, 3),
          "frame_id": frame["frame_id"], "observation": observation}
out = root / "Evidence/vision-api-smoke.json"
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=True))
