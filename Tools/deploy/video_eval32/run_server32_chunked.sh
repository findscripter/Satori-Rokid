#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${GPURUN_JOB:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo 'Native 32-frame model loading requires gpurun allocation' >&2
    exit 2
fi
TASK_ROOT=/home/gpu-user/work/videochat3-satori
EVAL_ROOT=/home/gpu-user/work/satori-video-eval
cd "$TASK_ROOT"
export HF_MODULES_CACHE="$TASK_ROOT/cache/modules"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$TASK_ROOT/.venv/bin/python" -c '
import json
from pathlib import Path
x=json.loads(Path("evidence/download-integrity.json").read_text())
assert x["complete"] is True and x["revision"] == "9f6107bdfee9686dfe493444022f5cf79a1a6acf"
assert len(x["verified_files"]) == 28 and all(f["verified"] for f in x["verified_files"])
'
exec "$TASK_ROOT/.venv/bin/python" "$EVAL_ROOT/Tools/deploy/video_eval32/server.py" \
  --backend-file "$EVAL_ROOT/Tools/deploy/video_eval32/backend32_chunked.py" \
  --model-path "$TASK_ROOT/model" \
  --model-id VideoChat3-4B \
  --revision 9f6107bdfee9686dfe493444022f5cf79a1a6acf \
  --metadata-file "$EVAL_ROOT/data32/registry32-all.json" "$@"
