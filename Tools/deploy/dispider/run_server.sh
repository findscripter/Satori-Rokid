#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ -z "${GPURUN_JOB:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == *,* || "${CUDA_VISIBLE_DEVICES}" == "-1" ]]; then
    echo 'Dispider requires a single GPU allocated by gpurun; backend uses logical cuda:0.' >&2
    exit 2
fi
if [[ $# != 2 || "$1" != "--token-file" ]]; then
    echo 'Usage: gpurun -n 1 -d bash run_server.sh --token-file /absolute/private/token-file' >&2
    exit 2
fi
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
eval_root=/home/gpu-user/work/satori-video-eval
token_file="$2"
model_root=/dev/shm/gpu-host-satori-dispider-7c5517e
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
export HF_HOME="$task_root/cache/hf" HF_MODULES_CACHE="$task_root/cache/modules"
export DISPIDER_RUNTIME="$task_root/runtime" DISPIDER_ATTN_IMPLEMENTATION=sdpa
export PYTHONPATH="$task_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$task_root/evidence" "$task_root/logs" "$task_root/cache/modules"
cd "$task_root"
"$task_root/.venv/bin/python" "$task_root/restore_model.py" --project-root "$task_root"
stamp="$(date -u +%Y%m%dT%H%M%S)-${GPURUN_JOB}-$$"
export DISPIDER_SERVER_STARTUP_RECORD="$task_root/evidence/server-start-${stamp}.json"
"$task_root/.venv/bin/python" - "$token_file" <<'PY'
import json, os, stat, sys, time
from pathlib import Path
token = Path(sys.argv[1])
if not token.is_absolute() or not token.is_file() or token.is_symlink():
    raise SystemExit('Token must be an existing absolute private file, without a symlink.')
info = token.stat()
if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
    raise SystemExit('Token file must belong to the current user and have 0600 permissions.')
record = {'status': 'starting', 'scheduler_job': os.environ['GPURUN_JOB'],
          'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'logical_device': 'cuda:0',
          'model_id': 'Dispider', 'hf_revision': '7c5517eaaa4dc3788602af40e077063792c75f6c',
          'source_revision': '803f6b4634b3232293e89a38bcce9f5815c33ed4',
          'attention_backend': 'sdpa', 'original_official_runtime_stack': False,
          'listen': '127.0.0.1:18868', 'registry': '/home/gpu-user/work/satori-video-eval/data/registry-all.json',
          'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
Path(os.environ['DISPIDER_SERVER_STARTUP_RECORD']).write_text(json.dumps(record, indent=2))
print(json.dumps({'event': 'server_start', **record}), flush=True)
PY
# Preserve gpurun's physical-device mapping; only the backend selects logical cuda:0.
exec "$task_root/.venv/bin/python" "$eval_root/Tools/deploy/video_eval/server.py" \
    --backend-file "$task_root/backend.py" \
    --model-path "$model_root" \
    --model-id Dispider \
    --revision 7c5517eaaa4dc3788602af40e077063792c75f6c \
    --token-file "$token_file" \
    --metadata-file "$eval_root/data/registry-all.json" \
    --audit-file "$task_root/evidence/http-audit-${stamp}.jsonl" \
    --port 18868
