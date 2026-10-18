#!/usr/bin/env bash
set -euo pipefail
project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${GPURUN_JOB:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  printf 'Start this service through: gpurun -n 1 -d bash %s/run_server.sh\n' "$project" >&2
  exit 2
fi
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$project/hf-home" HF_MODULES_CACHE="$project/hf-home/modules"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
export TMPDIR="$project/tmp"
cd "$project"
"$project/.venv/bin/python" - <<'PY'
import json, os
from pathlib import Path
p = Path.cwd()
manifest = json.loads((p / 'model/mage-download-manifest.json').read_text())
if manifest.get('status') != 'completed':
    raise SystemExit('The model download has not completed its SHA256 verification.')
state = {'scheduler_job': os.environ['GPURUN_JOB'],
         'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
         'hf_reference_revision': manifest['hf_reference_revision'],
         'verified_files': len(manifest['files']),
         'model_dir': str((p / 'model').resolve()), 'listen': '127.0.0.1:18866'}
(p / 'evidence/scheduler.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
print(json.dumps({'event': 'scheduler_assignment', **state}), flush=True)
PY
# Configured admission ceilings, not validated GPU capacity or native 1M support.
# Additional CLI arguments override these defaults; server hard limits still apply.
exec "$project/.venv/bin/python" "$project/server.py" \
  --model-dir "$project/model" --served-model-name Mage-VL \
  --revision d88b153285f1633a61b2f693c59c8576693af185 \
  --token-file "$project/.secrets/api.token" \
  --manifest-out "$project/evidence/deployment-manifest.json" \
  --max-context-tokens 96000 --max-new-tokens 8192 --default-new-tokens 4096 \
  --max-text-chars 2000000 --max-images 128 \
  --host 127.0.0.1 --port 18866 "$@"
