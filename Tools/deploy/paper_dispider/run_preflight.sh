#!/usr/bin/env bash
set -euo pipefail
test -n "${GPURUN_JOB:-}"
test -n "${CUDA_VISIBLE_DEVICES:-}"
export DISPIDER_RUNTIME=/home/gpu-user/work/dispider-satori/runtime
export PYTHONDONTWRITEBYTECODE=1
cd /home/gpu-user/work/dispider-satori
.venv/bin/python restore_model.py
cd /home/gpu-user/work/satori-dispider-paper-20261018
exec /home/gpu-user/work/dispider-satori/.venv/bin/python -u Tools/deploy/paper_dispider/run.py \
  --manifest Research/Data/dispider-paper-preflight-20261018/infer-manifest.json \
  --taxonomy Research/Data/dispider-paper-preflight-20261018/taxonomy.json \
  --protocol dispider-protocols-20261018/preflight.json \
  --model-path /dev/shm/gpu-host-satori-dispider-7c5517e \
  --output runs/dispider-non-gt-preflight-20261018
