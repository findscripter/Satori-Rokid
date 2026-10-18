#!/usr/bin/env bash
set -euo pipefail
test -n "${GPURUN_JOB:-}"
test -n "${CUDA_VISIBLE_DEVICES:-}"
export DISPIDER_RUNTIME=/home/gpu-user/work/dispider-satori/runtime
export PYTHONDONTWRITEBYTECODE=1
cd /home/gpu-user/work/dispider-satori
.venv/bin/python restore_model.py
cd /home/gpu-user/work/satori-dispider-paper-20261018
exec /home/gpu-user/work/dispider-satori/.venv/bin/python -u Tools/deploy/paper_dispider/run_suite.py
