#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
task_python="${PYTHON_BIN:-.venv/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
"$task_python" -u -m stg_ddqn.full_study "$@"
