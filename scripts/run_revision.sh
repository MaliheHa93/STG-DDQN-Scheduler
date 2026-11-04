#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
task_python="${PYTHON:-python}"
"$task_python" -m unittest discover -s tests -v
"$task_python" -m stg_ddqn.train --config configs/revision_validation.yaml \
  --checkpoint checkpoints/revision_stg_ddqn.pt --output results/revision_training
"$task_python" -m stg_ddqn.experiments --config configs/revision_validation.yaml \
  --checkpoint checkpoints/revision_stg_ddqn.pt --output results/revision_validation --workers 1
"$task_python" -m stg_ddqn.validate_results \
  --raw results/revision_validation/evaluation_raw.csv \
  --summary results/revision_validation/evaluation_summary.csv --strict-curves
if command -v matlab >/dev/null 2>&1; then
  matlab -batch "run('matlab/render_revision.m')"
else
  printf '%s\n' 'Measured CSVs are ready. MATLAB is unavailable; run matlab/render_revision.m in MATLAB.'
fi
