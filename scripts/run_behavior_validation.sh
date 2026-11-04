#!/usr/bin/env bash
set -euo pipefail

# Stabilize CPU timing and prevent each simulator worker from spawning its own
# BLAS/OpenMP thread pool.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

python -m stg_ddqn.train \
  --config configs/behavior.yaml \
  --checkpoint checkpoints/behavior_stg_ddqn.pt \
  --output results/behavior_training

python -m stg_ddqn.experiments \
  --config configs/behavior.yaml \
  --checkpoint checkpoints/behavior_stg_ddqn.pt \
  --output results/behavior_dynamic \
  --experiments traffic,mobility \
  --resume \
  --workers 3

# Runtime measurements must not compete with other simulator workers.  Keep
# this sweep sequential, then merge it with the parallel-safe measurements.
python -m stg_ddqn.experiments \
  --config configs/behavior.yaml \
  --checkpoint checkpoints/behavior_stg_ddqn.pt \
  --output results/behavior_scalability \
  --experiments scalability \
  --resume \
  --workers 1

python -m stg_ddqn.merge_results \
  --inputs results/behavior_dynamic,results/behavior_scalability \
  --output results/behavior

python -m stg_ddqn.validate_results \
  --raw results/behavior/evaluation_raw.csv \
  --summary results/behavior/evaluation_summary.csv \
  --strict-curves

python -m stg_ddqn.plot_results \
  --input results/behavior/evaluation_summary.csv \
  --output plots/behavior \
  --basename stg_ddqn_behavior_validation_panels \
  --separate
