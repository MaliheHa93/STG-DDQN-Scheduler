#!/usr/bin/env bash
set -euo pipefail

python -m stg_ddqn.selftest
python -m stg_ddqn.experiments \
  --config configs/smoke.yaml \
  --output results/smoke \
  --experiments traffic,mobility,scalability
python -m stg_ddqn.validate_results \
  --raw results/smoke/evaluation_raw.csv \
  --summary results/smoke/evaluation_summary.csv
# Smoke results validate execution and accounting only.  The strict plotting
# command intentionally rejects this two-point/one-seed grid as paper evidence.
