#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
profile="${1:-paper}"
stage="${2:-main}"
case "$profile" in paper|validation) ;; *) echo 'Profile: paper or validation' >&2; exit 2;; esac
case "$stage" in main|extended|generate|train|evaluate) ;; *) echo 'Stage: main, extended, generate, train, or evaluate' >&2; exit 2;; esac
python_bin="${PYTHON_BIN:-.venv/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/stg-sumo-matplotlib"
config="configs/sumo_${profile}.yaml"
out="${SUMO_OUTPUT_DIR:-results/sumo_${profile}}"
checkpoint="checkpoints/sumo_stg_ddqn.pt"
if [ "$profile" = validation ]; then checkpoint="checkpoints/sumo_validation_stg_ddqn.pt"; fi
checkpoint="${SUMO_CHECKPOINT:-$checkpoint}"
mkdir -p "$out"
if [ "$stage" = main ] || [ "$stage" = generate ]; then
    "$python_bin" -u -m stg_ddqn.sumo_scenario --config "$config" --resume
fi
if [ "$stage" = main ] || [ "$stage" = train ]; then
    "$python_bin" -u -m stg_ddqn.train --config "$config" --checkpoint "$checkpoint" --output "${out}_training"
fi
if [ "$stage" = main ] || [ "$stage" = evaluate ]; then
    "$python_bin" -u -m stg_ddqn.experiments --config "$config" --checkpoint "$checkpoint" --output "$out" --resume
    "$python_bin" -m stg_ddqn.validate_results --raw "$out/evaluation_raw.csv" --summary "$out/evaluation_summary.csv" --strict-curves
fi
if [ "$stage" = extended ]; then
    for study in reward bursty ablation; do
        "$python_bin" -u -m stg_ddqn.revision_study --config "$config" --study "$study" --output "$out/$study"
    done
fi
echo "MATLAB: addpath('matlab'); render_sumo_results('$out','plots/sumo_${profile}_matlab');"
