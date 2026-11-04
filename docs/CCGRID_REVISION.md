# CCGrid revision: implementation and evidence

## Changes implemented

- SUMO speed sweeps select `mobility.sumo_csv_by_speed`; missing or identical-content traces fail explicitly. Each run records the selected trace's SHA-256 fingerprint. A label or normalization change is not treated as a changed trajectory.
- Migration routing now uses Dijkstra on the capacity-filtered graph instead of enumerating simple paths to find a single route. Saturated edges are excluded explicitly, allowing a positive-capacity alternative to be found rather than rejecting a saturated shortcut.
- Candidate search prunes bandwidth-infeasible paths and CPU/delay-infeasible assignment branches. It also proves infeasibility immediately when aggregate CPU is insufficient, or when RETAIN fails and no single VNF transfer can fit within the remaining lifetime. These are necessary-condition checks, not timeouts; exhaustive fallback can still have exponential worst-case cost.
- Candidate fallback is triggered by the absence of fully migration-feasible actions, including when preliminary service-feasible plans all fail transfer or remaining-lifetime checks. Fallback stops after collecting fully screened actions, exhausting route enumeration, or reaching the explicitly configured route budget.
- A vehicle still present in the mobility trace but without a covering RSU experiences `coverage_gap`, a continuity failure. Actual disappearance from the trace and lifetime expiry remain distinct normal termination reasons. A truncated or incomplete SUMO trace can still cause artificial departures: validate the exported trace before scientific use.
- Constant-velocity ray–circle coverage exit replaces radial extrapolation by default. `mobility.dwell_estimator: radial` retains the old estimator for a controlled ablation. Neither estimator predicts road turns or closest-RSU reassociation within overlapping coverage. The changed feature requires fresh training.
- Actual transferred payload bytes are accumulated during preparation, including partial transfers and canceled attempts. `migration_volume_total_mb` and `migration_mb_per_admitted_sfc` now use these actual bytes. `completed_migration_volume_total_mb` separately retains successful-remap payload volume. Schema version is 3; do not merge these results with schema-2 results.
- Request audit records include observation duration, admission/end timestamps, completed migration volume, canceled bytes and cancellations. Aggregates separate lifetime expiry, vehicle departure, coverage failure, and measurement-boundary exclusion, and include failures per observed service-hour.
- Temporary destination CPU, new-only service-route reservations, transfer-bandwidth reservations and simultaneous pending migrations are measured. Integrals use the simulator's epoch-held allocations. Bandwidth aggregates sum directed-link reservations; a multi-hop transfer therefore contributes once per directed link. They are not end-user payload-byte measures.
- Optional `simulation.audit_temporary_blocking` reruns rejected initial admissions on a copied topology without temporary reservations. This is a snapshot counterfactual, not a deployable early-source-release policy or a full counterfactual simulation. The audit is outside the measured online decision runtime.
- Runtime summaries include candidate construction, policy selection, and fallback runtime separately. Controller wall-clock computation is externally measured and does not advance simulated physical time. This is explicitly recorded as `controller_time_model`; hard real-time orchestration is not established.
- `traffic.arrival_model: mmpp` enables a continuous-time two-state modulated Poisson process integrated within each epoch. The configured low/high rate factors and transition rates preserve the configured long-run mean arrival rate. Defaults are 0.25x/2.5x and 10-s mean high-state duration; the stationary high-state probability is 1/3 and mean low-state duration is 20 s.
- MATLAB plotting accepts evidence labels and optional 95% CI error bars. Additional scripts plot resource/exposure/runtime diagnostics and independently retrained study variants. Valid flat or zero-migration measurements are allowed; structural or numerical errors still fail.

## Search-budget interpretation (material method change)

The revised validation and paper profiles set `candidates.maximum_fallback_paths: 500`. This is a deterministic cap on fallback route prefixes, not a wall-clock deadline. Fully screened feasible actions found within the budget remain usable. If no action is available when the budget expires, blocking or continuity failure is recorded as `search_budget_exhausted`, not as a proof of physical infeasibility. The raw and summary CSVs expose budget-exhaustion decisions, blocked requests and continuity failures separately. Set the limit to `null` for the uncapped reference search.

**Remove the manuscript's unconditional no-false-blocking/no-false-continuity-failure claim when using this budget.** The cap changes controller behavior and can sacrifice feasible admissions or remappings. It bounds the number of fallback route prefixes considered, not total controller time or all lower-level routing work. The final measurements use a freshly retrained model with this setting. Do not mix them with the earlier uncapped exploratory runs.

## What is deliberately still an assumption

Wireless delay remains fixed at 2 ms; the simulator does not model radio fading, retransmissions, packet queueing, RAM, GPUs, controller signaling or failover. These remain manuscript limitations. CPU is expressed in model capacity units, not verified physical cores. Cutover downtime is a configured penalty and is not a packet-level outage trace. The experiment is epoch-discretized; final intervals, handoff cancellation and departure are evaluated at sampled epochs.

IM and DLAPM remain common-model adaptations. Their initial placement uses a shared greedy rule; STG-DDQN's initial placement remains learned. Thus, the full pipeline comparison does not isolate migration-policy superiority. `graph_ddqn` is a temporal-encoder ablation, not an independent reproduction of DRL-GAMO or another literature baseline.

## Local diagnostic rerun

Use a local virtual environment with `requirements.txt`. Limit CPU math libraries to one thread, select CPU inference for reproducible timing, and keep scalability runs sequential on an otherwise idle host.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest discover -s tests -v
python -m stg_ddqn.train --config configs/revision_validation.yaml \
  --checkpoint checkpoints/revision_stg_ddqn.pt --output results/revision_training
python -m stg_ddqn.experiments --config configs/revision_validation.yaml \
  --checkpoint checkpoints/revision_stg_ddqn.pt --output results/revision_validation --workers 1
python -m stg_ddqn.validate_results \
  --raw results/revision_validation/evaluation_raw.csv \
  --summary results/revision_validation/evaluation_summary.csv --strict-curves
```

This configuration uses synthetic mobility, ten training episodes, one trained policy, three paired evaluation seeds, six traffic rates, five speed values, and four topology sizes. It is a diagnostic rerun, not the paper's ten-seed SUMO experiment or a multiple-training-seed study. Evaluation seed variation does not measure training variation. Older `results/behavior` and `checkpoints/behavior_stg_ddqn.pt` are legacy inputs and were not rewritten.

## Render the revised CSV with MATLAB

From the repository root in MATLAB R2020a or later (functions use `exportgraphics`):

```matlab
addpath('matlab');
label = 'Synthetic validation; 3 seeds; one trained policy; 500-route fallback budget';
plot_paper_results('results/revision_validation/evaluation_summary.csv', ...
    'plots/revision_matlab', label, true);
plot_revision_diagnostics('results/revision_validation/evaluation_summary.csv', ...
    'plots/revision_matlab', label);
```

Alternatively, run `matlab -batch "run('matlab/render_revision.m')"` from the repository root. Rendering requires an installed/licensed MATLAB runtime. Python preview images, if supplied, are explicitly labeled and must not be described as MATLAB exports.

## Retrained sensitivity and ablation studies

```bash
python -m stg_ddqn.revision_study --config configs/revision_validation.yaml \
  --study reward --output results/revision_reward
python -m stg_ddqn.revision_study --config configs/revision_validation.yaml \
  --study bursty --output results/revision_bursty
python -m stg_ddqn.revision_study --config configs/revision_validation.yaml \
  --study ablation --output results/revision_ablation
```

The study runner trains each variant independently at three training seeds and evaluates each trained model using the same configured evaluation seeds. Summary confidence intervals use independent training-replicate means, rather than treating all nested evaluation runs as independent samples. Each variant retains its own checkpoint, configuration, logs and request audit. These are fixed-load/fixed-speed studies; the bursty comparison changes both training and evaluation workload and does not establish cross-workload generalization. Run `plot_revision_study` on each completed study summary in MATLAB. Do not claim these studies have completed merely because the runner is supplied.

## Final SUMO study

Supply valid per-speed SUMO CSVs listed in `configs/paper.yaml`, a sufficiently long trace and correctly aligned RSU coordinates/topology. Inspect actual trace speed distributions and coverage before training. Train a new paper-profile model, run the complete paired-seed sweeps, and render those CSVs using the same MATLAB functions with an accurate SUMO evidence label. Do not relabel synthetic diagnostics as SUMO results. Do not infer the cause of a continuity rebound from the curve alone.
