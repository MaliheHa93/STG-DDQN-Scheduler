# Full-duration SUMO workflow

The current full-run workflow is documented in [FULL_DURATION_WORKFLOW.md](docs/FULL_DURATION_WORKFLOW.md).
It retains the 600/3600-second protocol, full model, 15 training episodes and ten
evaluation seeds, with resumable training and an automatic counts/exposure/speed
audit before figures are published. Progress and completion are separate states.

```bash
bash scripts/run_full_sumo.sh --training-device auto
```

Check `results/sumo_paper/full_study_status.json` and
`results/sumo_paper_training/progress.json`. Open `matlab/RUN_FULL_SUMO_PLOTS.m`
after `FULL_MAIN_COMPLETE.json` is present. Final archives are generated under
`../output/SUMO_FULL_DURATION/` after all extended studies finish. Included local
figures are Python renders; the MATLAB code is supplied for your installation.

# Earlier SUMO implementation entry points

Use [the complete SUMO guide](docs/SUMO_IMPLEMENTATION.md) for the new runnable
road scenario, paper-sized configuration, extended experiments and separate
MATLAB figures. The earlier synthetic runs below are retained as historical
diagnostics; they are not SUMO results. The original abstract terminology is
**Vehicular Fog Computing (VFC)**.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-sumo.txt
bash scripts/run_sumo.sh paper main
bash scripts/run_sumo.sh paper extended
```

In MATLAB, from the extracted repository folder:

```matlab
addpath('matlab');
render_sumo_results('results/sumo_paper', 'plots/sumo_paper_matlab');
```

For a short end-to-end check, replace `paper` with `validation` in both shell
commands and MATLAB paths. This reduced model/horizon is not paper-scale evidence.
The MATLAB renderer exports separate PNG/PDF files without top titles or panel letters.

# Previous CCGrid revision

See [revision changes and commands](docs/CCGRID_REVISION.md) for the corrected simulator, fresh validation workflow, evidence limits, and MATLAB rendering. Existing behavior results below are legacy synthetic diagnostics.

# STG-DDQN: Mobility-Aware SFC Continuity

This package implements the paper **Mobility-Aware Service Function Chain Continuity in Vehicular Fog-Edge Networks**. It is a clean replacement for the earlier security-aware SM-SFC simulator; the two models and their results must not be mixed.

## Implemented paper mechanisms

- heterogeneous 24-RSU/43-link directed fog model;
- SUMO CSV mobility, closest-covering RSU association, handoffs, and dwell-time state;
- Poisson streaming-SFC arrivals, 30–120 s lifetimes, and simultaneous CPU/bandwidth reservations;
- complete-chain placement vectors with ordered loop-free physical routes;
- RETAIN, route update, and selective VNF-remapping actions;
- hard CPU, bandwidth, routing, latency, and migration-feasibility masks;
- `Hc=4`, `Kp=3`, `Mmax=32` candidate generation with budgeted fallback (500 prefixes);
- sequential image/state transfer in MB over MB/s links;
- make-before-break destination reservation, cutover, and source release;
- graph attention over physical neighbors and five-snapshot temporal self-attention;
- candidate-conditioned Double DQN with replay, Huber loss, Adam, clipping, and soft target updates;
- IM-style and DLAPM-style behavior-level baselines plus debugging/ablation policies;
- paired seeds, 95% confidence intervals, request-level audit CSVs, and scalability timing;
- Python and MATLAB curve-only figures with line/marker/legend styling and no shaded bands.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

PyTorch is required for STG-DDQN training. The core simulator and heuristic regression tests can run without it.

## Validate the package

```bash
python -m stg_ddqn.selftest
bash scripts/run_smoke.sh
```

The smoke configuration deliberately uses a small synthetic scenario, one seed,
two operating points, and shorter lifetimes. It validates execution and CSV
accounting only. The paper plotting command intentionally rejects smoke output.

For a measured, nontrivial synthetic behavior check, run:

```bash
bash scripts/run_behavior_validation.sh
```

This profile trains the real compact STG-DDQN path and evaluates six traffic
loads, five mobility speeds, four topology sizes, three paired seeds, and the
STG-DDQN/IM/DLAPM schemes. Its curves are diagnostic because mobility remains
synthetic; final manuscript claims still require `paper.yaml` and SUMO. The
script parallelizes only traffic and mobility; it measures scalability with one
worker and merges the partitions afterward so runtime is not CPU-contention
data. The measured diagnostic snapshot and its evidence boundary are recorded
in `docs/BEHAVIOR_VALIDATION.md`.

Decision runtime includes candidate generation for both feasible selections
and rejected attempts. The behavior script fixes BLAS/OpenMP to one thread for
stable small-network inference timing.

## Prepare SUMO mobility

```bash
python tools/export_sumo_trace.py \
  --sumocfg /path/to/scenario.sumocfg \
  --output data/mobility/sumo_trace.csv \
  --epoch 1
```

The final `paper.yaml` run stops if this trace is missing. It never silently substitutes synthetic mobility for a claimed SUMO experiment.

## Train and evaluate

```bash
python -m stg_ddqn.train --config configs/paper.yaml
python -m stg_ddqn.experiments \
  --config configs/paper.yaml \
  --output results/paper \
  --resume \
  --workers 1
python -m stg_ddqn.validate_results \
  --raw results/paper/evaluation_raw.csv \
  --summary results/paper/evaluation_summary.csv \
  --strict-curves
```

The paper configuration uses 600 s warm-up, 3600 s measurement, ten paired seeds, the traffic rates and speed values from Table I, and all three paper schemes.

Use one worker for the publishable scalability/runtime sweep and keep the host
otherwise idle. Parallel workers are available for non-runtime sweeps, but
concurrent simulations would contaminate decision-time measurements.

## Create figures

Python:

```bash
python -m stg_ddqn.plot_results \
  --input results/paper/evaluation_summary.csv \
  --output plots/paper \
  --separate
```

MATLAB:

```matlab
cd matlab
plot_paper_results('../results/paper/evaluation_summary.csv', '../plots/paper_matlab')
```

Both paths export a vector PDF and 600-dpi PNG. The four compact panels are acceptance vs. arrival rate, continuity vs. speed, migration volume vs. speed, and 95th-percentile decision runtime vs. RSU count. The numerical CSV keeps the 95% confidence intervals, while the figures intentionally show only the three mean curves requested for STG-DDQN, IM, and DLAPM.

The packaged diagnostic exports are in `plots/curve_only_behavior/`. They were
generated from `results/behavior/evaluation_summary.csv`; they confirm measured
curve behavior but must not be described as the final ten-seed SUMO results.

The behavior-validation script names its combined export
`stg_ddqn_behavior_validation_panels.*`; the default paper command retains
`stg_ddqn_paper_panels.*`. This naming boundary prevents synthetic diagnostics
from being confused with the required ten-seed SUMO result.

The Python and MATLAB plotting paths reject missing algorithms, fewer than four
operating points, one-seed grids, incomplete comparisons, invalid confidence
intervals, fully flat/overlapping panels, and zero-migration data. They never
smooth or fabricate results. Python exports are published atomically so a
truncated PNG cannot appear under the final figure name.

## Result files

- `evaluation_raw.csv`: one wide row per algorithm, scenario, and seed;
- `evaluation_summary.csv`: long means and 95% confidence intervals;
- `request_records.csv`: admission, continuity outcome, migration, handoff, and delay audit trail;
- `training_log.csv`: episode-level training diagnostics;
- `stg_ddqn_paper_panels.pdf/png`: compact paper figure.

## Scientific-use notes

The 80–250 ms latency range is retained because processing rates are 30–60 MI/ms; migration delay is separate. Bandwidth is consistently MB/s, so Eq. (19) multiplies MB/MB/s by 1000 to obtain milliseconds. Active requests at the measurement boundary are excluded from the continuity denominator exactly as stated in the paper. See `docs/PAPER_CODE_ALIGNMENT.md` and `docs/RESULTS_PROTOCOL.md` before writing numerical claims.

Instantaneous infrastructure actions are applied immediately, but their RL
transition ends at the next request-specific decision or terminal event. This
avoids a duplicate same-time RETAIN self-loop and attributes later handoff
failure to the preceding action. Synchronize the manuscript using the exact
replacement text in `docs/PAPER_TEXT_UPDATE.md`.

The exact common setup and decision rules used for the two comparison methods
are stated in `docs/BASELINE_SPECIFICATION.md`. The manuscript should describe
them as common-model adaptations of IM [3] and DLAPM [5], rather than claiming
that unavailable external simulator code was reproduced exactly.
