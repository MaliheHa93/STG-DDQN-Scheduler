# Full-duration SUMO study

This workflow executes the actual full protocol. It does not relabel the
previous 80-second validation data. Current progress is written to
`results/sumo_paper/full_study_status.json`; training progress is in
`results/sumo_paper_training/progress.json`. A running status is not completion.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-sumo.txt
bash scripts/run_full_sumo.sh --training-device auto
```

The Apple GPU is selected through MPS when available; CUDA is selected on a
CUDA workstation. The full 128-dimensional architecture, 256-sample batches,
15 training episodes, 600-second scheduler warmup, 3,600-second measurement and
ten evaluation seeds are retained. A separate 120-second SUMO traffic pre-roll
precedes scheduler time zero. All 53 mobility traces are generated from SUMO.

## Compute time and recovery

This is a long research computation. The local Mac benchmark measured roughly
0.5–1 second per full-sized gradient update, not per episode. There are many
updates per episode and multiple independently trained study variants. No
wall-clock completion time or convergence is guaranteed by an episode count.

Training uses the Apple GPU only for acceleration. Since fused MPS attention
does not support nonzero training dropout in this environment, the portable
layer evaluates the same scaled dot-product attention explicitly on MPS.
The original dropout remains enabled; parameter names and architecture are
unchanged. The explicit attention values and gradients are tested against
PyTorch's standard attention. CPU and CUDA retain the standard implementation.
Evaluation of the main sweep uses CPU and one math thread. Training hardware
and evaluation hardware are recorded separately.

Progress is written every 60 simulated seconds. At 300-simulated-second intervals,
the program writes an atomic resume state including environment, replay,
optimizer, policy and RNG states. SUMO frames are reloaded from their verified
trace rather than embedded in this state. The inference checkpoint is published
only when all 15 episodes complete. Resumption checks source/configuration and
checksums. Internal resume pickles are trusted local state, not imported model
files, and are excluded from the delivery archive. Resume on the same device.

```bash
bash scripts/run_full_sumo.sh --training-device mps
```

Repeating that command resumes completed stages and internal training state.
Exclusive file locks prevent two trainers or result writers from racing.
Changing source or scientific settings requires a separate run; do not bypass
the resume checks. To stop the workflow, terminate its process and any active
child jobs; completed rows and the latest training state remain recoverable.

## Execution and timing

The main study contains 450 evaluations: 15 conditions × 3 algorithms × 10
paired seeds. Independently trained reward, bursty-arrival and encoder studies
add 240 evaluations. Identical condition/seed/configuration/source checkpoints
can be reused across study headings; this does not create extra independent
training replicates. Each variant's uncertainty uses its three independent
training-replicate means over ten evaluation seeds.

An initial IM/DLAPM traffic/mobility phase may run alongside training; those
rows are marked `concurrent_full_training`. Their scientific outcomes do not
depend on wall-clock speed under the stated instantaneous-controller model.
Scalability timing is run after training with one evaluation worker, without
another study job. Normal host background activity is not controlled. The
decision timer includes reference routing, candidate generation, observation
construction and policy selection. It is not a packet-level response-time bound.

## Completion audit and figures

The audit requires all 450 unique main keys, paired traces/requests, verified
4,200-epoch inputs, and the completed matching full-training checkpoint. It
reports, per speed condition and scheme:

- services admitted, resolved, failed and censored;
- lifetime expiry, vehicle departure, coverage and search-budget outcomes;
- active service exposure and failures per service-hour;
- actual SUMO speed and population statistics across the ten traces.

The primary continuity curve remains the mean of seed-level ratios. Pooled
counts/rates are supplementary and do not silently change that metric. Curves
are not forced to decline, rebound or favor STG-DDQN. Student-t intervals remain
unbounded; zero observed failures does not prove zero failure probability.

`FULL_MAIN_COMPLETE.json` is written only after the main audit passes. Primary
and diagnostic Python figures can then be produced. After extended studies
finish, `FULL_STUDY_COMPLETE.json` is written and all plots and archives are
generated under `../output/SUMO_FULL_DURATION/`. No final-result archive is
published from an incomplete study.

In MATLAB, open `matlab/RUN_FULL_SUMO_PLOTS.m` and click Run. It checks that the
full main evaluation has passed its audit, then exports separate vector PDFs
and PNGs without top titles or panel letters. MATLAB code is supplied for your
installation; figures rendered by this workflow on the current host are
explicitly identified as Python output. Missing full data never fall back to
the short validation CSVs.

The full code/results archive excludes multi-gigabyte mobility CSVs and internal
resume states. SUMO generation code, route/network inputs, seed recipes and
trace checksums are included so trajectories can be regenerated. The trained
main inference checkpoint is included; extended model binaries can be retained
locally while their configurations, training logs and result CSVs are packaged.

Fog terminology and IM/DLAPM thresholds (.90/.05) remain unchanged. Shared
search limits remain 500 fallback prefixes and 4,096 partial-assignment trials.
Fixed radio delay, CPU/bandwidth-only feasibility, local dwell prediction,
centralized instantaneous simulation control and the generated-map scope remain
limitations. Longer duration does not implement fading, RAM/GPU capacity, an
external DRL-GAMO method, or controller signaling.
