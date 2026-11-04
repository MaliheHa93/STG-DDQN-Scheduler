# Complete Python/SUMO implementation and MATLAB plotting

This is the current workflow. It supersedes the earlier synthetic revision
workflow for experiments described as SUMO based. It preserves Vehicular Fog
Computing (VFC), RSU-hosted VNFs, a directed fog backhaul, and a centralized
controller. Changing the word fog to edge is not required by any implementation.

## Install

Python 3.10+ is required; this delivery was exercised with Python 3.12 and
SUMO 1.28.0 on Apple Silicon macOS. From the extracted repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-sumo.txt
python -m unittest discover -s tests -v
```

The SUMO wheel includes `sumo`, `netgenerate`, `duarouter` and `randomTrips.py`.
The generator locates binaries from the installed package, so a global SUMO_HOME
is not required. A system SUMO installation remains usable through the older
`tools/export_sumo_trace.py` path, but the new scenario workflow uses the pinned
SUMO version. macOS wheels may need system dependencies on some machines; see
https://sumo.dlr.de/docs/Downloads.php .

## Paper-sized experiment

```bash
bash scripts/run_sumo.sh paper main
bash scripts/run_sumo.sh paper extended
```

The first command creates the SUMO scenario and traces, trains a fresh model,
then runs traffic, mobility and scalability sweeps sequentially with paired
evaluation seeds. It validates CSV accounting. The second command retrains and
evaluates every reward, arrival-model and encoder variant. It does not reuse a
single model and merely relabel its output as a sensitivity study.

Individual stages can be run as follows:

```bash
bash scripts/run_sumo.sh paper generate
bash scripts/run_sumo.sh paper train
bash scripts/run_sumo.sh paper evaluate
```

Generate and evaluate resume safely. Evaluation checks the configuration,
checkpoint hash and trace hash before accepting old rows. The `train` stage
starts a fresh training run and replaces its named checkpoint only after all
episodes complete. Do not retrain that checkpoint into an existing evaluation
directory: keep the frozen evaluation model or choose a new result directory.
Training and extended studies are computationally substantial; they run in the
foreground and are not claimed complete merely because their scripts exist.

Full runs can produce several gigabytes of traces. This archive includes the
short integration traces, not a fabricated full paper dataset. The paper-size
protocol must be executed before replacing the manuscript's numerical claims.

## Reproducible SUMO road scenario

`stg_ddqn/sumo_scenario.py` builds a two-lane 5-by-5 road grid over the configured
2,400-by-1,600 m region, with traffic signals inferred by SUMO. Routes are generated
with seeded `randomTrips.py`, two intermediate waypoints, and a 0.3 s departure
period. Passenger cars use the Krauss car-following model, acceleration 2.6 m/s²,
deceleration 4.5 m/s², sigma 0.5, length 5 m and minimum gap 2.5 m. Road speeds and
vehicle maxSpeed are capped at each experimental maximum; speedFactor=1 and
speedDev=0 prevent intentional speeding. Traffic queues and intersection turning
can yield speeds well below the cap. Thus the x-axis is a maximum speed condition,
not a claim that all vehicles move at that speed.

SUMO enforces at most 300 simultaneously active vehicles. Vehicles can enter,
finish a trip, and leave; the observed population is not fixed at 300. The main
paper table already specifies a maximum vehicle count. Report measured population
and speed distributions from each trace sidecar. Teleportation is disabled, and
SUMO warnings/logs are retained. A 120 s road-traffic pre-roll precedes scheduler
time zero in the full profile; this is additional to the 600 s scheduler warmup.

The road network is a new reproducible benchmark, not a recovered copy of the
unavailable original scenario. Explicitly describe it in the revised paper.
RSU coordinates and road coordinates share the configured rectangular region.
At smaller scalability topologies, the same region and workload are retained;
coverage and offered load per RSU can change. This is not constant-load-per-node
scaling or an isolated inference microbenchmark.

SUMO Floating Car Data are sampled every second, then converted to a CSV with
position, speed and heading. SUMO navigation angle is converted to the controller's
mathematical heading: heading = radians(90 - SUMO_angle). The pre-roll offset is
removed. Each CSV has a JSON sidecar with all sampled epochs, duration, SUMO
version, command recipe, SHA-256 hashes and observed traffic statistics. Empty
vehicle frames are valid and distinct from missing simulation epochs. Missing,
shortened or modified traces fail validation. No synthetic fallback occurs.

Training traces use seeds 10001, 20001, 30001, while the full evaluation uses
101, 202, ..., 1010. Every compared algorithm receives the same trace for a given
evaluation seed and speed. One source checkpoint is frozen for all main sweep
runs. Full evaluation comprises 450 runs: (6 traffic + 5 speed + 4 size conditions)
× 3 algorithms × 10 seeds. Its CIs describe evaluation variation conditional on
one trained policy, not variation across independently trained models.

## Paper parameter alignment

`configs/sumo_paper.yaml` inherits `paper.yaml` and `default.yaml`, preserving:

| Item | Full SUMO profile |
|---|---|
| Network | 24 fog RSUs, 43 physical links / 86 directed arcs, 350 m coverage |
| Warmup / measurement / epoch | 600 / 3600 / 1 s |
| Traffic sweep | 0.1, 0.2, 0.4, 0.6, 0.8, 1.0 requests/s; cap 30 m/s |
| Mobility sweep | caps 10, 20, 30, 40, 50 m/s; load 0.4 requests/s |
| Scalability | 12, 24, 48, 96 fog RSUs; physical links 2N−5 |
| SFC | 3–6 VNFs; CPU 5–15; workload 300–900 MI; bandwidth 5–20 MB/s |
| Latency / lifetime | 80–250 ms / 30–120 s |
| Nodes / backhaul | CPU 65–95 units; processing 30–60 MI/ms; 105–135 MB/s links |
| Migration | images 50–200 MB + 5 MB state; at most 20 MB/s; 100 ms nominal cutover |
| Action construction | Hc=4, Kp=3, Mmax=32; fallback budget 500 route prefixes; assignment budget 4096 partial trials |
| State / model | W=5; 3 graph + 3 temporal layers; 8 heads; dimension 128; FF 512; Q 5×256 |
| Learning | 15 episodes, LR 3e−5, gamma .95, replay 1e6, batch 256, warmup 256 |
| Reward | .35 latency, .25 lifetime, .25 payload, .15 nominal downtime |
| IM / DLAPM thresholds | .90 / .05 |

CPU device is explicitly selected for portability. No physical inference hardware
or runtime measurements should be inferred from Table I's abstract parameters.

## Implemented extensions and their limits

1. **Dwell feature:** constant-velocity ray/circle coverage-exit time clipped to
   60 s replaces the radial projection. SUMO supplies turns and speed changes,
   but the predictor uses only current observable velocity; it does not inspect
   future trace frames or become a learned route/turn predictor.
2. **Migration correctness and accounting:** all schemes use fully screened
   candidates, capacity-filtered Dijkstra migration routes, temporary destination
   CPU and new-only service reservations, and actual payload accounting including
   canceled/partial transfers. Successful planned volume is also retained.
3. **MBB costs:** peak and integrated extra reservations are measured. A copied
   topology audit counts admissions recovered without those reservations. This
   audit also changes bounded candidate search and is not a causal full-run
   break-before-make ablation. It does not alter the running allocation state.
4. **Bounded search:** exhaustion is reported separately from proven physical
   infeasibility. A 500-prefix budget provides a work limit for fallback, not a
   strict millisecond bound on all routing or controller operations. The new
   SUMO profile also limits partial VNF-assignment trials to 4096 per decision,
   across preliminary construction and fallback combined. Feasible RETAIN is
   screened first; only fully screened actions are selected. Truncation can omit
   better or feasible plans and is explicitly counted. This additional parameter
   must be added to the revised manuscript rather than presented as an original
   Table I setting. Set `maximum_assignment_nodes: null` for the uncapped reference.
   Ordered retention routing uses A* with admissible relaxed segment distances
   and exact infeasibility proofs, checked against exhaustive small-graph search.
5. **Bursty arrivals:** an MMPP condition uses low/high rate factors .25/2.5 and
   a 10 s high-state mean duration, with stationary mean matched to the Poisson
   rate. Both conditions are retrained and evaluated with matched seeds.
6. **Reward study:** original, delay-oriented, migration-oriented and equal
   weight sets are independently retrained with three training replicate seeds.
7. **Learning comparison:** Graph-DDQN is a spatial-only ablation trained within
   the same code; it is not an implementation of external DRL-GAMO. No claim of
   reproducing DRL-GAMO is justified by this implementation.
8. **Mobility outcome diagnostics:** departure, lifetime expiry, coverage loss,
   measurement-end censoring, canceled transfers and failures per service-hour
   are recorded separately. A high-speed rebound must be explained from these
   outputs if it occurs, rather than assumed in advance.

IM and DLAPM selection code is unchanged from the supplied repository. Both
are common-model adaptations, not externally reproduced reference simulators.
STG-DDQN learns initial placement, while the two heuristics share a greedy rule.
Their subsequent realized resource states can diverge. No baseline change was
made because of the fog/edge terminology. The `.80/.02` synthetic behavior-profile
thresholds are not used by either new SUMO profile.

Wireless access remains an idealized fixed 2 ms, not a packet-level LTE/NR-V2X
channel. RAM/GPU resources, queueing, controller signaling/failover and packet
loss are not implemented. Nominal cutover is a configured cost, not a measured
packet outage; controller wall-clock computation does not advance simulation
time. These are explicit paper limitations, not newly evaluated capabilities.

The current decision timer includes reference-route computation, candidate
construction, state preparation and policy selection, including rejected attempts.
`mean_reference_route_runtime_ms` reports the first component separately. Some
pre-event state bookkeeping, post-selection application, audit work and controller
signaling remain outside this timer. Old synthetic runtime figures excluded the
reference-route computation and are not directly comparable to this definition.

`evaluation_metric_counts.csv` records how many evaluation realizations
contribute to each study replicate/metric. Undefined ratios remain missing;
replicate means use available cases rather than replacing missing measurements
with zeros. Inspect these counts, especially for the short integration profile.

## MATLAB: separate figures with no top labels

Run MATLAB from the extracted repository root:

```matlab
addpath('matlab');
render_sumo_results('results/sumo_paper','plots/sumo_paper_matlab');
```

Or at a terminal on your MATLAB machine:

```bash
matlab -batch "addpath('matlab'); render_sumo_results('results/sumo_paper','plots/sumo_paper_matlab');"
```

The renderer exports separate vector PDF and 600 dpi PNG files for acceptance,
continuity, migration volume, runtime, exposure/termination diagnostics, MBB costs
and search-budget effects. If extended study summaries exist, it also exports
four separate metrics for each study. It preserves axis labels, legends and CIs,
omits top titles and panel letters, and writes provenance to a separate text file.
Student-t intervals are displayed as recorded, without clipping percentages,
smoothing, fabricating values or requiring a preferred curve shape.

## Short integration run included with the delivery

```bash
bash scripts/run_sumo.sh validation main
bash scripts/run_sumo.sh validation extended
```

To recompute after extracting the archive into a different folder, write to a
new result directory rather than resuming the bundled historical run:

```bash
SUMO_OUTPUT_DIR=results/sumo_validation_local bash scripts/run_sumo.sh validation main
SUMO_OUTPUT_DIR=results/sumo_validation_local bash scripts/run_sumo.sh validation extended
```

The bundled CSVs remain directly readable by MATLAB regardless of extraction
path. Strict experiment resumption checks include resolved input paths, so a
relocated historical run is intentionally not merged with new measurements.

This profile uses real SUMO trajectories, a 10 s scheduler warmup, 80 s measurement,
three evaluation seeds and a smaller 32-dimensional one-layer encoder, with three
training episodes. Network/workload/baseline parameters and all x-axis values
remain the same as the full profile. The additional SUMO pre-roll is 60 s.
It checks the workflow and supplies example CSVs, not final scientific claims or
evidence of convergence. Use `results/sumo_validation` in the MATLAB invocation.
