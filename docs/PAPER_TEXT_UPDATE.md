# Required manuscript synchronization

The current SUMO workflow and additional required edits are documented in
`SUMO_IMPLEMENTATION.md` and `SUMO_PAPER_UPDATES.md`. In particular, preserve
Vehicular Fog Computing in the abstract, disclose the new 4096-assignment-trial
budget, and use the revised decision timer including reference routing. The
historical synthetic results discussed below are not the new SUMO validation.

Replace the transition-accounting sentences in Section V-E with the following
compact wording:

> Each request-specific control event constitutes one decision step. Initial
> placement, RETAIN, and route-only actions are applied immediately, while the
> corresponding transition is completed at the next decision event for that
> request or at a terminal outcome. Consequently, the next state represents
> actual mobility and resource evolution rather than a duplicate same-time
> state. A remapping action remains pending until make-before-break preparation
> completes, is canceled, or ends in continuity failure. The resulting
> transition is then stored in replay.

This correction does not change Eqs. (23)--(25). It clarifies the definition of
`S_(l+1)` and makes the learning description match the simulator.

Replace the first paragraph of Section VI-B with wording that does not imply
access to unavailable external simulator code:

> STG-DDQN is compared with common-model adaptations of Iterative Migration
> (IM) [3] and Dynamic Latency-Aware Partial VNF Migration (DLAPM) [5]. All
> schemes use identical mobility traces, requests, initial states, candidate
> placements, hard feasibility checks, routing, and make-before-break migration
> mechanics. IM and DLAPM share a greedy initial-placement rule because they primarily
> define migration behavior; STG-DDQN learns both initial placement and later
> migration decisions. Thus, the full comparison includes initial-placement
> differences. IM uses a latency trigger and iteratively selects
> the smallest feasible partial relocation. DLAPM selects a latency-improving
> partial relocation only when its normalized improvement exceeds the stated
> threshold. The exact common-model rules and parameter values are published
> with the source code.


## Revised dwell feature (schema-3 implementation)

The current default is constant-velocity ray-circle coverage-exit time, clipped at the configured bound. Replace the radial-extrapolation equations and identify this as a local predictor; it does not predict road turns or nearest-RSU reassociation within overlapping coverage. The legacy radial estimator remains an explicit ablation option. Train a new model when changing this state feature.

## Revised evaluation definitions

Actual transferred migration payload includes successful, canceled and partially completed preparations. Completed-remap volume is reported separately. Coverage loss while the vehicle remains in the trace is a continuity failure; lifetime expiry and disappearance from the trace are separately reported normal departures. Boundary-active requests remain excluded from the original continuity denominator, and exposure-normalized failure reporting supplements it. Do not claim that all admitted requests run for their nominal service lifetime when vehicles depart earlier.

Temporary destination CPU and new-route service reservations are measured separately from transfer bandwidth. The admission-impact audit is a measurement-window snapshot counterfactual on a copied topology, not an early-source-release execution policy. Controller runtime is external wall-clock computation time; it does not advance physical simulation time. The evaluation therefore does not establish a real-time operational latency guarantee.

Each SUMO speed condition must use a separately generated trace, with the selected trace fingerprint retained. The fresh local revision_validation outputs use synthetic mobility, three evaluation seeds and a single trained model. They must not be described as ten-seed SUMO results or as a study of training variability. Reward sensitivity, bursty-workload comparisons and temporal-encoder ablations require their own completed retraining studies before results can be claimed.


## Budgeted fallback — replace the unconditional completeness claim

> If the initial exploration returns no feasible action, fallback exploration considers at most 500 loop-free route prefixes. Only fully screened actions are eligible for selection. Budget exhaustion without an available action is recorded separately from proven infeasibility. Consequently, limiting search work may cause blocking or continuity loss even when an unexplored feasible action exists. An uncapped configuration is available as a reference; the work budget does not establish a hard controller response-time guarantee.
