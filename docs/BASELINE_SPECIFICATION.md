# Comparison-method specification

The manuscript compares STG-DDQN with Iterative Migration (IM) [3] and
Dynamic Latency-Aware Partial VNF Migration (DLAPM) [5]. Their published
systems do not use exactly the same streaming-SFC, fog topology, resource, and
make-before-break model as this paper. The package therefore implements
transparent common-model adaptations. Every scheme receives the same requests,
mobility trace, initial resource realization, candidate-generation rules, hard
feasibility checks, and migration mechanism. Their later resource states and
candidate sets can diverge because their action histories differ. STG-DDQN also
learns initial placement; IM and DLAPM share the greedy initialization below.

## Shared initial placement

IM and DLAPM are migration methods. For a new request, both use the same
feasible initial-placement procedure stated in Section VI-B of the manuscript:

1. Generate complete ordered-SFC candidates.
2. Reject candidates that violate CPU, bandwidth, loop-free routing, or
   end-to-end latency constraints.
3. Select the feasible candidate with the lowest service delay; break ties by
   larger residual CPU, larger residual bandwidth, and stable candidate order.

This controls initial placement between IM and DLAPM. The full comparison with
STG-DDQN includes differences in initial placement as well as later migration.

## IM [3] adaptation

At an active-request decision event, IM first checks the current feasible
RETAIN or route-update plan. It retains the current VNF hosts while the current
delay is no more than

```text
im_latency_trigger_ratio * request_latency_requirement
```

When the trigger is exceeded, IM examines genuine host-changing plans. It
selects the plan that changes the fewest VNFs, then the one with the smallest
migration volume and service delay. This preserves the published method's
iterative principle: use the smallest partial relocation that restores service
quality instead of remapping the whole chain.

## DLAPM [5] adaptation

DLAPM evaluates feasible partial-remapping candidates in this order:

1. fewest changed VNFs;
2. lowest post-action service delay;
3. lowest migration volume.

Let `D_before` and `D_after` denote the current and candidate service delays.
The normalized improvement is

```text
(D_before - D_after) / request_latency_requirement.
```

DLAPM retains the current hosts when this value is below
`dlapm_improvement_threshold`; otherwise, it applies the selected partial
remapping. Candidate masking already removes overloaded destinations and
latency-infeasible plans. Thus, the rule remains latency-aware while sharing
the same hard resource constraints as the proposed method.

## Default parameters

| Parameter | Paper profile | Behavior-validation profile |
|---|---:|---:|
| `im_latency_trigger_ratio` | 0.90 | 0.80 |
| `dlapm_improvement_threshold` | 0.05 | 0.02 |

The behavior profile is a stress test, not final evidence. Final manuscript
results must use the paper profile, ten paired seeds, and the frozen SUMO
trace.

## Evidence boundary

The code reproduces the published decision principles inside one controlled
simulation model. It is not a bit-for-bit reproduction of external simulator
implementations that were not supplied with this project. The paper should use
the wording "common-model adaptations of IM [3] and DLAPM [5]" and publish
this specification with the source code.

Primary references:

- IM: https://doi.org/10.1016/j.comnet.2024.110571
- DLAPM: https://doi.org/10.1016/j.comnet.2024.110205
