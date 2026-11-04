# Synthetic behavior-validation snapshot

This snapshot exists to verify that the implementation produces measured,
nondegenerate algorithm behavior before the final SUMO study. It uses the
compact `configs/behavior.yaml` profile, a 10-episode STG-DDQN checkpoint, and
three paired seeds (31, 47, and 73). It is not final paper evidence.

## Design

- Algorithms: STG-DDQN, IM, and DLAPM.
- Traffic: six arrival rates from 0.25 to 1.25 requests/s.
- Mobility: five maximum speeds from 10 to 50 m/s.
- Scalability: 8, 12, 16, and 24 fog RSUs.
- Replication: three paired seeds at every algorithm/operating point.
- Uncertainty: means and Student-t 95% confidence intervals.
- Plot style: mean line-and-marker curves only; confidence intervals remain in
  `evaluation_summary.csv` and are not drawn as shaded bands.
- Timing: one simulator worker and one BLAS/OpenMP thread on an otherwise idle
  experiment runtime.
- Total: 135 runs; the raw and request-level records are retained.

## What the measured curves show

- Admission is load-sensitive rather than flat: all schemes start at 100% at
  0.25 requests/s and fall to 43.21--47.65% at 1.25 requests/s.
- Continuity changes with speed and policy. Across the five speeds the means
  span 84.24--97.62%; the wide three-seed intervals warn against claiming a
  universal winner from this diagnostic.
- Migration is nonzero and policy-dependent. Mean volume per admitted SFC is
  22.84--29.04 MB for STG-DDQN, 11.43--23.70 MB for IM, and 54.19--76.70 MB for
  DLAPM.
- Decision time grows nonlinearly with topology size. After including rejected
  decision attempts, mean p95 decision time at 8 RSUs is 11.29 ms for
  STG-DDQN, 14.41 ms for IM, and 13.91 ms for DLAPM; at 24 RSUs it is 60.42,
  49.44, and 41.58 ms, respectively.

These results reveal trade-offs rather than being tuned to force STG-DDQN to
win every point. Numerical manuscript claims must come from the ten-seed SUMO
protocol in `docs/RESULTS_PROTOCOL.md`.
