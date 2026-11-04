# Final results protocol

1. Export and freeze the SUMO trace at 1 s resolution.
2. Export the 24-RSU/43-link topology realization or provide custom topology CSVs.
3. Run the regression suite and retain its log.
4. Train STG-DDQN for the configured 15 episodes; inspect reward/loss stability and save the checkpoint.
5. Evaluate STG-DDQN, IM, and DLAPM with the same ten scenario seeds.
6. Run traffic, mobility, and scalability sweeps without changing any algorithm-specific capacity, request, or mobility input.
   Run the publishable scalability/runtime sweep sequentially on an otherwise
   idle host; parallel workers change wall-clock decision times.
7. Keep raw request records and wide per-run CSVs; draw plots only from the long summary CSV.
8. Run `validate_results --strict-curves` before plotting. It must confirm all
   schemes, at least four operating points, paired multi-seed samples,
   nondegenerate behavior, and nonzero migration activity.
9. Report mean and 95% confidence interval. Do not turn smoke or synthetic
   behavior-validation output into final paper evidence.
10. Add a spatial-only Graph-DDQN ablation if page space permits; it directly tests the contribution of the temporal encoder.
11. Replace every `Fig. ??` and result placeholder only after checking the corresponding raw CSV rows.

Recommended commands:

```bash
python -m stg_ddqn.selftest
python -m stg_ddqn.train --config configs/paper.yaml
python -m stg_ddqn.experiments --config configs/paper.yaml --output results/paper \
  --resume --workers 1
python -m stg_ddqn.validate_results \
  --raw results/paper/evaluation_raw.csv \
  --summary results/paper/evaluation_summary.csv \
  --strict-curves
python -m stg_ddqn.plot_results --input results/paper/evaluation_summary.csv \
  --output plots/paper --separate
```
