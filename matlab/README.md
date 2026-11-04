# MATLAB rendering for the revised simulator

For the current SUMO implementation use this entry point from the repository root:

```matlab
addpath('matlab');
render_sumo_results('results/sumo_paper','plots/sumo_paper_matlab');
% Bundled short integration results:
render_sumo_results('results/sumo_validation','plots/sumo_validation_matlab');
```

`render_sumo_results.m` checks that the raw runs used SUMO, exports each main and
diagnostic plot separately without titles or panel letters, and discovers the
reward/bursty/ablation study CSVs if present. It never substitutes legacy synthetic
results when SUMO data are missing. Requires MATLAB R2020a or newer, no toolboxes.
The MATLAB code is provided for execution on your installation; it was not run here.

The functions below are older rendering entry points and retain their old labeling behavior.

Run `render_revision` in MATLAB, or use `matlab -batch "run('matlab/render_revision.m')"` from the repository root after completing the Python experiments.

`plot_paper_results(summary_csv, output_dir, evidence_label, show_ci)` exports combined and separate vector PDFs and 600-dpi PNGs. The fourth argument defaults to true and displays 95% confidence intervals as error bars. Set false for mean curves only.

`plot_revision_diagnostics` adds exposure, migration reservation and runtime plots. `plot_revision_study` plots independently retrained reward, arrival-model or encoder variants; at least two training replicates are required.

The scripts read measured CSVs without modifying data, smoothing curves or requiring a preferred result shape. Synthetic data must retain their synthetic evidence label. See `docs/CCGRID_REVISION.md` for input requirements and limitations.
