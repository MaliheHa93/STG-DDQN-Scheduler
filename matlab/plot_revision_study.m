function plot_revision_study(summary_csv, output_dir, evidence_label)
%PLOT_REVISION_STUDY Compare independently retrained study variants with CIs.
if nargin < 3, evidence_label = 'Evidence source unspecified'; end
if ~isfolder(output_dir), mkdir(output_dir); end
T = readtable(summary_csv, 'TextType', 'string');
metrics = ["acceptance_ratio_pct", "continuity_satisfaction_ratio_pct", ...
           "migration_mb_per_admitted_sfc", "failures_per_service_hour"];
labels = ["Acceptance (%)", "Continuity satisfaction (%)", ...
          "Transferred MB per admitted SFC", "Failures per service-hour"];
fig = figure('Color', 'w', 'Position', [100 100 1300 850]);
layout = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact');
for p = 1:numel(metrics)
    ax = nexttile(layout); R = sortrows(T(T.metric == metrics(p), :), 'x_value');
    assert(~isempty(R) && all(R.n >= 2), 'Need at least two independent training replicates.');
    assert(all(isfinite(R.mean)), 'Nonfinite study measurements.');
    errorbar(ax, R.x_value, R.mean, R.mean-R.ci95_low, R.ci95_high-R.mean, ...
        'o', 'LineWidth', 1.5, 'MarkerFaceColor', [0 .447 .698]);
    xticks(ax, R.x_value); xticklabels(ax, replace(R.algorithm, '_', ' '));
    xtickangle(ax, 20); ylabel(ax, labels(p)); grid(ax, 'on'); box(ax, 'on');
end
title(layout, evidence_label, 'Interpreter', 'none');
exportgraphics(fig, fullfile(output_dir, 'study.pdf'), 'ContentType', 'vector');
exportgraphics(fig, fullfile(output_dir, 'study.png'), 'Resolution', 300);
close(fig);
end
