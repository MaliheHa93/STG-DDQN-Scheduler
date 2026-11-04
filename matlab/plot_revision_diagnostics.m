function plot_revision_diagnostics(summary_csv, output_dir, evidence_label)
%PLOT_REVISION_DIAGNOSTICS Plot measured exposure, reservation and timing data.
% Missing metrics are reported; no result is substituted or interpolated.
if nargin < 3, evidence_label = 'Evidence source unspecified'; end
if ~isfolder(output_dir), mkdir(output_dir); end
T = readtable(summary_csv, 'TextType', 'string');
panels = {
    "mobility", "failures_per_service_hour", 'Maximum vehicle speed (m/s)', 'Failures per service-hour';
    "traffic", "peak_temporary_cpu", 'Arrival rate (requests/s)', 'Peak temporary CPU (model units)';
    "traffic", "temporary_cpu_seconds", 'Arrival rate (requests/s)', 'Temporary CPU reservation (unit s)';
    "mobility", "canceled_transfer_total_mb", 'Maximum vehicle speed (m/s)', 'Canceled transfer volume (MB)';
    "scalability", "mean_candidate_runtime_ms", 'Number of fog RSUs', 'Mean candidate construction time (ms)';
    "scalability", "complete_fallback_count", 'Number of fog RSUs', 'Fallback decisions';
    "traffic", "search_budget_blocked_requests", 'Arrival rate (requests/s)', 'Admissions lost to search budget';
    "mobility", "search_budget_continuity_failures", 'Maximum vehicle speed (m/s)', 'Continuity failures from search budget'
};
fig = figure('Color', 'w', 'Position', [100 100 1400 1250]);
layout = tiledlayout(fig, 4, 2, 'TileSpacing', 'compact');
for p = 1:size(panels, 1)
    ax = nexttile(layout); hold(ax, 'on');
    S = T(T.experiment == panels{p,1} & T.metric == panels{p,2}, :);
    assert(~isempty(S), 'Missing metric: %s', panels{p,2});
    algorithms = unique(S.algorithm, 'stable');
    for a = 1:numel(algorithms)
        R = sortrows(S(S.algorithm == algorithms(a), :), 'x_value');
        assert(all(isfinite(R.mean)), 'Nonfinite metric: %s', panels{p,2});
        errorbar(ax, R.x_value, R.mean, R.mean-R.ci95_low, R.ci95_high-R.mean, ...
            '-o', 'LineWidth', 1.5, 'DisplayName', replace(algorithms(a), '_', '-'));
    end
    xlabel(ax, panels{p,3}); ylabel(ax, panels{p,4});
    grid(ax, 'on'); box(ax, 'on'); legend(ax, 'Location', 'best');
end
title(layout, evidence_label, 'Interpreter', 'none');
exportgraphics(fig, fullfile(output_dir, 'revision_diagnostics.pdf'), 'ContentType', 'vector');
exportgraphics(fig, fullfile(output_dir, 'revision_diagnostics.png'), 'Resolution', 300);
close(fig);
end
