function plot_paper_results(summary_csv, output_dir, evidence_label, show_ci)
%PLOT_PAPER_RESULTS Draw the four compact STG-DDQN panels from Python CSV.
%   plot_paper_results('../results/paper/evaluation_summary.csv', ...
%                      '../plots/paper_matlab')
%
% The script does not recompute or replace simulation results. It reads the
% paired-seed means written by Python. Optional 95% CI error bars are enabled
% by default. Supply an explicit evidence label for synthetic diagnostics.

if nargin < 1 || strlength(string(summary_csv)) == 0
    summary_csv = fullfile('..', 'results', 'paper', 'evaluation_summary.csv');
end
if nargin < 2 || strlength(string(output_dir)) == 0
    output_dir = fullfile('..', 'plots', 'paper_matlab');
end
if nargin < 3, evidence_label = 'Evidence source unspecified'; end
if nargin < 4, show_ci = true; end
if ~isfolder(output_dir)
    mkdir(output_dir);
end

T = readtable(summary_csv, 'TextType', 'string');
required = ["experiment", "x_value", "algorithm", "metric", ...
            "n", "mean", "ci95_low", "ci95_high"];
missing = setdiff(required, string(T.Properties.VariableNames));
assert(isempty(missing), 'Missing summary columns: %s', strjoin(missing, ', '));
assert(all(T.ci95_low <= T.mean | isnan(T.ci95_low)), 'CI lower bound exceeds mean.');
assert(all(T.mean <= T.ci95_high | isnan(T.ci95_high)), 'CI upper bound is below mean.');

panels = {
    "traffic", "acceptance_ratio_pct", ...
        'Arrival rate \lambda (requests/s)', 'Request acceptance ratio (%)', 'southwest';
    "mobility", "continuity_satisfaction_ratio_pct", ...
        'Maximum vehicle speed (m/s)', 'SFC continuity satisfaction (%)', 'southwest';
    "mobility", "migration_mb_per_admitted_sfc", ...
        'Maximum vehicle speed (m/s)', 'Migration volume per admitted SFC (MB)', 'northwest';
    "scalability", "p95_decision_runtime_ms", ...
        'Number of fog RSUs', '95th-percentile decision runtime (ms)', 'northwest'
};
expectedAlgorithms = ["stg_ddqn", "im", "dlapm"];
validate_curve_data(T, panels, expectedAlgorithms, 4);

fig = figure('Color', 'w', 'Units', 'pixels', 'Position', [100 100 1400 920]);
layout = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for p = 1:size(panels, 1)
    ax = nexttile(layout);
    draw_panel(ax, T, panels(p, :), expectedAlgorithms, show_ci);
end
title(layout, evidence_label, 'Interpreter', 'none');
exportgraphics(fig, fullfile(output_dir, 'stg_ddqn_paper_panels.pdf'), ...
    'ContentType', 'vector');
exportgraphics(fig, fullfile(output_dir, 'stg_ddqn_paper_panels.png'), ...
    'Resolution', 600);

for p = 1:size(panels, 1)
    separate = figure('Color', 'w', 'Units', 'pixels', 'Position', [100 100 850 600]);
    ax = axes(separate);
    draw_panel(ax, T, panels(p, :), expectedAlgorithms, show_ci);
    title(ax, evidence_label, 'Interpreter', 'none', 'FontSize', 10);
    letter = char('a' + p - 1);
    exportgraphics(separate, fullfile(output_dir, ['figure_' letter '.pdf']), ...
        'ContentType', 'vector');
    exportgraphics(separate, fullfile(output_dir, ['figure_' letter '.png']), ...
        'Resolution', 600);
    close(separate);
end
close(fig);
end


function draw_panel(ax, T, panel, expectedAlgorithms, show_ci)
experiment = panel{1};
metric = panel{2};
subset = T(T.experiment == experiment & T.metric == metric, :);
if isempty(subset)
    axis(ax, 'off');
    text(ax, 0.5, 0.5, sprintf('No %s/%s data', experiment, metric), ...
        'HorizontalAlignment', 'center');
    return;
end

hold(ax, 'on');
algorithms = expectedAlgorithms;
handles = gobjects(0);
labels = strings(0);
for a = 1:numel(algorithms)
    algorithm = algorithms(a);
    rows = sortrows(subset(subset.algorithm == algorithm, :), 'x_value');
    [label, color, marker, line_style] = algorithm_style(algorithm);
    x = rows.x_value;
    handles(end + 1) = plot(ax, x, rows.mean, ... %#ok<AGROW>
        'Color', color, 'LineStyle', line_style, 'LineWidth', 2.5, ...
        'Marker', marker, 'MarkerSize', 8, 'MarkerFaceColor', color);
    if show_ci
        errorbar(ax, x, rows.mean, rows.mean - rows.ci95_low, ...
            rows.ci95_high - rows.mean, 'LineStyle', 'none', ...
            'Color', color, 'HandleVisibility', 'off', 'CapSize', 5);
    end
    labels(end + 1) = label; %#ok<AGROW>
end

xlabel(ax, panel{3}, 'FontSize', 12);
ylabel(ax, panel{4}, 'FontSize', 12);
xticks(ax, unique(subset.x_value));
grid(ax, 'on');
box(ax, 'on');
ax.FontName = 'Arial';
ax.FontSize = 11;
ax.LineWidth = 1.25;
ax.GridAlpha = 0.25;
axis(ax, 'tight');
allMeans = subset.mean(isfinite(subset.mean));
if show_ci
    allMeans = [allMeans; subset.ci95_low; subset.ci95_high];
end
if endsWith(metric, "_pct")
    span = max(5, max(allMeans) - min(allMeans));
    ylim(ax, [max(0, min(allMeans) - 0.18 * span), ...
              max(100.5, max(allMeans) + 0.12 * span)]);
else
    upperLimit = max(allMeans);
    if upperLimit <= 0
        upperLimit = 1;
    else
        upperLimit = 1.16 * upperLimit;
    end
    ylim(ax, [0, upperLimit]);
end
legend(ax, handles, labels, 'Location', panel{5}, 'NumColumns', numel(labels), ...
    'Box', 'on', 'Color', [1 1 1], 'EdgeColor', [0.78 0.78 0.78]);
hold(ax, 'off');
end


function validate_curve_data(T, panels, expectedAlgorithms, minimumPoints)
problems = strings(0);
for p = 1:size(panels, 1)
    experiment = panels{p, 1};
    metric = panels{p, 2};
    subset = T(T.experiment == experiment & T.metric == metric, :);
    if isempty(subset)
        problems(end + 1) = "missing " + experiment + "/" + metric; %#ok<AGROW>
        continue;
    end
    xReference = [];
    for a = 1:numel(expectedAlgorithms)
        algorithm = expectedAlgorithms(a);
        rows = sortrows(subset(subset.algorithm == algorithm, :), 'x_value');
        if isempty(rows)
            problems(end + 1) = experiment + "/" + metric + ...
                " lacks " + algorithm; %#ok<AGROW>
            continue;
        end
        xValues = unique(rows.x_value);
        if numel(xValues) < minimumPoints
            problems(end + 1) = experiment + "/" + metric + "/" + ...
                algorithm + " has fewer than " + minimumPoints + " x-values"; %#ok<AGROW>
        end
        if min(rows.n) < 2
            problems(end + 1) = experiment + "/" + metric + "/" + ...
                algorithm + " has fewer than two seeds"; %#ok<AGROW>
        end
        if isempty(xReference)
            xReference = xValues;
        elseif ~isequal(xReference, xValues)
            problems(end + 1) = experiment + "/" + metric + ...
                " has an incomplete algorithm grid"; %#ok<AGROW>
        end
        if numel(xValues) ~= height(rows)
            problems(end + 1) = experiment + "/" + metric + " has duplicate x-values";
        end
    end
    % Flat, overlapping and zero-migration measurements can be valid.
    % Validate structure and numeric values, never require a desired shape.
    if any(~isfinite(subset.mean)) || any(~isfinite(subset.ci95_low)) || any(~isfinite(subset.ci95_high))
        problems(end + 1) = experiment + "/" + metric + " contains nonfinite values";
    end

end
assert(isempty(problems), ...
    'Invalid figure data:\n%s', ...
    strjoin(problems, newline));
end


function [label, color, marker, line_style] = algorithm_style(algorithm)
switch char(algorithm)
    case 'stg_ddqn'
        label = "STG-DDQN"; color = [0.000 0.447 0.698]; marker = 'o'; line_style = '-';
    case 'graph_ddqn'
        label = "Graph-DDQN"; color = [0.835 0.369 0.000]; marker = 's'; line_style = '-';
    case 'im'
        label = "IM"; color = [0.337 0.706 0.914]; marker = '^'; line_style = '--';
    case 'dlapm'
        label = "DLAPM"; color = [0.902 0.624 0.000]; marker = 'd'; line_style = '--';
    case 'delay_greedy'
        label = "Delay-Greedy"; color = [0.000 0.620 0.451]; marker = 'v'; line_style = ':';
    otherwise
        label = replace(algorithm, '_', ' '); color = [0.3 0.3 0.3]; marker = 'o'; line_style = '-';
end
end
