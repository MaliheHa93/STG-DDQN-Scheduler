function render_sumo_results(results_dir, output_dir)
%RENDER_SUMO_RESULTS Separate PDF/PNG figures from Python/SUMO result CSVs.
% MATLAB R2020a or newer. No toolboxes required. No panel letters or titles.
% From repository root:
% addpath('matlab'); render_sumo_results('results/sumo_paper','plots/sumo_matlab')
% For the bundled short integration run, use results/sumo_validation instead.
% CSVs contain Student-t 95% intervals computed in Python. This function
% displays those measurements without fitting or changing their values.
repository = fileparts(fileparts(mfilename('fullpath')));
if nargin < 1, results_dir = fullfile(repository,'results','sumo_paper'); end
if nargin < 2, output_dir = fullfile(repository,'plots','sumo_matlab'); end
assert(isfile(fullfile(results_dir,'evaluation_summary.csv')), ...
    'Run the SUMO training/evaluation pipeline first.');
if ~isfolder(output_dir), mkdir(output_dir); end
raw = readtable(fullfile(results_dir,'evaluation_raw.csv'),'TextType','string');
assert(all(raw.mobility_mode == "sumo_csv"), ...
    'This entry point requires real SUMO-based results.');
assert(all(strlength(raw.sumo_trace_sha256) == 64), ...
    'Missing SUMO trace fingerprints.');
T = readtable(fullfile(results_dir,'evaluation_summary.csv'),'TextType','string');
panels = {
 'traffic','acceptance_ratio_pct','Arrival rate \lambda (requests/s)', ...
     'Request acceptance ratio (%)','request_acceptance';
 'mobility','continuity_satisfaction_ratio_pct','Maximum vehicle speed (m/s)', ...
     'SFC continuity satisfaction (%)','sfc_continuity';
 'mobility','migration_mb_per_admitted_sfc','Maximum vehicle speed (m/s)', ...
     'Transferred payload per admitted SFC (MB)','migration_volume';
 'scalability','p95_decision_runtime_ms','Number of fog RSUs', ...
     '95th-percentile decision runtime (ms)','decision_runtime';
 'mobility','failures_per_service_hour','Maximum vehicle speed (m/s)', ...
     'Failures per service-hour','mobility_failure_rate';
 'mobility','vehicle_departure_requests','Maximum vehicle speed (m/s)', ...
     'SFCs resolved by vehicle departure','vehicle_departures';
 'mobility','coverage_gap_failures','Maximum vehicle speed (m/s)', ...
     'Coverage-gap failures','coverage_failures';
 'mobility','active_excluded_requests','Maximum vehicle speed (m/s)', ...
     'SFCs censored at measurement end','censored_sfcs';
 'traffic','peak_temporary_cpu','Arrival rate \lambda (requests/s)', ...
     'Peak temporary CPU reservation (model units)','temporary_cpu_peak';
 'traffic','temporary_cpu_seconds','Arrival rate \lambda (requests/s)', ...
     'Temporary CPU reservation (model unit s)','temporary_cpu_integral';
 'traffic','blocked_recoverable_without_temporary_reservations', ...
     'Arrival rate \lambda (requests/s)', ...
     'Admissions recovered in snapshot audit','reservation_audit';
 'traffic','search_budget_blocked_requests','Arrival rate \lambda (requests/s)', ...
     'Requests blocked after search-budget exhaustion','search_budget_blocks';
 'scalability','mean_candidate_runtime_ms','Number of fog RSUs', ...
     'Mean candidate construction runtime (ms)','candidate_runtime';
 'scalability','mean_policy_runtime_ms','Number of fog RSUs', ...
     'Mean policy selection runtime (ms)','policy_runtime';
 'scalability','mean_reference_route_runtime_ms','Number of fog RSUs', ...
     'Mean reference-route runtime (ms)','reference_route_runtime';
 'scalability','assignment_nodes_examined','Number of fog RSUs', ...
     'Partial assignment trials during measurement','assignment_search_work'
};
algorithms = ["stg_ddqn","im","dlapm"];
for p = 1:size(panels,1)
    R = T(T.experiment == string(panels{p,1}) & T.metric == string(panels{p,2}),:);
    if isempty(R)
        warning('SUMO:MissingMetric','Skipping unavailable metric %s',panels{p,2});
        continue
    end
    fig = figure('Visible','off','Color','w','Position',[100 100 900 620]);
    cleaner = onCleanup(@() close(fig));
    ax = axes(fig); hold(ax,'on');
    colors = [0 .447 .698; .337 .706 .914; .902 .624 0];
    markers = {'o','^','d'}; styles = {'-','--','--'};
    labels = {'STG-DDQN','IM','DLAPM'};
    reference = [];
    for a = 1:numel(algorithms)
        S = sortrows(R(R.algorithm == algorithms(a),:),'x_value');
        assert(~isempty(S),'Missing algorithm %s',algorithms(a));
        assert(numel(unique(S.x_value)) == height(S),'Duplicate x-values.');
        assert(all(isfinite(S.mean)),'Nonfinite measurements; extend the experiment.');
        if isempty(reference), reference = S.x_value;
        else, assert(isequal(reference,S.x_value),'Algorithms use different scenario grids.'); end
        plot(ax,S.x_value,S.mean,'Color',colors(a,:),'Marker',markers{a}, ...
            'LineStyle',styles{a},'MarkerFaceColor',colors(a,:), ...
            'LineWidth',2,'MarkerSize',7,'DisplayName',labels{a});
        valid = S.n >= 2 & isfinite(S.ci95_low) & isfinite(S.ci95_high);
        assert(all(S.ci95_low(valid) <= S.mean(valid) & S.mean(valid) <= S.ci95_high(valid)), ...
            'Invalid confidence interval.');
        errorbar(ax,S.x_value(valid),S.mean(valid), ...
            S.mean(valid)-S.ci95_low(valid),S.ci95_high(valid)-S.mean(valid), ...
            'LineStyle','none','Color',colors(a,:),'HandleVisibility','off','CapSize',5);
    end
    xlabel(ax,panels{p,3}); ylabel(ax,panels{p,4});
    xticks(ax,unique(R.x_value)); grid(ax,'on'); box(ax,'on');
    set(ax,'FontName','Arial','FontSize',12,'LineWidth',1.1);
    legend(ax,'show','Location','best','Box','on');
    exportgraphics(fig,fullfile(output_dir,[panels{p,5} '.pdf']),'ContentType','vector');
    exportgraphics(fig,fullfile(output_dir,[panels{p,5} '.png']),'Resolution',600);
    clear cleaner
end
% Every extension gets its own figures. CIs here are across independently
% trained policies, after averaging their paired evaluation realizations.
studies = {'reward','bursty','ablation'};
for s = 1:numel(studies)
    path = fullfile(results_dir,studies{s},'evaluation_summary.csv');
    if isfile(path)
        render_study(path,fullfile(output_dir,studies{s}));
    end
end
fid = fopen(fullfile(output_dir,'PROVENANCE.txt'),'w');
assert(fid >= 0,'Cannot write figure provenance.');
fprintf(fid,'Source: %s\nRenderer: MATLAB %s\n',results_dir,version);
fprintf(fid,'No titles or panel letters. Original CSV measurements and intervals retained.\n');
fprintf(fid,'Main sweep intervals: evaluation realizations of one trained policy.\n');
fprintf(fid,'Study intervals: independently trained policy replicate means.\n');
fprintf(fid,'Use protocol.json and trace sidecars to determine duration/model size and SUMO scenario.\n');
fclose(fid);
end


function render_study(summary_csv, output_dir)
if ~isfolder(output_dir), mkdir(output_dir); end
T = readtable(summary_csv,'TextType','string');
metrics = ["acceptance_ratio_pct","continuity_satisfaction_ratio_pct", ...
    "migration_mb_per_admitted_sfc","failures_per_service_hour"];
labels = {'Request acceptance (%)','SFC continuity satisfaction (%)', ...
    'Transferred payload per admitted SFC (MB)','Failures per service-hour'};
for m = 1:numel(metrics)
    R = sortrows(T(T.metric == metrics(m),:),'x_value');
    if isempty(R), continue; end
    assert(all(R.n >= 2),'Study CIs require at least two independent training replicates.');
    fig = figure('Visible','off','Color','w','Position',[100 100 900 620]);
    cleaner = onCleanup(@() close(fig));
    ax = axes(fig);
    errorbar(ax,R.x_value,R.mean,R.mean-R.ci95_low,R.ci95_high-R.mean, ...
        'o','LineWidth',1.8,'MarkerSize',8,'Color',[0 .447 .698], ...
        'MarkerFaceColor',[0 .447 .698]);
    xticks(ax,R.x_value); xticklabels(ax,replace(R.algorithm,'_',' '));
    xtickangle(ax,20); ylabel(ax,labels{m});
    grid(ax,'on'); box(ax,'on');
    set(ax,'FontName','Arial','FontSize',12);
    exportgraphics(fig,fullfile(output_dir,char(metrics(m)+".pdf")),'ContentType','vector');
    exportgraphics(fig,fullfile(output_dir,char(metrics(m)+".png")),'Resolution',600);
    clear cleaner
end
end
