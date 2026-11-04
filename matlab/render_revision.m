% Run from any working directory. Requires MATLAB R2020a or newer.
repository = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repository, 'matlab'));
summary = fullfile(repository, 'results', 'revision_validation', 'evaluation_summary.csv');
output = fullfile(repository, 'plots', 'revision_matlab');
label = 'Synthetic validation; 3 evaluation seeds; one trained policy; 500-route fallback budget';
assert(isfile(summary), 'Run the revised Python experiment first.');
plot_paper_results(summary, output, label, true);
plot_revision_diagnostics(summary, output, label);
