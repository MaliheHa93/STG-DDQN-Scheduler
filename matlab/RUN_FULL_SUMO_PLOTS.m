% Run after the full Python/SUMO study has completed. Separate, untitled plots.
repository = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repository,'matlab'));
results = fullfile(repository,'results','sumo_paper');
marker = fullfile(results,'FULL_MAIN_COMPLETE.json');
assert(isfile(marker), ...
    'The full main sweep is not complete. Check results/sumo_paper/full_study_status.json.');
render_sumo_results(results,fullfile(repository,'plots','sumo_full_matlab'));
