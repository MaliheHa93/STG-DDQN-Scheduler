% Open this script in MATLAB and click Run. No top titles or panel letters.
% 'validation' uses the bundled short SUMO integration measurements.
% Change to 'paper' after completing the full Python/SUMO paper experiment.
profile = 'validation';
repository = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repository,'matlab'));
render_sumo_results( ...
    fullfile(repository,'results',['sumo_' profile]), ...
    fullfile(repository,'plots',['sumo_' profile '_matlab']));
