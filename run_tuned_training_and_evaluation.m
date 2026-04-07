clear; clc; close all;

thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir)
    thisDir = pwd;
end
oldDir = pwd;
cleanupObj = onCleanup(@() cd(oldDir)); %#ok<NASGU>
cd(thisDir);
addpath(thisDir);

% -------- User-adjustable knobs --------
maxEpisodes = 1200;   % increase for better convergence
showPlots = true;     % set false for batch/headless execution
% --------------------------------------

fprintf('=== Tuned RL training + evaluation pipeline ===\n');
fprintf('1) Setup environment and apply tuned profile...\n');
setup_rl_environment('dc_motor_rl');
configure_rl_training_profile('dc_motor_rl');

fprintf('2) Train tuned TD3 agent for %d episodes...\n', maxEpisodes);
[agent, trainingStats] = train_rl_agent_tuned(maxEpisodes, showPlots); %#ok<NASGU>

fprintf('3) Generate RL scenario data from tuned agent...\n');
generate_rl_scenario_data(fullfile(thisDir,'trained_rl_agent_tuned.mat'));

fprintf('4) Compare RL vs LQR vs Pole Placement...\n');
run('compare_controllers.m');
run('create_graph_table_comparison.m');

fprintf('5) Robustness summary (classical baseline script)...\n');
run('phase6_robustness_tests.m');

fprintf('Pipeline complete.\n');
