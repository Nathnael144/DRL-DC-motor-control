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
maxEpisodesStage1 = 300;
maxEpisodesStage2 = 900;
showPlots = true;     % true for interactive monitor, false for batch/headless
% --------------------------------------

fprintf('=== Curriculum RL training + evaluation pipeline ===\n');

fprintf('1) Setup environment and tuned reward profile...\n');
setup_rl_environment('dc_motor_rl');
configure_rl_training_profile('dc_motor_rl');

fprintf('2) Curriculum training (stage1=%d, stage2=%d)...\n', ...
    maxEpisodesStage1, maxEpisodesStage2);
[agent, stats1, stats2] = train_rl_agent_curriculum( ...
    maxEpisodesStage1, maxEpisodesStage2, showPlots); %#ok<NASGU>

fprintf('3) Generate RL scenario files (curriculum agent)...\n');
generate_rl_scenario_data(fullfile(thisDir,'trained_rl_agent_curriculum.mat'));

fprintf('4) Compare RL vs LQR vs Pole Placement...\n');
run('compare_controllers.m');
run('create_graph_table_comparison.m');

fprintf('5) Robustness script (classical baseline with saturation)...\n');
run('phase6_robustness_tests.m');

fprintf('Pipeline complete.\n');
