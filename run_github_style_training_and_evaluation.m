clear; clc; close all;

thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir)
    thisDir = pwd;
end
oldDir = pwd;
cleanupObj = onCleanup(@() cd(oldDir)); %#ok<NASGU>
cd(thisDir);
addpath(thisDir);

maxEpisodes = 1000;
showPlots = true;

fprintf('=== GitHub-inspired DDPG pipeline ===\n');
fprintf('1) Setup GitHub-style env/profile...\n');
setup_rl_environment_github_style('dc_motor_rl');

fprintf('2) Train DDPG for %d episodes...\n', maxEpisodes);
[agent, stats] = train_rl_agent_github_style(maxEpisodes, showPlots); %#ok<NASGU>

fprintf('3) Evaluate trained agent...\n');
generate_rl_scenario_data(fullfile(thisDir,'trained_rl_agent_github_style.mat'));

fprintf('4) Compare RL vs LQR vs PP...\n');
run('compare_controllers.m');
run('create_graph_table_comparison.m');

fprintf('5) Robustness (classical baseline)...\n');
run('phase6_robustness_tests.m');

fprintf('GitHub-inspired pipeline complete.\n');
