clear; clc; close all;

thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir)
    thisDir = pwd;
end
oldDir = pwd;
cleanupObj = onCleanup(@() cd(oldDir)); %#ok<NASGU>
cd(thisDir);
addpath(thisDir);

fprintf('=== Post-training evaluation pipeline ===\n');
fprintf('1) Export RL scenario data...\n');
T = generate_rl_scenario_data(); %#ok<NASGU>

fprintf('2) Run RL vs LQR vs Pole Placement comparison...\n');
run('compare_controllers.m');

fprintf('3) Run robustness tests (classical baseline script)...\n');
run('phase6_robustness_tests.m');

fprintf('Pipeline complete.\n');
fprintf('Generated files include:\n');
fprintf('  - rl_step_nominal.mat\n');
fprintf('  - rl_step_load_disturbance.mat\n');
fprintf('  - rl_ramp.mat\n');
fprintf('  - rl_sine.mat\n');
fprintf('  - phase5_comparison_results.mat\n');
fprintf('  - rl_scenario_metrics.mat\n');
