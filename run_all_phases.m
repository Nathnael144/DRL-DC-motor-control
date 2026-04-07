clear; clc; close all;

thisDir = fileparts(mfilename('fullpath'));
oldDir = pwd;
cleanupObj = onCleanup(@() cd(oldDir)); %#ok<NASGU>

if ~isempty(thisDir)
    addpath(thisDir);
    cd(thisDir);
end

fprintf('===============================================\n');
fprintf('Running DC Motor Project: Phases 1 to 6\n');
fprintf('===============================================\n');

run('phase1_model.m');
run('phase2_classical_demo.m');

% Phase 3 + 4 (requires Simulink model + RL Toolbox)
try
    setup_rl_environment('dc_motor_rl');
    train_rl_agent('td3');
catch ME
    warning('Phase 3-4 skipped: %s', ME.message);
end

% Phase 5 comparison (RL overlay uses optional rl_<scenario>.mat files)
run('compare_controllers.m');

% Phase 6 robustness
run('phase6_robustness_tests.m');

fprintf('All available phases completed.\n');
