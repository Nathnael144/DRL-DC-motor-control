clear; clc;

thisDir = fileparts(mfilename('fullpath'));
if ~isempty(thisDir) && isempty(which('dc_motor_params'))
	addpath(thisDir);
end

modelName = 'dc_motor_rl';

hasRlToolbox = (exist('rlSimulinkEnv','file') == 2) && ...
			   ((exist('rlNumericSpec','class') == 8) || (exist('rlNumericSpec','file') == 2));

fprintf('Building fully wired Simulink model: %s.slx ...\n', modelName);
create_dc_motor_rl_simulink_model(modelName, hasRlToolbox);

if hasRlToolbox
	fprintf('Configuring RL environment ...\n');
	[env, obsInfo, actInfo] = setup_rl_environment(modelName); %#ok<NASGU>
	fprintf('Done. Model and environment are ready.\n');
	fprintf('Next: run train_rl_agent(''td3'') to start training.\n');
else
	warning(['Reinforcement Learning Toolbox is not available. ', ...
			 'Created model with placeholder RL Agent and skipped environment setup.']);
	fprintf(['You can run classical phases now (phase1_model, phase2_classical_demo, ', ...
			 'compare_controllers, phase6_robustness_tests).\n']);
	fprintf('Install RL Toolbox, then rerun one_click_build_rl_model for RL training support.\n');
end
