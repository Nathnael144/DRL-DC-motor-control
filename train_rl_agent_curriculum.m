function [agent, statsStage1, statsStage2] = train_rl_agent_curriculum(maxEpisodesStage1, maxEpisodesStage2, showPlots)
%TRAIN_RL_AGENT_CURRICULUM Two-stage TD3 curriculum training.
%   [agent,s1,s2] = train_rl_agent_curriculum(eps1,eps2,showPlots)
%   Stage 1 (easy): zero load disturbance, narrow reference range.
%   Stage 2 (hard): nonzero load disturbance, full reference range.
%
%   Saves: trained_rl_agent_curriculum.mat

if nargin < 1 || isempty(maxEpisodesStage1)
    maxEpisodesStage1 = 300;
end
if nargin < 2 || isempty(maxEpisodesStage2)
    maxEpisodesStage2 = 900;
end
if nargin < 3 || isempty(showPlots)
    showPlots = false;
end

thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir)
    thisDir = pwd;
end
oldDir = pwd;
cleanupObj = onCleanup(@() cd(oldDir)); %#ok<NASGU>
cd(thisDir);
addpath(thisDir);

% Ensure environment + tuned reward profile are applied.
[env, obsInfo, actInfo] = setup_rl_environment('dc_motor_rl');
configure_rl_training_profile('dc_motor_rl');
[env, obsInfo, actInfo] = setup_rl_environment('dc_motor_rl');

% Build TD3 agent
initOpts = rlAgentInitializationOptions(NumHiddenUnit=256);
agent = rlTD3Agent(obsInfo, actInfo, initOpts);

agent = localConfigureAgent(agent);

plotMode = 'none';
if showPlots
    plotMode = 'training-progress';
end

fprintf('--- Curriculum Stage 1 (easy regime) ---\n');
env.ResetFcn = @localResetFcnEasy;
opts1 = rlTrainingOptions( ...
    'MaxEpisodes', maxEpisodesStage1, ...
    'MaxStepsPerEpisode', 5000, ...
    'StopTrainingCriteria', 'EpisodeCount', ...
    'StopTrainingValue', maxEpisodesStage1, ...
    'ScoreAveragingWindowLength', 20, ...
    'Verbose', false, ...
    'Plots', plotMode);
statsStage1 = train(agent, env, opts1);

% Reduce exploration for stage 2 refinement.
if isprop(agent.AgentOptions,'ExplorationModel')
    agent.AgentOptions.ExplorationModel.StandardDeviation = 0.12;
    agent.AgentOptions.ExplorationModel.StandardDeviationDecayRate = 5e-5;
    agent.AgentOptions.ExplorationModel.StandardDeviationMin = 0.005;
end

fprintf('--- Curriculum Stage 2 (disturbance/uncertainty regime) ---\n');
env.ResetFcn = @localResetFcnHard;
opts2 = rlTrainingOptions( ...
    'MaxEpisodes', maxEpisodesStage2, ...
    'MaxStepsPerEpisode', 5000, ...
    'StopTrainingCriteria', 'EpisodeCount', ...
    'StopTrainingValue', maxEpisodesStage2, ...
    'ScoreAveragingWindowLength', 30, ...
    'Verbose', false, ...
    'Plots', plotMode);
statsStage2 = train(agent, env, opts2);

save(fullfile(thisDir,'trained_rl_agent_curriculum.mat'), ...
    'agent','statsStage1','statsStage2','opts1','opts2');
fprintf('Saved curriculum-trained agent to %s\n', ...
    fullfile(thisDir,'trained_rl_agent_curriculum.mat'));

try
    bind_agent_to_model('dc_motor_rl', fullfile(thisDir,'trained_rl_agent_curriculum.mat'));
catch ME
    warning('Could not bind curriculum agent to model: %s', ME.message);
end

% Policy sanity snapshot
obsList = {[100;0;0;0],[50;0;0;0],[10;0;0;0],[-10;0;0;0]};
for i = 1:numel(obsList)
    o = obsList{i};
    a = getAction(agent,o);
    if iscell(a), a = a{1}; end
    fprintf('Curriculum action obs[%d]=[%6.1f %6.1f %6.1f %6.1f]: % .4f\n', ...
        i,o(1),o(2),o(3),o(4),double(a(1)));
end

end

function agent = localConfigureAgent(agent)
if evalin('base','exist(''Ts'',''var'')') == 1
    agent.AgentOptions.SampleTime = evalin('base','Ts');
else
    agent.AgentOptions.SampleTime = 1e-4;
end

agent.AgentOptions.DiscountFactor = 0.995;
agent.AgentOptions.ExperienceBufferLength = 1e6;
agent.AgentOptions.MiniBatchSize = 256;
agent.AgentOptions.TargetSmoothFactor = 5e-3;
agent.AgentOptions.TargetUpdateFrequency = 2;
agent.AgentOptions.PolicyUpdateFrequency = 2;
agent.AgentOptions.NumWarmStartSteps = 3000;
agent.AgentOptions.NumEpoch = 1;
agent.AgentOptions.MaxMiniBatchPerEpoch = 200;
agent.AgentOptions.ResetExperienceBufferBeforeTraining = true;

if isprop(agent.AgentOptions,'ExplorationModel')
    agent.AgentOptions.ExplorationModel.Mean = 0;
    agent.AgentOptions.ExplorationModel.StandardDeviation = 0.20;
    agent.AgentOptions.ExplorationModel.StandardDeviationDecayRate = 8e-5;
    agent.AgentOptions.ExplorationModel.StandardDeviationMin = 0.01;
    agent.AgentOptions.ExplorationModel.LowerLimit = -1;
    agent.AgentOptions.ExplorationModel.UpperLimit = 1;
end

if isprop(agent.AgentOptions,'TargetPolicySmoothModel')
    agent.AgentOptions.TargetPolicySmoothModel.Mean = 0;
    agent.AgentOptions.TargetPolicySmoothModel.StandardDeviation = 0.12;
    agent.AgentOptions.TargetPolicySmoothModel.StandardDeviationDecayRate = 1e-5;
    agent.AgentOptions.TargetPolicySmoothModel.StandardDeviationMin = 0.02;
    agent.AgentOptions.TargetPolicySmoothModel.LowerLimit = -0.25;
    agent.AgentOptions.TargetPolicySmoothModel.UpperLimit = 0.25;
end

agent.AgentOptions.ActorOptimizerOptions.LearnRate = 1e-4;
agent.AgentOptions.ActorOptimizerOptions.GradientThreshold = 1;
for i = 1:numel(agent.AgentOptions.CriticOptimizerOptions)
    agent.AgentOptions.CriticOptimizerOptions(i).LearnRate = 5e-4;
    agent.AgentOptions.CriticOptimizerOptions(i).GradientThreshold = 1;
end
end

function in = localResetFcnEasy(in)
% Stage 1: easier regime, no disturbance.
in = setVariable(in,'w_ref', 60 + 50*rand);   % 60..110 rad/s
in = setVariable(in,'Tload', 0.0);
end

function in = localResetFcnHard(in)
% Stage 2: harder regime with load variation.
in = setVariable(in,'w_ref', 80 + 40*rand);        % 80..120 rad/s
in = setVariable(in,'Tload', 0.015*(2*rand - 1));  % +/-0.015 N.m
end
