function [agent, trainingStats] = train_rl_agent_tuned(maxEpisodes, showPlots)
%TRAIN_RL_AGENT_TUNED Retrain TD3 with tuned reward + observation normalization.
%   [agent,stats] = train_rl_agent_tuned(maxEpisodes, showPlots)
%   saves output to trained_rl_agent_tuned.mat

if nargin < 1 || isempty(maxEpisodes)
    maxEpisodes = 1000;
end
if nargin < 2 || isempty(showPlots)
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

% Ensure model + env exist
[env, obsInfo, actInfo] = setup_rl_environment('dc_motor_rl');

% Apply tuned model profile and rebind env to updated model
configure_rl_training_profile('dc_motor_rl');
[env, obsInfo, actInfo] = setup_rl_environment('dc_motor_rl');
env.ResetFcn = @localResetFcnTuned;

% Build TD3 agent
initOpts = rlAgentInitializationOptions(NumHiddenUnit=256);
agent = rlTD3Agent(obsInfo, actInfo, initOpts);

% Core options
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

% Exploration and target smoothing noise
if isprop(agent.AgentOptions,'ExplorationModel')
    agent.AgentOptions.ExplorationModel.Mean = 0;
    agent.AgentOptions.ExplorationModel.StandardDeviation = 0.25;
    agent.AgentOptions.ExplorationModel.StandardDeviationDecayRate = 1e-4;
    agent.AgentOptions.ExplorationModel.StandardDeviationMin = 0.01;
    agent.AgentOptions.ExplorationModel.LowerLimit = -1;
    agent.AgentOptions.ExplorationModel.UpperLimit = 1;
end
if isprop(agent.AgentOptions,'TargetPolicySmoothModel')
    agent.AgentOptions.TargetPolicySmoothModel.Mean = 0;
    agent.AgentOptions.TargetPolicySmoothModel.StandardDeviation = 0.15;
    agent.AgentOptions.TargetPolicySmoothModel.StandardDeviationDecayRate = 1e-5;
    agent.AgentOptions.TargetPolicySmoothModel.StandardDeviationMin = 0.03;
    agent.AgentOptions.TargetPolicySmoothModel.LowerLimit = -0.3;
    agent.AgentOptions.TargetPolicySmoothModel.UpperLimit = 0.3;
end

% Optimizers
agent.AgentOptions.ActorOptimizerOptions.LearnRate = 1e-4;
agent.AgentOptions.ActorOptimizerOptions.GradientThreshold = 1;
for i = 1:numel(agent.AgentOptions.CriticOptimizerOptions)
    agent.AgentOptions.CriticOptimizerOptions(i).LearnRate = 5e-4;
    agent.AgentOptions.CriticOptimizerOptions(i).GradientThreshold = 1;
end

% Training options
plotMode = 'none';
if showPlots
    plotMode = 'training-progress';
end

save(fullfile(thisDir,'trained_rl_agent_tuned.mat'), ...
fprintf('Saved tuned agent to %s\n', fullfile(thisDir,'trained_rl_agent_tuned.mat'));

chunkSize = 50;
totalEpisodes = 1000;
numChunks = totalEpisodes / chunkSize;
allStats = cell(numChunks,1);
episodesDone = 0;

for chunk = 1:numChunks
    fprintf('\n--- Training chunk %d/%d: episodes %d to %d ---\n', chunk, numChunks, episodesDone+1, episodesDone+chunkSize);
    trainOpts = rlTrainingOptions( ...
        'MaxEpisodes', chunkSize, ...
        'MaxStepsPerEpisode', 5000, ...
        'StopTrainingCriteria', 'EpisodeCount', ...
        'StopTrainingValue', chunkSize, ...
        'ScoreAveragingWindowLength', 30, ...
        'Verbose', false, ...
        'Plots', plotMode);
    trainingStatsChunk = train(agent, env, trainOpts);
    allStats{chunk} = trainingStatsChunk;
    episodesDone = episodesDone + chunkSize;
    save(fullfile(thisDir,'trained_rl_agent_tuned.mat'), ...
        'agent','allStats','trainOpts','episodesDone','chunkSize');
    fprintf('Checkpoint saved after %d episodes.\n', episodesDone);
    % Optionally print summary stats for this chunk
    if isprop(trainingStatsChunk, 'AverageReward')
        avgR = trainingStatsChunk.AverageReward(end);
    elseif isfield(trainingStatsChunk, 'AverageReward')
        avgR = trainingStatsChunk.AverageReward(end);
    else
        avgR = NaN;
    end
    fprintf('Chunk %d complete. Last avg reward: %.2f\n', chunk, avgR);
end

trainingStats = allStats;

fprintf('Saved tuned agent to %s\n', fullfile(thisDir,'trained_rl_agent_tuned.mat'));

% Bind for interactive Simulink simulation convenience.
try
    bind_agent_to_model('dc_motor_rl', fullfile(thisDir,'trained_rl_agent_tuned.mat'));
catch ME
    warning('Could not auto-bind tuned agent to model: %s', ME.message);
end

% Quick policy sanity snapshot after training
obsList = {[100;0;0;0],[50;0;0;0],[10;0;0;0],[-10;0;0;0]};
for i = 1:numel(obsList)
    o = obsList{i};
    a = getAction(agent,o);
    if iscell(a), a = a{1}; end
    fprintf('Snapshot action for obs[%d]=[%6.1f %6.1f %6.1f %6.1f]: % .4f\n', ...
        i,o(1),o(2),o(3),o(4),double(a(1)));
end

end

function in = localResetFcnTuned(in)
% Curriculum-style reset focused on practical operating region.
in = setVariable(in,'w_ref', 80 + 40*rand);        % 80..120 rad/s (near eval regime)
in = setVariable(in,'Tload', 0.01*(2*rand - 1));   % +/-0.01 N.m
end
