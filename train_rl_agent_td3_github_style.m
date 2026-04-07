function [agent, trainingStats] = train_rl_agent_td3_github_style(maxEpisodes, showPlots, trainingControl, ddpgWarmStartAgent)
%TRAIN_RL_AGENT_TD3_GITHUB_STYLE
% Train TD3 using the EXACT same GitHub-style flow that works for DDPG.
% The only differences from the DDPG trainer are:
%   - rlTD3Agent instead of rlDDPGAgent
%   - TD3-specific options (twin critics, policy smoothing)
%   - Optional warm-start from trained DDPG actor
% Everything else (env setup, training options, chunking) is identical.

if nargin < 1 || isempty(maxEpisodes)
    maxEpisodes = 1000;
end
if nargin < 2 || isempty(showPlots)
    showPlots = false;
end
if nargin < 3 || isempty(trainingControl)
    trainingControl = struct();
end
if nargin < 4
    ddpgWarmStartAgent = [];
end

trainingControl = localNormalizeTrainingControl(trainingControl, maxEpisodes);

thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir)
    thisDir = pwd;
end
oldDir = pwd;
cleanupObj = onCleanup(@() cd(oldDir)); %#ok<NASGU>
cd(thisDir);
addpath(thisDir);

% ---- Environment: identical to DDPG trainer, NO modifications ----
[env, obsInfo, actInfo] = setup_rl_environment_github_style('dc_motor_rl');

% Use the same easy reset function that works in DDPG pretd3 mode.
env.ResetFcn = @localResetFcnTD3;

% ---- Agent creation ----
initOpts = rlAgentInitializationOptions(NumHiddenUnit=128);
agent = rlTD3Agent(obsInfo, actInfo, initOpts);

if evalin('base','exist(''Ts'',''var'')') == 1
    agent.AgentOptions.SampleTime = evalin('base','Ts');
else
    agent.AgentOptions.SampleTime = 1e-4;
end

% ---- Agent options: mirror DDPG exactly, plus TD3-specific ----
agent.AgentOptions.DiscountFactor = 0.995;
agent.AgentOptions.TargetSmoothFactor = 1e-3;
agent.AgentOptions.ExperienceBufferLength = 1e6;
agent.AgentOptions.MiniBatchSize = 128;          % Same as DDPG (was 32)
agent.AgentOptions.ResetExperienceBufferBeforeTraining = true;

% TD3-specific: delayed policy update
if isprop(agent.AgentOptions,'PolicyUpdateFrequency')
    agent.AgentOptions.PolicyUpdateFrequency = 2;
end
if isprop(agent.AgentOptions,'TargetUpdateFrequency')
    agent.AgentOptions.TargetUpdateFrequency = 2;
end

% Exploration noise: same as DDPG pretd3 mode
if isprop(agent.AgentOptions,'ExplorationModel')
    agent.AgentOptions.ExplorationModel.Mean = 0;
    agent.AgentOptions.ExplorationModel.StandardDeviation = 0.12;
    agent.AgentOptions.ExplorationModel.StandardDeviationDecayRate = 6e-5;
    agent.AgentOptions.ExplorationModel.StandardDeviationMin = 0.005;
    agent.AgentOptions.ExplorationModel.LowerLimit = 0;
    agent.AgentOptions.ExplorationModel.UpperLimit = 1;
elseif isprop(agent.AgentOptions,'NoiseOptions')
    agent.AgentOptions.NoiseOptions.StandardDeviation = 0.12;
    agent.AgentOptions.NoiseOptions.StandardDeviationDecayRate = 6e-5;
end

% Target policy smoothing (TD3-specific)
if isprop(agent.AgentOptions,'TargetPolicySmoothModel')
    sm = agent.AgentOptions.TargetPolicySmoothModel;
    sm.Mean = 0;
    sm.StandardDeviation = 0.08;
    sm.StandardDeviationDecayRate = 1e-5;
    sm.StandardDeviationMin = 0.02;
    sm.LowerLimit = -0.1;
    sm.UpperLimit = 0.1;
end

% Optimizer: same as DDPG
agent.AgentOptions.ActorOptimizerOptions.LearnRate = 1e-4;
agent.AgentOptions.ActorOptimizerOptions.GradientThreshold = 1;
for ci = 1:numel(agent.AgentOptions.CriticOptimizerOptions)
    agent.AgentOptions.CriticOptimizerOptions(ci).LearnRate = 5e-4;
    agent.AgentOptions.CriticOptimizerOptions(ci).GradientThreshold = 1;
end

% ---- Warm-start from DDPG actor (optional) ----
didWarmStart = false;
if ~isempty(ddpgWarmStartAgent)
    try
        ddpgActor = getActor(ddpgWarmStartAgent);
        agent = setActor(agent, ddpgActor);
        didWarmStart = true;
    catch
        try
            ddpgActor = getActor(ddpgWarmStartAgent);
            actorModel = getModel(ddpgActor);
            actor = rlContinuousDeterministicActor(actorModel, obsInfo, actInfo, ...
                'ObservationInputNames','state');
            agent = setActor(agent, actor);
            didWarmStart = true;
        catch
        end
    end
end
if didWarmStart
    fprintf('TD3 warm-start: initialized actor from trained DDPG model.\n');
else
    fprintf('TD3 warm-start: no DDPG actor transfer applied.\n');
end

% ---- Training options: identical to DDPG ----
plotMode = 'none';
if showPlots
    plotMode = 'training-progress';
end

maxStepsPerEpisode = 5000;   % Same as DDPG (was 400/800)
scoreWindow = 20;

trainOpts = rlTrainingOptions( ...
    'MaxEpisodes', maxEpisodes, ...
    'MaxStepsPerEpisode', maxStepsPerEpisode, ...
    'ScoreAveragingWindowLength', scoreWindow, ...
    'StopTrainingCriteria', 'EpisodeCount', ...
    'StopTrainingValue', maxEpisodes, ...
    'Verbose', false, ...
    'Plots', plotMode);

% ---- Chunked training: identical structure to DDPG ----
if trainingControl.enableEarlyStop
    [agent, trainingStats, episodesCompleted] = localTrainWithPlateauStop( ...
        agent, env, trainOpts, maxEpisodes, trainingControl);
    fprintf('GitHub-style TD3 early-stop run completed at %d/%d episodes.\n', ...
        episodesCompleted, maxEpisodes);
else
    trainingStats = train(agent, env, trainOpts);
end

save(fullfile(thisDir,'trained_rl_agent_td3_github_style.mat'), ...
    'agent','trainingStats','trainOpts','trainingControl');
fprintf('Saved GitHub-style TD3 trained agent to %s\n', ...
    fullfile(thisDir,'trained_rl_agent_td3_github_style.mat'));

try
    bind_agent_to_model('dc_motor_rl', fullfile(thisDir,'trained_rl_agent_td3_github_style.mat'));
catch ME
    warning('Could not bind GitHub-style TD3 agent to model: %s', ME.message);
end

end

%% ---- Local helpers (mirrored from DDPG trainer) ----

function in = localResetFcnTD3(in)
% Same easy operating region as DDPG pretd3 mode.
in = setVariable(in,'w_ref', 50 + 40*rand);  % 50..90 rad/s
in = setVariable(in,'Tload', 0.0);
end

function cfg = localNormalizeTrainingControl(cfgIn, maxEpisodes)
defaults = struct( ...
    'enableEarlyStop', false, ...
    'maxChunks', inf, ...
    'chunkEpisodes', 25, ...
    'minEpisodesBeforeStop', 100, ...
    'patienceChunks', 3, ...
    'minImprovement', 75, ...
    'scoreWindow', 30);

cfg = defaults;
if ~isstruct(cfgIn)
    return;
end

f = fieldnames(defaults);
for i = 1:numel(f)
    k = f{i};
    if isfield(cfgIn, k)
        cfg.(k) = cfgIn.(k);
    end
end

cfg.enableEarlyStop = logical(cfg.enableEarlyStop);
cfg.maxChunks = double(cfg.maxChunks);
cfg.chunkEpisodes = max(1, round(double(cfg.chunkEpisodes)));
cfg.minEpisodesBeforeStop = max(1, round(double(cfg.minEpisodesBeforeStop)));
cfg.patienceChunks = max(1, round(double(cfg.patienceChunks)));
cfg.minImprovement = double(cfg.minImprovement);
cfg.scoreWindow = max(1, round(double(cfg.scoreWindow)));

if ~(isfinite(cfg.maxChunks) && cfg.maxChunks > 0)
    cfg.maxChunks = inf;
else
    cfg.maxChunks = max(1, round(cfg.maxChunks));
end

cfg.chunkEpisodes = min(cfg.chunkEpisodes, max(1, round(double(maxEpisodes))));
cfg.minEpisodesBeforeStop = min(cfg.minEpisodesBeforeStop, max(1, round(double(maxEpisodes))));
end

function [agentOut, trainingStats, episodesCompleted] = localTrainWithPlateauStop(agentIn, env, trainOptsBase, maxEpisodes, cfg)
% Identical to DDPG localTrainWithPlateauStop.
agentOut = agentIn;
episodesCompleted = 0;
chunkId = 0;
bestScore = -inf;
noImproveCount = 0;
allRewards = [];

while episodesCompleted < maxEpisodes && chunkId < cfg.maxChunks
    chunkId = chunkId + 1;
    chunkEpisodes = min(cfg.chunkEpisodes, maxEpisodes - episodesCompleted);

    try
        agentOut.AgentOptions.ResetExperienceBufferBeforeTraining = (chunkId == 1);
    catch
    end

    chunkOpts = trainOptsBase;
    chunkOpts.MaxEpisodes = chunkEpisodes;
    chunkOpts.StopTrainingCriteria = 'EpisodeCount';
    chunkOpts.StopTrainingValue = chunkEpisodes;

    statsChunk = train(agentOut, env, chunkOpts);
    chunkRewards = localExtractEpisodeRewards(statsChunk);
    if isempty(chunkRewards)
        warning('Could not extract episode rewards from TD3 chunk %d; stopping.', chunkId);
        break;
    end

    allRewards = [allRewards; chunkRewards(:)]; %#ok<AGROW>
    episodesCompleted = numel(allRewards);

    nWin = min(cfg.scoreWindow, numel(allRewards));
    currentScore = mean(allRewards(end-nWin+1:end));

    if currentScore > (bestScore + cfg.minImprovement)
        bestScore = currentScore;
        noImproveCount = 0;
    else
        noImproveCount = noImproveCount + 1;
    end

    fprintf(['TD3 chunk %d | episodes=%d/%d | score(Last%d)=%.2f | ', ...
             'best=%.2f | noImprove=%d/%d\n'], ...
        chunkId, episodesCompleted, maxEpisodes, nWin, currentScore, ...
        bestScore, noImproveCount, cfg.patienceChunks);

    if episodesCompleted >= cfg.minEpisodesBeforeStop && noImproveCount >= cfg.patienceChunks
        fprintf('TD3 early-stop triggered: reward plateau detected.\n');
        break;
    end
end

if chunkId >= cfg.maxChunks && episodesCompleted < maxEpisodes
    fprintf('TD3 stop: reached maxChunks=%d.\n', cfg.maxChunks);
end

if isempty(allRewards)
    trainingStats = struct('EpisodeReward',[],'AverageReward',[],'EpisodesCompleted',episodesCompleted,'EarlyStopConfig',cfg);
    return;
end

trainingStats = struct();
trainingStats.EpisodeReward = allRewards(:);
trainingStats.AverageReward = movmean(allRewards(:), [cfg.scoreWindow-1 0]);
trainingStats.EpisodesCompleted = episodesCompleted;
trainingStats.EarlyStopConfig = cfg;
trainingStats.Mode = 'td3-github-style-chunked';
end

function rewards = localExtractEpisodeRewards(stats)
rewards = [];

if isempty(stats)
    return;
end

if istable(stats)
    vn = stats.Properties.VariableNames;
    if ismember('EpisodeReward', vn)
        rewards = double(stats.EpisodeReward(:));
        rewards = rewards(isfinite(rewards));
        return;
    end
end

try
    rewards = double(stats.EpisodeReward(:));
    rewards = rewards(isfinite(rewards));
catch
    rewards = [];
end
end
