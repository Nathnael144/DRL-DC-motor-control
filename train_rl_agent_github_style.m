function [agent, trainingStats] = train_rl_agent_github_style(maxEpisodes, showPlots, trainingMode, trainingControl)
%TRAIN_RL_AGENT_GITHUB_STYLE
% Train DDPG in a GitHub-inspired setup:
% - compact 3-state observation
% - positive action [0,1] mapped to [0,Vmax]
% - optional pre-TD3 mode for smoother pretraining
% - optional plateau-based early stopping in chunked training

if nargin < 1 || isempty(maxEpisodes)
    maxEpisodes = 1000;
end
if nargin < 2 || isempty(showPlots)
    showPlots = false;
end
if nargin < 3 || isempty(trainingMode)
    trainingMode = "default";
end
if nargin < 4 || isempty(trainingControl)
    trainingControl = struct();
end
trainingMode = lower(string(trainingMode));
if ~ismember(trainingMode, ["default","pretd3"])
    error('trainingMode must be "default" or "pretd3".');
end
isPreTD3 = trainingMode == "pretd3";

trainingControl = localNormalizeTrainingControl(trainingControl, maxEpisodes);

thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir)
    thisDir = pwd;
end
oldDir = pwd;
cleanupObj = onCleanup(@() cd(oldDir)); %#ok<NASGU>
cd(thisDir);
addpath(thisDir);

[env, obsInfo, actInfo] = setup_rl_environment_github_style('dc_motor_rl');

if isPreTD3
    % Easier reset and a slightly stronger tracking bonus before TD3 handoff.
    env.ResetFcn = @localResetFcnPreTD3;
    assignin('base','TrackBonus',8.0);
    assignin('base','TrackTol',2.5);
end

initOpts = rlAgentInitializationOptions(NumHiddenUnit=128);
agent = rlDDPGAgent(obsInfo, actInfo, initOpts);

if evalin('base','exist(''Ts'',''var'')') == 1
    agent.AgentOptions.SampleTime = evalin('base','Ts');
else
    agent.AgentOptions.SampleTime = 1e-4;
end

agent.AgentOptions.DiscountFactor = 0.995;
agent.AgentOptions.TargetSmoothFactor = 1e-3;
agent.AgentOptions.ExperienceBufferLength = 1e6;
agent.AgentOptions.MiniBatchSize = 128;
agent.AgentOptions.ResetExperienceBufferBeforeTraining = true;

if isprop(agent.AgentOptions,'ExplorationModel')
    agent.AgentOptions.ExplorationModel.Mean = 0;
    if isPreTD3
        agent.AgentOptions.ExplorationModel.StandardDeviation = 0.12;
        agent.AgentOptions.ExplorationModel.StandardDeviationDecayRate = 6e-5;
        agent.AgentOptions.ExplorationModel.StandardDeviationMin = 0.005;
    else
        agent.AgentOptions.ExplorationModel.StandardDeviation = 0.15;
        agent.AgentOptions.ExplorationModel.StandardDeviationDecayRate = 5e-5;
        agent.AgentOptions.ExplorationModel.StandardDeviationMin = 0.005;
    end
    agent.AgentOptions.ExplorationModel.LowerLimit = 0;
    agent.AgentOptions.ExplorationModel.UpperLimit = 1;
elseif isprop(agent.AgentOptions,'NoiseOptions')
    if isPreTD3
        agent.AgentOptions.NoiseOptions.StandardDeviation = 0.12;
        agent.AgentOptions.NoiseOptions.StandardDeviationDecayRate = 6e-5;
    else
        agent.AgentOptions.NoiseOptions.StandardDeviation = 0.15;
        agent.AgentOptions.NoiseOptions.StandardDeviationDecayRate = 5e-5;
    end
end

if isPreTD3
    agent.AgentOptions.ActorOptimizerOptions.LearnRate = 1e-4;
    agent.AgentOptions.CriticOptimizerOptions.LearnRate = 4e-4;
else
    agent.AgentOptions.ActorOptimizerOptions.LearnRate = 1e-4;
    agent.AgentOptions.CriticOptimizerOptions.LearnRate = 5e-4;
end
agent.AgentOptions.ActorOptimizerOptions.GradientThreshold = 1;
agent.AgentOptions.CriticOptimizerOptions.GradientThreshold = 1;

plotMode = 'none';
if showPlots
    plotMode = 'training-progress';
end

maxStepsPerEpisode = 5000;
scoreWindow = 20;
if isPreTD3
    % Keep full horizon so agent can see settling behavior.
    maxStepsPerEpisode = 5000;
    scoreWindow = 30;
end

trainOpts = rlTrainingOptions( ...
    'MaxEpisodes', maxEpisodes, ...
    'MaxStepsPerEpisode', maxStepsPerEpisode, ...
    'ScoreAveragingWindowLength', scoreWindow, ...
    'StopTrainingCriteria', 'EpisodeCount', ...
    'StopTrainingValue', maxEpisodes, ...
    'Verbose', false, ...
    'Plots', plotMode);

if trainingControl.enableEarlyStop
    [agent, trainingStats, episodesCompleted] = localTrainWithPlateauStop( ...
        agent, env, trainOpts, maxEpisodes, trainingControl);
    fprintf(['GitHub-style (%s) early-stop run completed at %d/%d episodes ', ...
             '(patience=%d, minImprovement=%.2f).\n'], ...
        char(trainingMode), episodesCompleted, maxEpisodes, ...
        trainingControl.patienceChunks, trainingControl.minImprovement);
else
    trainingStats = train(agent, env, trainOpts);
end

save(fullfile(thisDir,'trained_rl_agent_github_style.mat'), ...
    'agent','trainingStats','trainOpts','trainingMode','trainingControl');
fprintf('Saved GitHub-style trained agent (%s mode) to %s\n', ...
    char(trainingMode), fullfile(thisDir,'trained_rl_agent_github_style.mat'));

try
    bind_agent_to_model('dc_motor_rl', fullfile(thisDir,'trained_rl_agent_github_style.mat'));
catch ME
    warning('Could not bind GitHub-style agent to model: %s', ME.message);
end

function in = localResetFcnPreTD3(in)
% Easier operating region for smoother DDPG pretraining before TD3.
in = setVariable(in,'w_ref', 50 + 40*rand);  % 50..90 rad/s
in = setVariable(in,'Tload', 0.0);            % no disturbance in pretrain
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
        warning('Could not extract episode rewards from DDPG chunk %d; stopping early-stop loop.', chunkId);
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

    fprintf(['DDPG preTD3 chunk %d | episodes=%d/%d | score(Last%d)=%.2f | ', ...
             'best=%.2f | noImprove=%d/%d\n'], ...
        chunkId, episodesCompleted, maxEpisodes, nWin, currentScore, ...
        bestScore, noImproveCount, cfg.patienceChunks);

    if episodesCompleted >= cfg.minEpisodesBeforeStop && noImproveCount >= cfg.patienceChunks
        fprintf('DDPG preTD3 early-stop triggered: reward plateau detected.\n');
        break;
    end
end

if chunkId >= cfg.maxChunks && episodesCompleted < maxEpisodes
    fprintf('DDPG preTD3 stop: reached maxChunks=%d.\n', cfg.maxChunks);
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
trainingStats.Mode = 'chunked-plateau-stop';
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

obsList = {[0;100;0],[0;50;0],[0;10;0],[0;-10;0]};
for i = 1:numel(obsList)
    o = obsList{i};
    a = getAction(agent,o);
    if iscell(a), a = a{1}; end
    fprintf('GitHub-style action obs[%d]=[%6.1f %6.1f %6.1f]: % .4f\n', ...
        i,o(1),o(2),o(3),double(a(1)));
end

end
