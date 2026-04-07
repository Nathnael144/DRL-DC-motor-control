function [agent, trainingStats] = train_rl_agent_sac_github_style(maxEpisodes, showPlots, trainingControl, warmStartAgent)
%TRAIN_RL_AGENT_SAC_GITHUB_STYLE
% Train SAC using the EXACT same GitHub-style flow that works for DDPG.
% The only differences from the DDPG trainer are:
%   - rlSACAgent instead of rlDDPGAgent
%   - SAC-specific options (twin critics, entropy tuning)
% Everything else (env setup, training options, chunking) is identical.
%
%   warmStartAgent (optional): a pre-trained rlSACAgent or path to .mat
%     file containing 'agent'. If provided, training continues from this
%     agent instead of creating a new one from scratch.

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
    warmStartAgent = [];
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

% Same easy reset function as DDPG pretd3 mode.
env.ResetFcn = @localResetFcnSAC;

% ---- Agent creation / warm-start ----
warmStarted = false;
if ~isempty(warmStartAgent)
    agent = localLoadWarmStartAgent(warmStartAgent, obsInfo, actInfo);
    if ~isempty(agent)
        warmStarted = true;
        fprintf('SAC warm-start: continuing from pre-trained agent.\n');
    end
end

if ~warmStarted
    initOpts = rlAgentInitializationOptions(NumHiddenUnit=256);
    agent = rlSACAgent(obsInfo, actInfo, initOpts);
    fprintf('SAC agent created from scratch.\n');
end

if evalin('base','exist(''Ts'',''var'')') == 1
    agent.AgentOptions.SampleTime = evalin('base','Ts');
else
    agent.AgentOptions.SampleTime = 1e-4;
end

% ---- Agent options ----
agent.AgentOptions.DiscountFactor = 0.99;
agent.AgentOptions.TargetSmoothFactor = 5e-3;
agent.AgentOptions.ExperienceBufferLength = 1e6;
agent.AgentOptions.MiniBatchSize = 256;
agent.AgentOptions.NumWarmStartSteps = 1000;  % fill replay buffer before learning

if warmStarted
    % Keep existing experience buffer; use lower LR for fine-tuning
    agent.AgentOptions.ResetExperienceBufferBeforeTraining = false;
    agent.AgentOptions.ActorOptimizerOptions.LearnRate = 3e-5;   % 3x lower
    agent.AgentOptions.ActorOptimizerOptions.GradientThreshold = 1;
    for ci = 1:numel(agent.AgentOptions.CriticOptimizerOptions)
        agent.AgentOptions.CriticOptimizerOptions(ci).LearnRate = 1e-4;  % 3x lower
        agent.AgentOptions.CriticOptimizerOptions(ci).GradientThreshold = 1;
    end
else
    agent.AgentOptions.ResetExperienceBufferBeforeTraining = true;
    agent.AgentOptions.ActorOptimizerOptions.LearnRate = 1e-4;
    agent.AgentOptions.ActorOptimizerOptions.GradientThreshold = 1;
    for ci = 1:numel(agent.AgentOptions.CriticOptimizerOptions)
        agent.AgentOptions.CriticOptimizerOptions(ci).LearnRate = 3e-4;
        agent.AgentOptions.CriticOptimizerOptions(ci).GradientThreshold = 1;
    end
end

% SAC-specific: entropy weight tuning
if isprop(agent.AgentOptions,'EntropyWeightOptions')
    try
        agent.AgentOptions.EntropyWeightOptions.LearnRate = 3e-4;
    catch
    end
end
if isprop(agent.AgentOptions,'EntropyWeight')
    try
        if warmStarted
            agent.AgentOptions.EntropyWeight = 0.05;  % less exploration for fine-tuning
        else
            agent.AgentOptions.EntropyWeight = 0.1;
        end
    catch
    end
end

fprintf('SAC agent ready (warmStart=%s).\n', string(warmStarted));

% ---- Training options: identical to DDPG ----
plotMode = 'none';
if showPlots
    plotMode = 'training-progress';
end

maxStepsPerEpisode = 5000;   % Same as DDPG
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
    fprintf('GitHub-style SAC early-stop run completed at %d/%d episodes.\n', ...
        episodesCompleted, maxEpisodes);
else
    trainingStats = train(agent, env, trainOpts);
end

save(fullfile(thisDir,'trained_rl_agent_sac_github_style.mat'), ...
    'agent','trainingStats','trainOpts','trainingControl');
fprintf('Saved GitHub-style SAC trained agent to %s\n', ...
    fullfile(thisDir,'trained_rl_agent_sac_github_style.mat'));

try
    bind_agent_to_model('dc_motor_rl', fullfile(thisDir,'trained_rl_agent_sac_github_style.mat'));
catch ME
    warning('Could not bind GitHub-style SAC agent to model: %s', ME.message);
end

end

%% ---- Local helpers (mirrored from DDPG trainer) ----

function in = localResetFcnSAC(in)
% Broad operating region covering all evaluation scenarios (w_ref=100..120).
% Old range 50-90 never included eval operating point!
in = setVariable(in,'w_ref', 60 + 80*rand);      % 60..140 rad/s
in = setVariable(in,'Tload', 0.015*(2*rand-1));   % +/-0.015 N.m (evals use 0..0.02)
end

function agentOut = localLoadWarmStartAgent(src, obsInfo, actInfo) %#ok<INUSD>
%LOCALLOADWARMSTARTAGENT Load an existing SAC agent for warm-starting.
agentOut = [];

if isa(src,'rl.agent.rlSACAgent') || isa(src,'rl.agent.AbstractAgent')
    agentOut = src;
    return;
end

if ischar(src) || isstring(src)
    f = char(src);
    if exist(f,'file') ~= 2
        warning('Warm-start file not found: %s. Starting from scratch.', f);
        return;
    end
    try
        S = load(f,'agent');
        if isfield(S,'agent')
            agentOut = S.agent;
            fprintf('Loaded warm-start agent from: %s\n', f);
        else
            warning('File %s does not contain ''agent'' variable.', f);
        end
    catch ME
        warning('Failed to load warm-start agent: %s', ME.message);
    end
end
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
        warning('Could not extract episode rewards from SAC chunk %d; stopping.', chunkId);
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

    fprintf(['SAC chunk %d | episodes=%d/%d | score(Last%d)=%.2f | ', ...
             'best=%.2f | noImprove=%d/%d\n'], ...
        chunkId, episodesCompleted, maxEpisodes, nWin, currentScore, ...
        bestScore, noImproveCount, cfg.patienceChunks);

    if episodesCompleted >= cfg.minEpisodesBeforeStop && noImproveCount >= cfg.patienceChunks
        fprintf('SAC early-stop triggered: reward plateau detected.\n');
        break;
    end
end

if chunkId >= cfg.maxChunks && episodesCompleted < maxEpisodes
    fprintf('SAC stop: reached maxChunks=%d.\n', cfg.maxChunks);
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
trainingStats.Mode = 'sac-github-style-chunked';
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
