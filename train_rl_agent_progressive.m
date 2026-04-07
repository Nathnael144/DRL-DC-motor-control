function results = train_rl_agent_progressive(maxEpisodesConfig, showTrainingMonitor, doPostEvaluation, algorithmOrder)
%TRAIN_RL_AGENT_PROGRESSIVE Progressive RL training for DC motor speed control.
%   results = train_rl_agent_progressive()
%   trains algorithms from simpler to harder in this default order:
%       DDPG -> TD3 -> SAC
%   and, for each algorithm, runs a two-stage curriculum:
%       Stage 1 (easy): low disturbance, narrower reference range
%       Stage 2 (hard): larger reference range + load disturbances
%
%   Inputs (all optional):
%     maxEpisodesConfig   scalar or struct with fields ddpg/td3/sac.
%                         Default struct: ddpg=500, td3=800, sac=1000
%     showTrainingMonitor true/false, default true (shows Training Monitor)
%     doPostEvaluation    true/false, default true
%                         if true, runs:
%                           generate_rl_scenario_data(bestAgent)
%                           compare_controllers.m
%                           create_graph_table_comparison.m (if present)
%     algorithmOrder      string array/cellstr subset of
%                         {"ddpg","td3","sac"}, default all in this order.
%
%   Outputs:
%     results struct containing per-algorithm files, estimated final scores,
%     and selected best algorithm/file.
%
%   Files created:
%     trained_rl_agent_ddpg_progressive.mat
%     trained_rl_agent_td3_progressive.mat
%     trained_rl_agent_sac_progressive.mat
%     trained_rl_agent.mat                         (best-agent compatibility)
%     trained_rl_agent_progressive_summary.mat

if nargin < 1 || isempty(maxEpisodesConfig)
    maxEpisodesConfig = struct('ddpg',500,'td3',800,'sac',1000);
end
if nargin < 2 || isempty(showTrainingMonitor)
    showTrainingMonitor = true;
end
if nargin < 3 || isempty(doPostEvaluation)
    doPostEvaluation = true;
end
if nargin < 4 || isempty(algorithmOrder)
    algorithmOrder = ["ddpg","td3","sac"];
end

if showTrainingMonitor && ~(usejava('desktop') && usejava('awt'))
    error(['showTrainingMonitor=true requires MATLAB Desktop (non-headless). ', ...
           'Start MATLAB with GUI and run again.']);
end

algorithmOrder = lower(string(algorithmOrder));
validAlgs = ["ddpg","td3","sac"];
algorithmOrder = algorithmOrder(ismember(algorithmOrder, validAlgs));
algorithmOrder = unique(algorithmOrder,'stable');
if isempty(algorithmOrder)
    error('algorithmOrder must include at least one of: ddpg, td3, sac.');
end

thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir)
    thisDir = pwd;
end
oldDir = pwd;
cleanupObj = onCleanup(@() cd(oldDir)); %#ok<NASGU>
cd(thisDir);
addpath(thisDir);

rng(0,'twister');

fprintf('=== Progressive RL training for DC motor speed control ===\n');
fprintf('Algorithms order: %s\n', strjoin(cellstr(upper(algorithmOrder)), ' -> '));

episodePlan = localNormalizeEpisodePlan(maxEpisodesConfig, algorithmOrder);

fprintf('Setting up GitHub-style environment/profile (stable training baseline)...\n');
[env, obsInfo, actInfo] = setup_rl_environment_github_style('dc_motor_rl'); %#ok<NASGU>

Ts = localGetSampleTime();
plotMode = 'none';
if showTrainingMonitor
    plotMode = 'training-progress';
end

fprintf('Training timing: Ts=%.6g s | MaxStepsPerEpisode=5000 (GitHub-style)\n', Ts);

progressiveResults = struct();
algoScores = -inf(1,numel(algorithmOrder));
algoFiles = strings(1,numel(algorithmOrder));
ddpgWarmStartAgent = [];

if ~ismember("ddpg", algorithmOrder) && ismember("td3", algorithmOrder)
    ddpgWarmStartAgent = localLoadDDPGWarmStart(thisDir);
    if isempty(ddpgWarmStartAgent)
        warning(['Algorithm order starts from TD3, but no trained DDPG model was found. ', ...
                 'TD3 will start from scratch.']);
    else
        fprintf('Loaded trained DDPG model for TD3 warm-start (DDPG stage skipped).\n');
    end
end

for i = 1:numel(algorithmOrder)
    alg = algorithmOrder(i);
    totalEpisodes = episodePlan.(char(alg));
    [episodesEasy, episodesHard] = localSplitEpisodes(totalEpisodes);

    fprintf('\n--- Training %s (total episodes: %d | easy: %d | hard: %d) ---\n', ...
        upper(alg), totalEpisodes, episodesEasy, episodesHard);

    statsEasy = [];
    optsEasy = [];
    optsHard = [];

    if alg == "ddpg"
        % Reuse the exact proven GitHub-style DDPG trainer for stable
        % episode progression in this project.
        fprintf(['DDPG stage uses train_rl_agent_github_style ', ...
                 '(pretd3 tuned mode: smoother rewards + faster handoff)...\n']);
        ddpgControl = struct( ...
            'enableEarlyStop', true, ...
            'maxChunks', 2, ...
            'chunkEpisodes', 25, ...
            'minEpisodesBeforeStop', 125, ...
            'patienceChunks', 4, ...
            'minImprovement', 25, ...
            'scoreWindow', 30);
        [agent, statsHard] = train_rl_agent_github_style( ...
            totalEpisodes, showTrainingMonitor, "pretd3", ddpgControl);
        ddpgWarmStartAgent = agent;

    elseif alg == "td3"
        % Use IDENTICAL control structure as DDPG (plateau-based early stop).
        fprintf(['TD3 stage uses train_rl_agent_td3_github_style ', ...
                 '(same flow as DDPG + DDPG warm-start + max 2 chunks)...\n']);
        td3Control = struct( ...
            'enableEarlyStop', true, ...
            'maxChunks', 2, ...
            'chunkEpisodes', 25, ...
            'minEpisodesBeforeStop', 100, ...
            'patienceChunks', 3, ...
            'minImprovement', 25, ...
            'scoreWindow', 20);
        [agent, statsHard] = train_rl_agent_td3_github_style( ...
            totalEpisodes, showTrainingMonitor, td3Control, ddpgWarmStartAgent);

        statsEasy = [];
        optsEasy = [];
        optsHard = td3Control;

    else
        % SAC: use dedicated GitHub-style trainer (same flow as DDPG/TD3).
        fprintf(['SAC stage uses train_rl_agent_sac_github_style ', ...
                 '(same flow as DDPG + max 2 chunks)...\n']);
        sacControl = struct( ...
            'enableEarlyStop', true, ...
            'maxChunks', 4, ...
            'chunkEpisodes', 500, ...
            'minEpisodesBeforeStop', 200, ...
            'patienceChunks', 4, ...
            'minImprovement', 15, ...
            'scoreWindow', 30);
        % Train SAC from scratch (action space changed to bipolar)
        [agent, statsHard] = train_rl_agent_sac_github_style( ...
            totalEpisodes, showTrainingMonitor, sacControl);

        statsEasy = [];
        optsEasy = [];
        optsHard = sacControl;
    end

    trainingStats = struct();
    trainingStats.easy = statsEasy;
    trainingStats.hard = statsHard;
    trainingStats.options.easy = optsEasy;
    trainingStats.options.hard = optsHard;

    agentFile = fullfile(thisDir, sprintf('trained_rl_agent_%s_progressive.mat', char(alg)));
    save(agentFile,'agent','trainingStats','alg','episodesEasy','episodesHard','totalEpisodes');
    fprintf('Saved %s agent to %s\n', upper(alg), agentFile);

    try
        bind_agent_to_model('dc_motor_rl', agentFile);
    catch ME
        warning('Could not bind %s agent automatically: %s', upper(alg), ME.message);
    end

    scoreEstimate = localEstimateFinalScore(statsHard, 30);
    algoScores(i) = scoreEstimate;
    algoFiles(i) = string(agentFile);

    bucket = struct();
    bucket.agentFile = agentFile;
    bucket.totalEpisodes = totalEpisodes;
    bucket.scoreEstimate = scoreEstimate;
    bucket.trainingStats = trainingStats;
    progressiveResults.(upper(char(alg))) = bucket;
end

[~, bestIdx] = max(algoScores);
bestAlg = algorithmOrder(bestIdx);
bestFile = char(algoFiles(bestIdx));

progressiveResults.bestAlgorithm = upper(char(bestAlg));
progressiveResults.bestAgentFile = bestFile;
progressiveResults.scoreBoard = table(cellstr(upper(algorithmOrder')),...
    algoScores', cellstr(algoFiles'), ...
    'VariableNames', {'Algorithm','FinalScoreEstimate','AgentFile'});

fprintf('\n=== Training summary ===\n');
disp(progressiveResults.scoreBoard);
fprintf('Selected best algorithm: %s\n', progressiveResults.bestAlgorithm);
fprintf('Selected best agent file: %s\n', progressiveResults.bestAgentFile);

Sbest = load(bestFile,'agent','trainingStats');
agent = Sbest.agent; %#ok<NASGU>
trainingStats = Sbest.trainingStats; %#ok<NASGU>
algorithm = bestAlg; %#ok<NASGU>

save(fullfile(thisDir,'trained_rl_agent.mat'), 'agent','trainingStats','algorithm');
results = progressiveResults; %#ok<NASGU>
save(fullfile(thisDir,'trained_rl_agent_progressive_summary.mat'), ...
    'results','episodePlan','algorithmOrder','showTrainingMonitor','doPostEvaluation');

fprintf('Saved compatibility file: %s\n', fullfile(thisDir,'trained_rl_agent.mat'));
fprintf('Saved summary file: %s\n', fullfile(thisDir,'trained_rl_agent_progressive_summary.mat'));

if doPostEvaluation
    fprintf('\nRunning post-training comparison with classical controllers...\n');
    generate_rl_scenario_data(bestFile);
    localRunScriptInBase(fullfile(thisDir,'compare_controllers.m'));
    if exist('create_graph_table_comparison.m','file') == 2
        localRunScriptInBase(fullfile(thisDir,'create_graph_table_comparison.m'));
    end
end

results = progressiveResults;

fprintf('\nProgressive RL training pipeline complete.\n');

end

function localRunScriptInBase(scriptFile)
if exist(scriptFile,'file') ~= 2
    error('Script file not found: %s', scriptFile);
end

scriptFileEscaped = strrep(scriptFile, '''', '''''');
evalin('base', sprintf('run(''%s'');', scriptFileEscaped));
end

function sacAgent = localLoadLatestSACAgent(thisDir)
%LOCALLOADLATESTSACAGENT Find the most recent SAC agent file and load it.
sacAgent = [];

candidateFiles = { ...
    fullfile(thisDir,'trained_rl_agent_sac_github_style.mat'), ...
    fullfile(thisDir,'trained_rl_agent_sac_progressive.mat'), ...
    fullfile(thisDir,'trained_rl_agent.mat')};

% Pick the most recently modified file among candidates
bestFile = '';
bestDate = 0;

for i = 1:numel(candidateFiles)
    f = candidateFiles{i};
    if exist(f,'file') ~= 2
        continue;
    end
    d = dir(f);
    if isempty(d), continue; end
    if d.datenum > bestDate
        bestDate = d.datenum;
        bestFile = f;
    end
end

if isempty(bestFile)
    fprintf('No existing SAC agent found for warm-start. Will train from scratch.\n');
    return;
end

try
    S = load(bestFile,'agent');
    if isfield(S,'agent')
        sacAgent = S.agent;
        fprintf('Loaded latest SAC warm-start agent from: %s\n', bestFile);
    end
catch ME
    warning('Could not load SAC warm-start agent: %s', ME.message);
end
end

function episodePlan = localNormalizeEpisodePlan(maxEpisodesConfig, algorithmOrder)
defaults = struct('ddpg',500,'td3',800,'sac',1000);
episodePlan = defaults;

if isnumeric(maxEpisodesConfig)
    if isscalar(maxEpisodesConfig)
        v = max(1, round(double(maxEpisodesConfig)));
        for i = 1:numel(algorithmOrder)
            episodePlan.(char(algorithmOrder(i))) = v;
        end
        return;
    end

    if numel(maxEpisodesConfig) ~= numel(algorithmOrder)
        error(['Numeric maxEpisodesConfig must be scalar or have one entry ', ...
               'per algorithm in algorithmOrder.']);
    end

    for i = 1:numel(algorithmOrder)
        v = max(1, round(double(maxEpisodesConfig(i))));
        episodePlan.(char(algorithmOrder(i))) = v;
    end
    return;
end

if ~isstruct(maxEpisodesConfig)
    error('maxEpisodesConfig must be numeric scalar/vector or struct.');
end

f = fieldnames(maxEpisodesConfig);
for i = 1:numel(f)
    k = lower(string(f{i}));
    if ~ismember(k, ["ddpg","td3","sac"])
        continue;
    end
    v = maxEpisodesConfig.(f{i});
    if ~(isnumeric(v) && isscalar(v) && isfinite(v))
        error('maxEpisodesConfig.%s must be a finite numeric scalar.', f{i});
    end
    episodePlan.(char(k)) = max(1, round(double(v)));
end

% Keep only requested algorithms in final plan usage; defaults remain harmless.
for i = 1:numel(algorithmOrder)
    k = char(algorithmOrder(i));
    if ~isfield(episodePlan,k)
        episodePlan.(k) = defaults.(k);
    end
end

end

function [episodesEasy, episodesHard] = localSplitEpisodes(totalEpisodes)
totalEpisodes = max(1, round(double(totalEpisodes)));

if totalEpisodes <= 2
    episodesEasy = 0;
    episodesHard = totalEpisodes;
    return;
end

episodesEasy = max(1, round(0.35*totalEpisodes));
episodesHard = max(1, totalEpisodes - episodesEasy);
end

function opts = localBuildTrainingOptions(maxEpisodes, plotMode, algorithm, stage)
maxStepsPerEpisode = 5000;

if nargin < 3 || isempty(algorithm)
    algorithm = "td3";
end
if nargin < 4 || isempty(stage)
    stage = "hard";
end

algorithm = lower(string(algorithm));
stage = lower(string(stage));

switch algorithm
    case "td3"
        if stage == "easy"
            maxStepsPerEpisode = 800;
        else
            maxStepsPerEpisode = 1200;
        end
    case "sac"
        if stage == "easy"
            maxStepsPerEpisode = 2000;
        else
            maxStepsPerEpisode = 3000;
        end
    otherwise
        maxStepsPerEpisode = 5000;
end

opts = rlTrainingOptions( ...
    'MaxEpisodes', maxEpisodes, ...
    'MaxStepsPerEpisode', maxStepsPerEpisode, ...
    'StopTrainingCriteria', 'EpisodeCount', ...
    'StopTrainingValue', maxEpisodes, ...
    'ScoreAveragingWindowLength', 20, ...
    'Verbose', false, ...
    'Plots', plotMode);
end

function agent = localBuildAgent(algorithm, obsInfo, actInfo)
initOpts = rlAgentInitializationOptions(NumHiddenUnit=256);

switch algorithm
    case "ddpg"
        try
            agent = rlDDPGAgent(obsInfo, actInfo, initOpts);
        catch
            agent = rlDDPGAgent(obsInfo, actInfo);
        end

    case "td3"
        try
            agent = rlTD3Agent(obsInfo, actInfo, initOpts);
        catch
            agent = rlTD3Agent(obsInfo, actInfo);
        end

    case "sac"
        try
            agent = rlSACAgent(obsInfo, actInfo, initOpts);
        catch ME1
            try
                agent = rlSACAgent(obsInfo, actInfo);
            catch ME2
                error(['Could not create SAC agent. MATLAB release/toolbox may not ', ...
                       'support this constructor.\nFirst error: %s\nSecond error: %s'], ...
                    ME1.message, ME2.message);
            end
        end

    otherwise
        error('Unsupported algorithm: %s', algorithm);
end
end

function agent = localConfigureAgent(agent, algorithm, Ts, actInfo)
% Common options
localTrySet(agent.AgentOptions, 'SampleTime', Ts);
localTrySet(agent.AgentOptions, 'DiscountFactor', 0.995);
localTrySet(agent.AgentOptions, 'ExperienceBufferLength', 1e6);
localTrySet(agent.AgentOptions, 'ResetExperienceBufferBeforeTraining', true);
localTrySet(agent.AgentOptions, 'NumWarmStartSteps', 100);
localTrySet(agent.AgentOptions, 'LearningFrequency', 4);
localTrySet(agent.AgentOptions, 'NumEpoch', 1);
localTrySet(agent.AgentOptions, 'MaxMiniBatchPerEpoch', 2);

actLower = -1;
actUpper = 1;
try
    if ~isempty(actInfo.LowerLimit)
        actLower = double(actInfo.LowerLimit(1));
    end
    if ~isempty(actInfo.UpperLimit)
        actUpper = double(actInfo.UpperLimit(1));
    end
catch
end

switch algorithm
    case "ddpg"
        % Match the proven GitHub-style DDPG defaults used in this project.
        localTrySet(agent.AgentOptions, 'MiniBatchSize', 128);
        localTrySet(agent.AgentOptions, 'TargetSmoothFactor', 1e-3);
        localTrySet(agent.AgentOptions, 'MaxMiniBatchPerEpoch', 1);

        if isprop(agent.AgentOptions, 'ExplorationModel')
            ex = agent.AgentOptions.ExplorationModel;
            localTrySet(ex,'Mean',0);
            localTrySet(ex,'StandardDeviation',0.15);
            localTrySet(ex,'StandardDeviationDecayRate',5e-5);
            localTrySet(ex,'StandardDeviationMin',0.005);
            localTrySet(ex,'LowerLimit',actLower);
            localTrySet(ex,'UpperLimit',actUpper);
        end

        if isprop(agent.AgentOptions,'ActorOptimizerOptions')
            localTrySet(agent.AgentOptions.ActorOptimizerOptions,'LearnRate',1e-4);
            localTrySet(agent.AgentOptions.ActorOptimizerOptions,'GradientThreshold',1);
        end
        if isprop(agent.AgentOptions,'CriticOptimizerOptions')
            localTrySet(agent.AgentOptions.CriticOptimizerOptions,'LearnRate',5e-4);
            localTrySet(agent.AgentOptions.CriticOptimizerOptions,'GradientThreshold',1);
        end

    case "td3"
        localTrySet(agent.AgentOptions, 'MiniBatchSize', 32);
        localTrySet(agent.AgentOptions, 'TargetSmoothFactor', 5e-3);
        localTrySet(agent.AgentOptions, 'PolicyUpdateFrequency', 2);
        localTrySet(agent.AgentOptions, 'TargetUpdateFrequency', 2);
        localTrySet(agent.AgentOptions, 'LearningFrequency', 16);
        localTrySet(agent.AgentOptions, 'MaxMiniBatchPerEpoch', 1);

        if isprop(agent.AgentOptions, 'ExplorationModel')
            ex = agent.AgentOptions.ExplorationModel;
            localTrySet(ex,'Mean',0);
            localTrySet(ex,'StandardDeviation',0.15);
            localTrySet(ex,'StandardDeviationDecayRate',5e-5);
            localTrySet(ex,'StandardDeviationMin',0.01);
            localTrySet(ex,'LowerLimit',actLower);
            localTrySet(ex,'UpperLimit',actUpper);
        end

        if isprop(agent.AgentOptions, 'TargetPolicySmoothModel')
            sm = agent.AgentOptions.TargetPolicySmoothModel;
            localTrySet(sm,'Mean',0);
            localTrySet(sm,'StandardDeviation',0.20);
            localTrySet(sm,'StandardDeviationDecayRate',1e-5);
            localTrySet(sm,'StandardDeviationMin',0.02);
            localTrySet(sm,'LowerLimit',-0.25);
            localTrySet(sm,'UpperLimit',0.25);
        end

        if isprop(agent.AgentOptions,'ActorOptimizerOptions')
            localTrySet(agent.AgentOptions.ActorOptimizerOptions,'LearnRate',1e-4);
            localTrySet(agent.AgentOptions.ActorOptimizerOptions,'GradientThreshold',1);
        end
        if isprop(agent.AgentOptions,'CriticOptimizerOptions')
            for i = 1:numel(agent.AgentOptions.CriticOptimizerOptions)
                localTrySet(agent.AgentOptions.CriticOptimizerOptions(i),'LearnRate',5e-4);
                localTrySet(agent.AgentOptions.CriticOptimizerOptions(i),'GradientThreshold',1);
            end
        end

    case "sac"
        localTrySet(agent.AgentOptions, 'MiniBatchSize', 128);
        localTrySet(agent.AgentOptions, 'TargetSmoothFactor', 5e-3);
        localTrySet(agent.AgentOptions, 'MaxMiniBatchPerEpoch', 2);

        if isprop(agent.AgentOptions,'ActorOptimizerOptions')
            localTrySet(agent.AgentOptions.ActorOptimizerOptions,'LearnRate',1e-4);
            localTrySet(agent.AgentOptions.ActorOptimizerOptions,'GradientThreshold',1);
        end
        if isprop(agent.AgentOptions,'CriticOptimizerOptions')
            for i = 1:numel(agent.AgentOptions.CriticOptimizerOptions)
                localTrySet(agent.AgentOptions.CriticOptimizerOptions(i),'LearnRate',3e-4);
                localTrySet(agent.AgentOptions.CriticOptimizerOptions(i),'GradientThreshold',1);
            end
        end

        % Entropy weight tuning options differ by MATLAB release.
        if isprop(agent.AgentOptions,'EntropyWeightOptions')
            localTrySet(agent.AgentOptions.EntropyWeightOptions,'LearnRate',3e-4);
        end
        if isprop(agent.AgentOptions,'EntropyWeight')
            localTrySet(agent.AgentOptions,'EntropyWeight',0.2);
        end

    otherwise
        error('Unsupported algorithm: %s', algorithm);
end
end

function localTrySet(obj, propertyName, propertyValue)
try
    if isprop(obj, propertyName)
        obj.(propertyName) = propertyValue;
    end
catch
    % Ignore properties unavailable in a given MATLAB release.
end
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
    if ismember('AverageReward', vn)
        rewards = double(stats.AverageReward(:));
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

function stats = localBuildChunkedStats(rewards, scoreWindow, maxChunks, episodesCompleted)
if nargin < 2 || isempty(scoreWindow)
    scoreWindow = 20;
end
if nargin < 3 || isempty(maxChunks)
    maxChunks = 2;
end
if nargin < 4 || isempty(episodesCompleted)
    episodesCompleted = numel(rewards);
end

rewards = double(rewards(:));
rewards = rewards(isfinite(rewards));

stats = struct();
stats.EpisodeReward = rewards;
if isempty(rewards)
    stats.AverageReward = [];
else
    stats.AverageReward = movmean(rewards, [scoreWindow-1 0]);
end
stats.MaxChunks = maxChunks;
stats.EpisodesCompleted = episodesCompleted;
stats.Mode = 'td3-two-chunk';
end

function [td3AgentOut, didWarmStart] = localWarmStartTD3FromDDPG(td3AgentIn, ddpgAgent, obsInfo, actInfo)
td3AgentOut = td3AgentIn;
didWarmStart = false;

if isempty(ddpgAgent)
    return;
end

% Preferred path: directly copy actor representation.
try
    ddpgActor = getActor(ddpgAgent);
    td3AgentOut = setActor(td3AgentOut, ddpgActor);
    didWarmStart = true;
    return;
catch
end

% Fallback path: rebuild actor from DDPG model if API differs.
try
    ddpgActor = getActor(ddpgAgent);
    actorModel = getModel(ddpgActor);
    actor = rlContinuousDeterministicActor(actorModel, obsInfo, actInfo, ...
        'ObservationInputNames','state');
    td3AgentOut = setActor(td3AgentOut, actor);
    didWarmStart = true;
catch
    didWarmStart = false;
end
end

function ddpgAgent = localLoadDDPGWarmStart(thisDir)
ddpgAgent = [];

candidateFiles = { ...
    fullfile(thisDir,'trained_rl_agent_github_style.mat'), ...
    fullfile(thisDir,'trained_rl_agent_ddpg_progressive.mat'), ...
    fullfile(thisDir,'trained_rl_agent.mat')};

for i = 1:numel(candidateFiles)
    f = candidateFiles{i};
    if exist(f,'file') ~= 2
        continue;
    end

    try
        S = load(f);
    catch
        continue;
    end

    if ~isfield(S,'agent')
        continue;
    end

    a = S.agent;
    cls = lower(class(a));
    isDDPGClass = contains(cls,'ddpg');

    isDDPGByFlag = false;
    if isfield(S,'algorithm')
        try
            isDDPGByFlag = lower(string(S.algorithm)) == "ddpg";
        catch
            isDDPGByFlag = false;
        end
    end

    if isDDPGClass || isDDPGByFlag
        ddpgAgent = a;
        return;
    end
end
end

function score = localEstimateFinalScore(stats, windowLength)
score = -inf;

if nargin < 2 || isempty(windowLength)
    windowLength = 30;
end

if isempty(stats)
    return;
end

% Table case
if istable(stats)
    vn = stats.Properties.VariableNames;
    if ismember('EpisodeReward', vn)
        r = stats.EpisodeReward;
    elseif ismember('AverageReward', vn)
        r = stats.AverageReward;
    else
        r = [];
    end
else
    % Struct / object-like case
    r = [];
    try
        r = stats.EpisodeReward;
    catch
    end
    if isempty(r)
        try
            r = stats.AverageReward;
        catch
        end
    end
end

if isempty(r)
    return;
end

r = double(r(:));
r = r(isfinite(r));
if isempty(r)
    return;
end

n = min(windowLength, numel(r));
score = mean(r(end-n+1:end));
end

function Ts = localGetSampleTime()
Ts = 1e-4;
try
    if evalin('base','exist(''Ts'',''var'')') == 1
        TsFromBase = evalin('base','Ts');
        if isnumeric(TsFromBase) && isscalar(TsFromBase) && isfinite(TsFromBase) && TsFromBase > 0
            Ts = double(TsFromBase);
        end
    end
catch
end
end

function in = localResetFcnEasy(in)
% Easier exploration regime.
in = setVariable(in,'w_ref', 60 + 50*rand);   % 60..110 rad/s
in = setVariable(in,'Tload', 0.0);            % no disturbance
end

function in = localResetFcnHard(in)
% Harder regime near practical operation with disturbances.
in = setVariable(in,'w_ref', 80 + 50*rand);       % 80..130 rad/s
in = setVariable(in,'Tload', 0.02*(2*rand - 1));  % +/- 0.02 N.m
end
