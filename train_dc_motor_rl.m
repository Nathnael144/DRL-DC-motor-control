%% train_dc_motor_rl
% Reinforcement learning training script for dc_motor_rl Simulink environment.
% This script creates rlSimulinkEnv, builds DDPG/TD3 (and optionally SAC)
% agents, trains them, and saves agents and training statistics.
%
% IMPORTANT: Adjust the model and block names in the CONFIG section below
% to match your actual Simulink model.

%% CONFIG: model binding and global options
clear; clc; close all;

% Reproducibility
rng(0,'twister');

% --- Simulink model / environment binding (EDIT TO MATCH YOUR MODEL) ---
mdl      = 'dc_motor_rl';          % Simulink model name (without .slx)
agentBlk = 'dc_motor_rl/RL Agent'; % Path to RL Agent block in the model

% If the environment uses workspace parameters (e.g. reference profiles),
% define them here so they are available during training and evaluation.
refProfileType = "step";   % e.g. "step", "ramp", "profile1" (adapt to your model)
simTime        = 2.0;      % default simulation time per episode [s]

% --- Observation and action specifications (EDIT TO MATCH YOUR MODEL) ---
% These specs bypass the need for an existing agent object in the workspace.
% Set obsDim to the number of observation signals your RL Agent block uses
% (e.g. speed, current, etc.). Set action limits to the range expected by
% your plant input (e.g. normalized [-1 1] or actual voltage limits).
obsDim = 2; % example: [speed; current]

obsInfo = rlNumericSpec([obsDim 1], ...
    "Name","observations", ...
    "Description","DC motor observations (e.g. speed, current)");

actInfo = rlNumericSpec([1 1], ...
    "LowerLimit",-1, ...
    "UpperLimit", 1, ...
    "Name","action", ...
    "Description","Normalized control input (e.g. voltage command)");

% --- Centralized hyperparameters (tuning knobs) ---
hp = struct();

% Environment / training horizon
hp.env.maxStepsPerEpisode = 1000;   % must be consistent with Simulink step size * episode length
hp.env.discountFactor     = 0.99;   % used in agent options (not rlTrainingOptions)

% Shared training options
hp.training.maxEpisodes           = 500;
hp.training.stopOnAverageReward   = true;
hp.training.stopReward            = 500;  % target average reward for early stopping (adjust)
hp.training.averageWindowEpisodes = 20;
hp.training.plotTraining          = "training-progress";
hp.training.useParallel           = false;
hp.training.parallelWorkers       = 4;    % used only if useParallel = true
hp.training.saveAgentDirectory    = "savedAgents";

% DDPG hyperparameters
hp.ddpg.actorLayerSizes   = [400 300];
hp.ddpg.criticLayerSizes  = [400 300];
hp.ddpg.actorLearningRate = 1e-4;
hp.ddpg.criticLearningRate= 1e-3;
hp.ddpg.targetSmoothFactor= 1e-3;
hp.ddpg.explorationStd    = 0.2;
hp.ddpg.miniBatchSize     = 256;
hp.ddpg.replayBufferSize  = 1e6;

% TD3 hyperparameters (inspired by common TD3 implementations)
hp.td3.actorLayerSizes        = [400 300];
hp.td3.criticLayerSizes       = [400 300];
hp.td3.actorLearningRate      = 1e-4;
hp.td3.criticLearningRate     = 1e-3;
hp.td3.targetSmoothFactor     = 5e-3;
hp.td3.targetPolicyNoiseStd   = 0.2;
hp.td3.targetPolicyNoiseClip  = 0.5;
hp.td3.policyUpdateFrequency  = 2;       % delayed policy updates
hp.td3.explorationStd         = 0.1;
hp.td3.miniBatchSize          = 256;
hp.td3.replayBufferSize       = 1e6;

% SAC hyperparameters (optional)
hp.sac.enable             = false;   % set true to train SAC
hp.sac.actorLayerSizes    = [256 256];
hp.sac.criticLayerSizes   = [256 256];
hp.sac.actorLearningRate  = 3e-4;
hp.sac.criticLearningRate = 3e-4;
hp.sac.alphaLearningRate  = 3e-4;    % entropy temperature learning
hp.sac.targetSmoothFactor = 5e-3;
hp.sac.miniBatchSize      = 256;
hp.sac.replayBufferSize   = 1e6;
hp.sac.targetEntropyScale = 0.98;    % scale of -numActions target entropy

%% Environment creation: bind Simulink model via rlSimulinkEnv
fprintf("Loading Simulink model '%s'...\n", mdl);
load_system(mdl);

% Optionally configure model-level parameters for RL episodes here
% For example, if your model reads 'refProfileType' or 'simTime' from
% the base workspace, they are already defined above.

% Create the rlSimulinkEnv. The reward signal must already be defined
% inside the Simulink model (e.g. using an RL Agent block reward port).
% We pass obsInfo and actInfo explicitly so no pre-existing agent object
% (e.g. 'agentObj') is required in the workspace.
env = rlSimulinkEnv(mdl, agentBlk, obsInfo, actInfo);

fprintf("Environment created.\n");
disp("Observation spec:");
disp(obsInfo);
disp("Action spec:");
disp(actInfo);

%% Shared training options template
% These base options are cloned/modified for each algorithm.
trainOptsBase = rlTrainingOptions( ...
    "MaxEpisodes",                hp.training.maxEpisodes, ...
    "MaxStepsPerEpisode",         hp.env.maxStepsPerEpisode, ...
    "ScoreAveragingWindowLength", hp.training.averageWindowEpisodes, ...
    "StopTrainingCriteria",       "AverageReward", ...
    "StopTrainingValue",          hp.training.stopReward, ...
    "Verbose",                    true, ...
    "Plots",                      hp.training.plotTraining);

if hp.training.useParallel
    trainOptsBase.UseParallel = true;
    trainOptsBase.ParallelizationOptions.Mode = "async";
    trainOptsBase.ParallelizationOptions.StepsUntilDataIsSent = 32;
    trainOptsBase.ParallelizationOptions.DataToSendFromWorkers = "Experiences";
    trainOptsBase.ParallelizationOptions.WorkerRandomSeeds = -1;
end

if ~exist(hp.training.saveAgentDirectory,"dir")
    mkdir(hp.training.saveAgentDirectory);
end

%% DDPG: build agent and train
fprintf("\n=== Training DDPG agent ===\n");
ddpgAgent = createDDPGAgent(obsInfo, actInfo, hp);

ddpgTrainOpts = trainOptsBase;
ddpgTrainOpts.SaveAgentCriteria = "EpisodeReward";
ddpgTrainOpts.SaveAgentValue    = hp.training.stopReward;
ddpgTrainOpts.SaveAgentDirectory= fullfile(hp.training.saveAgentDirectory,"DDPG");

ddpgResults = train(ddpgAgent, env, ddpgTrainOpts);

ddpgAgentFile = fullfile(hp.training.saveAgentDirectory,"ddpgAgent_dc_motor.mat");
save(ddpgAgentFile,"ddpgAgent","ddpgResults","hp");
fprintf("Saved DDPG agent to %s\n", ddpgAgentFile);

% Save DDPG training reward plot if available
try
    fig = findall(groot,"Type","Figure","Tag","rl_training_plots");
    if ~isempty(fig)
        saveas(fig, fullfile(hp.training.saveAgentDirectory,"ddpg_training.png"));
    end
catch ME
    warning("Could not save DDPG training plot: %s", ME.message);
end

%% TD3: build agent and train
fprintf("\n=== Training TD3 agent ===\n");
% Optionally initialize TD3 actor from trained DDPG actor
td3Agent = createTD3Agent(obsInfo, actInfo, hp, ddpgAgent);

td3TrainOpts = trainOptsBase;
td3TrainOpts.SaveAgentCriteria = "EpisodeReward";
td3TrainOpts.SaveAgentValue    = hp.training.stopReward;
td3TrainOpts.SaveAgentDirectory= fullfile(hp.training.saveAgentDirectory,"TD3");

td3Results = train(td3Agent, env, td3TrainOpts);

td3AgentFile = fullfile(hp.training.saveAgentDirectory,"td3Agent_dc_motor.mat");
save(td3AgentFile,"td3Agent","td3Results","hp");
fprintf("Saved TD3 agent to %s\n", td3AgentFile);

try
    fig = findall(groot,"Type","Figure","Tag","rl_training_plots");
    if ~isempty(fig)
        saveas(fig, fullfile(hp.training.saveAgentDirectory,"td3_training.png"));
    end
catch ME
    warning("Could not save TD3 training plot: %s", ME.message);
end

%% SAC (optional): build agent and train
if hp.sac.enable
    fprintf("\n=== Training SAC agent ===\n");
    sacAgent = createSACAgent(obsInfo, actInfo, hp);

    sacTrainOpts = trainOptsBase;
    sacTrainOpts.SaveAgentCriteria = "EpisodeReward";
    sacTrainOpts.SaveAgentValue    = hp.training.stopReward;
    sacTrainOpts.SaveAgentDirectory= fullfile(hp.training.saveAgentDirectory,"SAC");

    sacResults = train(sacAgent, env, sacTrainOpts);

    sacAgentFile = fullfile(hp.training.saveAgentDirectory,"sacAgent_dc_motor.mat");
    save(sacAgentFile,"sacAgent","sacResults","hp");
    fprintf("Saved SAC agent to %s\n", sacAgentFile);

    try
        fig = findall(groot,"Type","Figure","Tag","rl_training_plots");
        if ~isempty(fig)
            saveas(fig, fullfile(hp.training.saveAgentDirectory,"sac_training.png"));
        end
    catch ME
        warning("Could not save SAC training plot: %s", ME.message);
    end
else
    fprintf("\nSAC training is disabled (hp.sac.enable = false).\n");
end

fprintf("\nTraining script finished.\n");

%% Local function definitions

function agent = createDDPGAgent(obsInfo, actInfo, hp)
%CREATEDDPGAGENT Build a DDPG agent for continuous control.

numObs  = obsInfo.Dimension(1);
numAct  = numel(actInfo);
actMax  = actInfo.UpperLimit;

% Actor network: state -> action
statePath = featureInputLayer(numObs, ...
    "Normalization","none", ...
    "Name","state");

actorLayers = [
    fullyConnectedLayer(hp.ddpg.actorLayerSizes(1),"Name","fc1")
    reluLayer("Name","relu1")
    fullyConnectedLayer(hp.ddpg.actorLayerSizes(2),"Name","fc2")
    reluLayer("Name","relu2")
    fullyConnectedLayer(numAct,"Name","fcOut")
    tanhLayer("Name","tanhOut")];

actorNet = layerGraph(statePath);
actorNet = addLayers(actorNet, actorLayers);
actorNet = connectLayers(actorNet,"state","fc1");

actorOpts = rlRepresentationOptions( ...
    "LearnRate",hp.ddpg.actorLearningRate, ...
    "GradientThreshold",1);

actor = rlContinuousDeterministicActor(actorNet, obsInfo, actInfo, ...
    "ObservationInputNames","state", ...
    "Options", actorOpts);

% Critic network: state, action -> Q-value
statePathC = featureInputLayer(numObs, ...
    "Normalization","none", ...
    "Name","state");
actionPathC = featureInputLayer(numAct, ...
    "Normalization","none", ...
    "Name","action");

criticCommon = [
    concatenationLayer(1,2,"Name","concat")
    fullyConnectedLayer(hp.ddpg.criticLayerSizes(1),"Name","cfc1")
    reluLayer("Name","crelu1")
    fullyConnectedLayer(hp.ddpg.criticLayerSizes(2),"Name","cfc2")
    reluLayer("Name","crelu2")
    fullyConnectedLayer(1,"Name","qOut")];

criticNet = layerGraph(statePathC);
criticNet = addLayers(criticNet, actionPathC);
criticNet = addLayers(criticNet, criticCommon);

criticNet = connectLayers(criticNet,"state","concat/in1");
criticNet = connectLayers(criticNet,"action","concat/in2");

criticOpts = rlRepresentationOptions( ...
    "LearnRate",hp.ddpg.criticLearningRate, ...
    "GradientThreshold",1);

critic = rlQValueFunction(criticNet, obsInfo, actInfo, ...
    "ObservationInputNames","state", ...
    "ActionInputNames","action", ...
    criticOpts);

agentOpts = rlDDPGAgentOptions( ...
    "SampleTime",                 1, ... % overwritten by env step time
    "DiscountFactor",             hp.env.discountFactor, ...
    "TargetSmoothFactor",         hp.ddpg.targetSmoothFactor, ...
    "ExperienceBufferLength",     hp.ddpg.replayBufferSize, ...
    "MiniBatchSize",              hp.ddpg.miniBatchSize);

% Ornstein-Uhlenbeck exploration noise (or simple Gaussian)
agentOpts.NoiseOptions.StandardDeviation = hp.ddpg.explorationStd*actMax;
agentOpts.NoiseOptions.StandardDeviationDecayRate = 1e-5;

agent = rlDDPGAgent(actor, critic, agentOpts);
end

function agent = createTD3Agent(obsInfo, actInfo, hp, ddpgAgent)
%CREATETD3AGENT Build a TD3 agent with twin critics.

numObs  = obsInfo.Dimension(1);
numAct  = numel(actInfo);
actMax  = actInfo.UpperLimit;

% Actor network (optionally initialized from DDPG actor if provided)
if nargin >= 4 && ~isempty(ddpgAgent)
    baseActor = getActor(ddpgAgent);
    actorNet  = getModel(baseActor);
else
    statePath = featureInputLayer(numObs, ...
        "Normalization","none", ...
        "Name","state");
    actorLayers = [
        fullyConnectedLayer(hp.td3.actorLayerSizes(1),"Name","fc1")
        reluLayer("Name","relu1")
        fullyConnectedLayer(hp.td3.actorLayerSizes(2),"Name","fc2")
        reluLayer("Name","relu2")
        fullyConnectedLayer(numAct,"Name","fcOut")
        tanhLayer("Name","tanhOut")];
    actorNet = layerGraph(statePath);
    actorNet = addLayers(actorNet, actorLayers);
    actorNet = connectLayers(actorNet,"state","fc1");
end

actorOpts = rlRepresentationOptions( ...
    "LearnRate",hp.td3.actorLearningRate, ...
    "GradientThreshold",1);

actor = rlContinuousDeterministicActor(actorNet, obsInfo, actInfo, ...
    "ObservationInputNames","state", ...
    "Options", actorOpts);

% Two critic networks (twin critics)
critic1 = buildTD3Critic(obsInfo, actInfo, hp.td3.criticLayerSizes, hp.td3.criticLearningRate, "1");
critic2 = buildTD3Critic(obsInfo, actInfo, hp.td3.criticLayerSizes, hp.td3.criticLearningRate, "2");

agentOpts = rlTD3AgentOptions( ...
    "SampleTime",                 1, ...
    "DiscountFactor",             hp.env.discountFactor, ...
    "TargetSmoothFactor",         hp.td3.targetSmoothFactor, ...
    "ExperienceBufferLength",     hp.td3.replayBufferSize, ...
    "MiniBatchSize",              hp.td3.miniBatchSize, ...
    "PolicyUpdateFrequency",      hp.td3.policyUpdateFrequency);

% Exploration (Gaussian on actions)
agentOpts.ExplorationModel.StandardDeviation = hp.td3.explorationStd * actMax;
agentOpts.ExplorationModel.StandardDeviationDecayRate = 1e-5;

% Target policy smoothing
agentOpts.TargetPolicySmoothModel.StandardDeviation = hp.td3.targetPolicyNoiseStd * actMax;
agentOpts.TargetPolicySmoothModel.Saturation = hp.td3.targetPolicyNoiseClip * actMax;

agent = rlTD3Agent(actor, [critic1 critic2], agentOpts);
end

function critic = buildTD3Critic(obsInfo, actInfo, layerSizes, learnRate, suffix)
%BUILDTD3CRITIC Helper to construct a TD3 critic network.

numObs = obsInfo.Dimension(1);
numAct = numel(actInfo);

statePath = featureInputLayer(numObs, ...
    "Normalization","none", ...
    "Name","state"+suffix);
actionPath = featureInputLayer(numAct, ...
    "Normalization","none", ...
    "Name","action"+suffix);

commonLayers = [
    concatenationLayer(1,2,"Name","concat"+suffix)
    fullyConnectedLayer(layerSizes(1),"Name","cfc1"+suffix)
    reluLayer("Name","crelu1"+suffix)
    fullyConnectedLayer(layerSizes(2),"Name","cfc2"+suffix)
    reluLayer("Name","crelu2"+suffix)
    fullyConnectedLayer(1,"Name","qOut"+suffix)];

criticNet = layerGraph(statePath);
criticNet = addLayers(criticNet, actionPath);
criticNet = addLayers(criticNet, commonLayers);

criticNet = connectLayers(criticNet,"state"+suffix,"concat"+suffix+"/in1");
criticNet = connectLayers(criticNet,"action"+suffix,"concat"+suffix+"/in2");

criticOpts = rlRepresentationOptions( ...
    "LearnRate",learnRate, ...
    "GradientThreshold",1);

critic = rlQValueFunction(criticNet, obsInfo, actInfo, ...
    "ObservationInputNames","state"+suffix, ...
    "ActionInputNames","action"+suffix, ...
    criticOpts);
end

function agent = createSACAgent(obsInfo, actInfo, hp)
%CREATESACAGENT Build a SAC agent for continuous control (Gaussian policy).

numObs = obsInfo.Dimension(1);
numAct = numel(actInfo);

% Actor network (Gaussian policy: outputs mean and std)
statePath = featureInputLayer(numObs, ...
    "Normalization","none", ...
    "Name","state");

actorLayers = [
    fullyConnectedLayer(hp.sac.actorLayerSizes(1),"Name","fc1")
    reluLayer("Name","relu1")
    fullyConnectedLayer(hp.sac.actorLayerSizes(2),"Name","fc2")
    reluLayer("Name","relu2")];

meanLayer = fullyConnectedLayer(numAct,"Name","mean");
stdLayer  = fullyConnectedLayer(numAct,"Name","std");

actorNet = layerGraph(statePath);
actorNet = addLayers(actorNet, actorLayers);
actorNet = addLayers(actorNet, meanLayer);
actorNet = addLayers(actorNet, stdLayer);

actorNet = connectLayers(actorNet,"state","fc1");
actorNet = connectLayers(actorNet,"relu2","mean");
actorNet = connectLayers(actorNet,"relu2","std");

actorOpts = rlRepresentationOptions( ...
    "LearnRate",hp.sac.actorLearningRate, ...
    "GradientThreshold",1);

actor = rlContinuousGaussianActor(actorNet, obsInfo, actInfo, ...
    "ObservationInputNames","state", ...
    "Options", actorOpts);

% Two Q-value critics
critic1 = buildSACCritic(obsInfo, actInfo, hp.sac.criticLayerSizes, hp.sac.criticLearningRate, "1");
critic2 = buildSACCritic(obsInfo, actInfo, hp.sac.criticLayerSizes, hp.sac.criticLearningRate, "2");

% Value function (V-network)
valueNet = [
    featureInputLayer(numObs,"Normalization","none","Name","stateV")
    fullyConnectedLayer(hp.sac.criticLayerSizes(1),"Name","vfc1")
    reluLayer("Name","vrelu1")
    fullyConnectedLayer(hp.sac.criticLayerSizes(2),"Name","vfc2")
    reluLayer("Name","vrelu2")
    fullyConnectedLayer(1,"Name","vOut")];

valueOpts = rlRepresentationOptions( ...
    "LearnRate",hp.sac.criticLearningRate, ...
    "GradientThreshold",1);

value = rlValueFunction(valueNet, obsInfo, ...
    "ObservationInputNames","stateV", ...
    valueOpts);

% Target entropy: -dim(A) scaled
targetEntropy = -hp.sac.targetEntropyScale * numAct;

agentOpts = rlSACAgentOptions( ...
    "SampleTime",                 1, ...
    "ExperienceBufferLength",     hp.sac.replayBufferSize, ...
    "MiniBatchSize",              hp.sac.miniBatchSize, ...
    "TargetEntropy",              targetEntropy, ...
    "AlphaLearnRate",             hp.sac.alphaLearningRate, ...
    "DiscountFactor",             hp.env.discountFactor, ...
    "TargetSmoothFactor",         hp.sac.targetSmoothFactor);

agent = rlSACAgent(actor, [critic1 critic2], value, agentOpts);
end

function critic = buildSACCritic(obsInfo, actInfo, layerSizes, learnRate, suffix)
%BUILDSACCRITIC Helper to construct a SAC critic network.

numObs = obsInfo.Dimension(1);
numAct = numel(actInfo);

statePath = featureInputLayer(numObs, ...
    "Normalization","none", ...
    "Name","stateS"+suffix);
actionPath = featureInputLayer(numAct, ...
    "Normalization","none", ...
    "Name","actionS"+suffix);

commonLayers = [
    concatenationLayer(1,2,"Name","concatS"+suffix)
    fullyConnectedLayer(layerSizes(1),"Name","cfc1S"+suffix)
    reluLayer("Name","crelu1S"+suffix)
    fullyConnectedLayer(layerSizes(2),"Name","cfc2S"+suffix)
    reluLayer("Name","crelu2S"+suffix)
    fullyConnectedLayer(1,"Name","qOutS"+suffix)];

criticNet = layerGraph(statePath);
criticNet = addLayers(criticNet, actionPath);
criticNet = addLayers(criticNet, commonLayers);

criticNet = connectLayers(criticNet,"stateS"+suffix,"concatS"+suffix+"/in1");
criticNet = connectLayers(criticNet,"actionS"+suffix,"concatS"+suffix+"/in2");

criticOpts = rlRepresentationOptions( ...
    "LearnRate",learnRate, ...
    "GradientThreshold",1);

critic = rlQValueFunction(criticNet, obsInfo, actInfo, ...
    "ObservationInputNames","stateS"+suffix, ...
    "ActionInputNames","actionS"+suffix, ...
    criticOpts);
end

