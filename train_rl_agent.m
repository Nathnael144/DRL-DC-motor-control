function [agent, trainingStats] = train_rl_agent(algorithm)
%TRAIN_RL_AGENT Train TD3 or DDPG agent for DC motor speed control.

if nargin < 1 || isempty(algorithm)
    algorithm = "td3";
end
algorithm = lower(string(algorithm));

thisDir = fileparts(mfilename('fullpath'));
if ~isempty(thisDir)
    addpath(thisDir);
end

hasRlToolbox = (exist('rlSimulinkEnv','file') == 2) && ...
               ((exist('rlNumericSpec','class') == 8) || (exist('rlNumericSpec','file') == 2));
if ~hasRlToolbox
    error(['train_rl_agent requires Reinforcement Learning Toolbox. ', ...
           'Install/enable it (Add-On Explorer), then rerun training.']);
end

envSetupPath = 'rl_env_setup.mat';
if exist(envSetupPath,'file') ~= 2 && ~isempty(thisDir)
    envSetupPath = fullfile(thisDir,'rl_env_setup.mat');
end

if exist(envSetupPath,'file') == 2
    S = load(envSetupPath,'env','obsInfo','actInfo');
    env = S.env;
    obsInfo = S.obsInfo;
    actInfo = S.actInfo;
else
    [env, obsInfo, actInfo] = setup_rl_environment();
end

initOpts = rlAgentInitializationOptions(NumHiddenUnit=128);

switch algorithm
    case "ddpg"
        agent = rlDDPGAgent(obsInfo, actInfo, initOpts);
    otherwise
        agent = rlTD3Agent(obsInfo, actInfo, initOpts);
        algorithm = "td3";
end

agent.AgentOptions.SampleTime = 1e-4;
agent.AgentOptions.DiscountFactor = 0.995;
agent.AgentOptions.MiniBatchSize = 256;
agent.AgentOptions.ExperienceBufferLength = 1e6;

trainOpts = rlTrainingOptions( ...
    'MaxEpisodes',1200, ...
    'MaxStepsPerEpisode',3000, ...
    'ScoreAveragingWindowLength',20, ...
    'StopTrainingCriteria','AverageReward', ...
    'StopTrainingValue',-10, ...
    'Verbose',false, ...
    'Plots','training-progress');

trainingStats = train(agent, env, trainOpts);

save('trained_rl_agent.mat','agent','trainingStats','algorithm');
fprintf('Training complete. Saved trained agent to trained_rl_agent.mat\n');

end
