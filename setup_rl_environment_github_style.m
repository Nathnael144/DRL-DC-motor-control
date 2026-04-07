function [env, obsInfo, actInfo] = setup_rl_environment_github_style(modelName)
%SETUP_RL_ENVIRONMENT_GITHUB_STYLE
% Create env with full observation and bipolar action:
%   obs = [integral_error; error; d_error; speed]  (4x1)
%   action in [-1,1] then model maps to [-Vmax,Vmax]

if nargin < 1 || isempty(modelName)
    modelName = 'dc_motor_rl';
end

if ~localHasRlToolbox()
    error('Reinforcement Learning Toolbox is required.');
end

thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir)
    thisDir = pwd;
end
addpath(thisDir);

[modelFile, isLoaded] = localResolveModelFile(modelName, thisDir);
if ~isLoaded
    if isempty(modelFile)
        error('Model file not found for %s.slx (checked project/cwd/path).', modelName);
    end
    load_system(modelFile);
else
    if ~isempty(modelFile)
        fprintf('Using loaded model: %s\n', modelFile);
    else
        fprintf('Using loaded model: %s\n', modelName);
    end
end

% Apply profile before creating env specs
configure_rl_training_profile_github_style(modelName);

agentBlk = [modelName '/RL Agent'];
if getSimulinkBlockHandle(agentBlk) == -1
    error('Missing RL Agent block: %s', agentBlk);
end

obsInfo = rlNumericSpec([4 1], ...
    'LowerLimit',-inf(4,1), ...
    'UpperLimit', inf(4,1));
obsInfo.Name = 'obs_github_style';

actInfo = rlNumericSpec([1 1], ...
    'LowerLimit',-1, ...
    'UpperLimit',1);
actInfo.Name = 'act_github_style';

env = rlSimulinkEnv(modelName, agentBlk, obsInfo, actInfo);
env.ResetFcn = @localResetFcn;

% Workspace variables
assignin('base','motorParams',dc_motor_params());
assignin('base','w_ref',100);
assignin('base','Tload',0);
assignin('base','Vmax',24);
assignin('base','Imax',8);
assignin('base','Ts',1e-4);

% Bind existing agent if present (optional)
try
    bind_agent_to_model(modelName);
catch
end

setupFile = fullfile(thisDir,'rl_env_setup_github_style.mat');
save(setupFile,'env','obsInfo','actInfo','modelName');
fprintf('GitHub-style env configured and saved to %s\n', setupFile);

end

function [modelFile, isLoaded] = localResolveModelFile(modelName, thisDir)
isLoaded = bdIsLoaded(modelName);
modelFile = '';

if isLoaded
    try
        loadedFile = get_param(modelName,'FileName');
        if ~isempty(loadedFile)
            modelFile = loadedFile;
            return;
        end
    catch
    end
end

candidate1 = fullfile(thisDir, [modelName '.slx']);
if exist(candidate1,'file') == 2
    modelFile = candidate1;
    return;
end

candidate2 = fullfile(pwd, [modelName '.slx']);
if exist(candidate2,'file') == 2
    modelFile = candidate2;
    return;
end

whichResult = which([modelName '.slx']);
if ~isempty(whichResult)
    modelFile = whichResult;
end
end

function in = localResetFcn(in)
% Broad reset covering evaluation operating points (w_ref=100..120).
in = setVariable(in,'w_ref', 60 + 80*rand);       % 60..140 rad/s
in = setVariable(in,'Tload', 0.015*(2*rand - 1)); % +/-0.015 N.m
end

function tf = localHasRlToolbox()
tf = (exist('rlSimulinkEnv','file') == 2) && ...
     ((exist('rlNumericSpec','class') == 8) || (exist('rlNumericSpec','file') == 2));
end
