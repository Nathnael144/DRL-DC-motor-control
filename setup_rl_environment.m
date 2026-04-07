function [env, obsInfo, actInfo] = setup_rl_environment(modelName)
%SETUP_RL_ENVIRONMENT Configure RL environment from Simulink model.
% Expected in the model:
%   1) RL Agent block named exactly: "RL Agent"
%   2) Observation vector to agent: [e; de; ie; omega]
%   3) Agent action in [-1,1], scaled by Vmax before motor voltage input

if nargin < 1 || isempty(modelName)
    modelName = 'dc_motor_rl';
end

if ~localHasRlToolbox()
    error(['setup_rl_environment requires Reinforcement Learning Toolbox. ', ...
           'Install/enable it (Add-On Explorer), then rerun this function.']);
end

% Try to ensure sibling helper files in this project are visible on path.
thisDir = fileparts(mfilename('fullpath'));
if ~isempty(thisDir) && exist(fullfile(thisDir,'dc_motor_params.m'),'file') == 2
    if isempty(which('dc_motor_params'))
        addpath(thisDir);
    end
end
if isempty(thisDir)
    thisDir = pwd;
end

% Resolve model file robustly: loaded model -> project file -> cwd -> path.
[modelFile, isLoaded] = localResolveModelFile(modelName, thisDir);

if isempty(modelFile) && ~isLoaded
    requestedFile = fullfile(thisDir, [modelName '.slx']);
    fprintf('Model %s not found on disk/path. Auto-creating it now...\n', requestedFile);
    create_dc_motor_rl_simulink_model(modelName, true);
    [modelFile, isLoaded] = localResolveModelFile(modelName, thisDir);
end

if ~isLoaded
    if isempty(modelFile)
        error('Unable to locate %s.slx even after auto-create.', modelName);
    end
    load_system(modelFile);
else
    % Model is already loaded; trust active loaded instance.
    if ~isempty(modelFile)
        fprintf('Using loaded model: %s\n', modelFile);
    else
        fprintf('Using loaded model: %s\n', modelName);
    end
end

agentBlk = [modelName '/RL Agent'];
if isempty(find_system(modelName,'SearchDepth',1,'Name','RL Agent'))
    error(['Required block "' agentBlk '" is missing. ' ...
           'Add Reinforcement Learning Toolbox RL Agent block first.']);
end

% If model currently has a placeholder subsystem, rebuild with true RL Agent.
if localIsPlaceholderAgent(agentBlk)
    warning(['Detected placeholder "RL Agent" block in %s. Rebuilding ', ...
             'model with real RL Agent block...'], modelName);
    create_dc_motor_rl_simulink_model(modelName, true);
    if ~bdIsLoaded(modelName)
        load_system(modelName);
    end
end

obsInfo = rlNumericSpec([4 1], ...
    'LowerLimit', -inf(4,1), ...
    'UpperLimit',  inf(4,1));
obsInfo.Name = 'observations';

actInfo = rlNumericSpec([1 1], ...
    'LowerLimit', -1, ...
    'UpperLimit',  1);
actInfo.Name = 'action';

try
    env = rlSimulinkEnv(modelName, agentBlk, obsInfo, actInfo);
catch ME
    % One recovery attempt: rebuild model forcing real RL Agent block.
    warning('First rlSimulinkEnv attempt failed: %s', ME.message);
    create_dc_motor_rl_simulink_model(modelName, true);
    if ~bdIsLoaded(modelName)
        load_system(modelName);
    end
    try
        env = rlSimulinkEnv(modelName, agentBlk, obsInfo, actInfo);
    catch ME2
        error(['Failed to create rlSimulinkEnv after rebuilding model.\n', ...
               'First error: %s\nSecond error: %s\n', ...
               'Check that block "' agentBlk '" is an RL Agent block ', ...
               '(not a plain Subsystem), then retry.'], ...
               ME.message, ME2.message);
    end
end
env.ResetFcn = @localResetFcn;

% Base workspace variables commonly used by the model
assignin('base','motorParams',localGetMotorParams());
assignin('base','w_ref',100);
assignin('base','Tload',0);
assignin('base','Vmax',24);
assignin('base','Imax',8);
assignin('base','Ts',1e-4);

% Ensure RL Agent block variable exists for interactive Simulink runs.
try
    bind_agent_to_model(modelName);
catch ME
    warning(['Could not bind agentObj automatically (%s). ', ...
             'You can still train via train(...), but for manual model run ', ...
             'load a trained agent using bind_agent_to_model.'], ME.message);
end

envSetupFile = fullfile(thisDir,'rl_env_setup.mat');
save(envSetupFile,'env','obsInfo','actInfo','modelName');
fprintf('RL environment configured and saved to %s\n', envSetupFile);

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
        % continue to disk search
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
    return;
end
end

function p = localGetMotorParams()
% Try project helper first; fallback to built-in defaults if unavailable.
p = [];

if exist('dc_motor_params','file') == 2
    try
        p = dc_motor_params();
    catch ME
        warning('Failed to evaluate dc_motor_params(): %s. Using built-in defaults instead.', ME.message);
    end
end

if isempty(p) || ~isstruct(p)
    warning(['dc_motor_params.m not found on current MATLAB path. ', ...
             'Using built-in motor parameters from your provided table.']);

    p.La = 0.58e-3;
    p.Ra = 2.59;
    p.J  = 5.69e-4;
    p.Bm = 1e-6;
    p.Kt = 28.6e-3;
    p.Ke = p.Kt;

    p.A = [-p.Ra/p.La, -p.Ke/p.La;
            p.Kt/p.J,  -p.Bm/p.J];
    p.B = [1/p.La; 0];
    p.C = [0 1];
    p.D = 0;
end
end

function tf = localHasRlToolbox()
tf = (exist('rlSimulinkEnv','file') == 2) && ...
    ((exist('rlNumericSpec','class') == 8) || (exist('rlNumericSpec','file') == 2));
end

function tf = localIsPlaceholderAgent(agentBlk)
tf = false;
if getSimulinkBlockHandle(agentBlk) == -1
    return;
end

blkType = '';
maskType = '';
try
    blkType = get_param(agentBlk,'BlockType');
catch
end
try
    maskType = get_param(agentBlk,'MaskType');
catch
end

% Placeholder generated by this project is a plain SubSystem with no RL mask.
if strcmpi(blkType,'SubSystem') && ...
   ~(contains(lower(maskType),'rl') && contains(lower(maskType),'agent'))
    tf = true;
end
end

function in = localResetFcn(in)
% Randomized initial conditions per episode
in = setVariable(in,'w_ref', 50 + 150*rand);
in = setVariable(in,'Tload', 0.02*(2*rand - 1));
end
