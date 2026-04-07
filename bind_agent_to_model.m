function agentObj = bind_agent_to_model(modelName, agentFile)
%BIND_AGENT_TO_MODEL Load an agent and bind it to Simulink RL Agent block.
%   agentObj = bind_agent_to_model() tries to load a trained agent from:
%     1) trained_rl_agent_tuned.mat
%     2) trained_rl_agent.mat
%   and assigns it to base workspace variable `agentObj`.
%
%   agentObj = bind_agent_to_model(modelName, agentFile)
%   explicitly sets model name and MAT-file path.

if nargin < 1 || isempty(modelName)
    modelName = 'dc_motor_rl';
end

thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir)
    thisDir = pwd;
end

if nargin < 2 || isempty(agentFile)
    candidates = { ...
        fullfile(thisDir,'trained_rl_agent_tuned.mat'), ...
        fullfile(thisDir,'trained_rl_agent.mat')};

    agentFile = '';
    for i = 1:numel(candidates)
        if exist(candidates{i},'file') == 2
            agentFile = candidates{i};
            break;
        end
    end

    if isempty(agentFile)
        error(['No trained agent MAT file found. Expected one of:\n', ...
               '  %s\n  %s\n', ...
               'Train first, then rerun this function.'], candidates{1}, candidates{2});
    end
end

if exist(agentFile,'file') ~= 2
    error('Agent file not found: %s', agentFile);
end

S = load(agentFile);
if isfield(S,'agent')
    agentObj = S.agent;
elseif isfield(S,'agentObj')
    agentObj = S.agentObj;
else
    error('MAT-file %s has no variable named "agent" or "agentObj".', agentFile);
end

assignin('base','agentObj',agentObj);

% Ensure model is loaded and RL Agent block points to agentObj.
if ~bdIsLoaded(modelName)
    modelPath = which([modelName '.slx']);
    if ~isempty(modelPath)
        load_system(modelPath);
    end
end

agentBlk = [modelName '/RL Agent'];
if getSimulinkBlockHandle(agentBlk) ~= -1
    try
        set_param(agentBlk,'Agent','agentObj');
    catch ME
        warning('Could not set Agent parameter on %s: %s', agentBlk, ME.message);
    end
end

fprintf('Bound base workspace variable agentObj from: %s\n', agentFile);

end
