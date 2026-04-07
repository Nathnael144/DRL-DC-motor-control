function create_dc_motor_rl_simulink_model(modelName, requireRealAgent)
%CREATE_DC_MOTOR_RL_SIMULINK_MODEL Build a fully wired RL Simulink model.
%   create_dc_motor_rl_simulink_model() creates dc_motor_rl.slx with:
%   - RL Agent block (observation, reward, isDone -> action)
%   - Motor plant (state-space with input voltage + load torque)
%   - Observation vector: [e; de; ie; omega]
%   - Reward: -(5e^2 + 0.1de^2 + 0.01u^2 + 50*max(0,|ia|-Imax)^2)
%   - isDone: |ia| > 1.5*Imax
%
%   Required toolbox: Reinforcement Learning Toolbox.

if nargin < 1 || isempty(modelName)
    modelName = 'dc_motor_rl';
end
if nargin < 2 || isempty(requireRealAgent)
    requireRealAgent = false;
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
modelFile = fullfile(thisDir, [modelName '.slx']);

% Nominal parameters in base workspace for block expressions.
motorParams = localGetMotorParams();
assignin('base','motorParams',motorParams);
if evalin('base','exist(''w_ref'',''var'')') == 0
    assignin('base','w_ref',100);
end
if evalin('base','exist(''Tload'',''var'')') == 0
    assignin('base','Tload',0);
end
if evalin('base','exist(''Vmax'',''var'')') == 0
    assignin('base','Vmax',24);
end
if evalin('base','exist(''Imax'',''var'')') == 0
    assignin('base','Imax',8);
end
if evalin('base','exist(''Ts'',''var'')') == 0
    assignin('base','Ts',1e-4);
end

% Resolve RL Agent block path.
[rlAgentLibPath, hasRlAgentBlock, rlDiag] = localFindRlAgentBlockPath();
if requireRealAgent && ~hasRlAgentBlock
    error(['Reinforcement Learning Toolbox functions are available, but ', ...
           'the Simulink RL Agent library block could not be located.\n', ...
           'Diagnostics:\n%s\n', ...
           'Try: restart MATLAB, run "ver", then rerun this script. ', ...
           'For R2025b the expected block is "rllib/RL Agent".'], rlDiag);
end

if bdIsLoaded(modelName)
    close_system(modelName,0);
end

if exist(modelFile,'file') == 2
    delete(modelFile);
end

new_system(modelName);
open_system(modelName);

set_param(modelName, ...
    'SolverType','Fixed-step', ...
    'Solver','ode4', ...
    'FixedStep','Ts', ...
    'StopTime','0.5');

% ------------------------- Sources -------------------------
add_block('simulink/Sources/Constant', [modelName '/w_ref_const'], ...
    'Position',[40 55 110 85], 'Value','w_ref');
add_block('simulink/Sources/Constant', [modelName '/Tload_const'], ...
    'Position',[40 265 110 295], 'Value','Tload');
add_block('simulink/Sources/Constant', [modelName '/done_limit'], ...
    'Position',[980 440 1060 470], 'Value','1.5*Imax');

% ------------------- Error & observations ------------------
add_block('simulink/Math Operations/Sum', [modelName '/error_sum'], ...
    'Position',[170 55 205 95], 'Inputs','+-');
add_block('simulink/Continuous/Derivative', [modelName '/d_error'], ...
    'Position',[250 35 320 65]);
add_block('simulink/Continuous/Integrator', [modelName '/i_error'], ...
    'Position',[250 90 320 120], 'InitialCondition','0');
add_block('simulink/Signal Routing/Mux', [modelName '/obs_mux'], ...
    'Position',[360 40 365 135], 'Inputs','4');

% ------------------------- Agent ---------------------------
if hasRlAgentBlock
    created = false;
    try
        % Canonical path in modern releases (including R2025b).
        add_block('rllib/RL Agent', [modelName '/RL Agent'], ...
            'Position',[430 50 560 150]);
        created = true;
    catch
    end

    if ~created
        add_block(rlAgentLibPath, [modelName '/RL Agent'], ...
            'Position',[430 50 560 150]);
    end
else
    localAddPlaceholderAgent(modelName, [430 50 560 150]);
    warning(['Reinforcement Learning Toolbox not detected. Created a ', ...
             'placeholder "RL Agent" subsystem that outputs zero action.']);
end

% ---------------------- Action path ------------------------
add_block('simulink/Math Operations/Gain', [modelName '/action_scale'], ...
    'Position',[610 88 690 122], 'Gain','Vmax');
add_block('simulink/Discontinuities/Saturation', [modelName '/voltage_sat'], ...
    'Position',[740 88 820 122], 'UpperLimit','Vmax', 'LowerLimit','-Vmax');
add_block('simulink/Signal Routing/Mux', [modelName '/plant_in_mux'], ...
    'Position',[860 160 865 260], 'Inputs','2');

% ------------------------ Plant ----------------------------
add_block('simulink/Continuous/State-Space', [modelName '/motor_plant'], ...
    'Position',[910 160 1090 260], ...
    'A','motorParams.A', ...
    'B','[motorParams.B [0; -1/motorParams.J]]', ...
    'C','eye(2)', ...
    'D','zeros(2,2)', ...
    'X0','[0;0]');
add_block('simulink/Signal Routing/Demux', [modelName '/state_demux'], ...
    'Position',[1140 180 1145 250], 'Outputs','2');

% -------------------- Reward and done ----------------------
add_block('simulink/Signal Routing/Mux', [modelName '/reward_in_mux'], ...
    'Position',[790 315 795 415], 'Inputs','4');
add_block('simulink/User-Defined Functions/MATLAB Fcn', ...
    [modelName '/reward_fcn'], ...
    'Position',[840 345 960 385], ...
    'MATLABFcn','-(5*u(1)^2 + 0.1*u(2)^2 + 0.01*u(3)^2 + 50*max(0,abs(u(4))-Imax)^2)');
add_block('simulink/Math Operations/Abs', [modelName '/abs_ia'], ...
    'Position',[900 435 940 465]);
add_block('simulink/Logic and Bit Operations/Relational Operator', ...
    [modelName '/done_relop'], ...
    'Position',[1090 435 1160 470], ...
    'Operator','>');

% --------------------- Logging/outputs ---------------------
add_block('simulink/Sinks/Out1', [modelName '/omega_out'], ...
    'Position',[1280 205 1310 225]);
add_block('simulink/Sinks/Out1', [modelName '/ia_out'], ...
    'Position',[1280 175 1310 195]);

add_block('simulink/Sinks/To Workspace', [modelName '/omega_log'], ...
    'Position',[1210 248 1330 272], ...
    'VariableName','omega_log', ...
    'SaveFormat','Array');
add_block('simulink/Sinks/To Workspace', [modelName '/ia_log'], ...
    'Position',[1210 276 1330 300], ...
    'VariableName','ia_log', ...
    'SaveFormat','Array');
add_block('simulink/Sinks/To Workspace', [modelName '/u_log'], ...
    'Position',[890 88 1010 112], ...
    'VariableName','u_log', ...
    'SaveFormat','Array');
add_block('simulink/Sinks/To Workspace', [modelName '/reward_log'], ...
    'Position',[980 345 1100 369], ...
    'VariableName','reward_log', ...
    'SaveFormat','Array');

% ------------------------ Wiring ---------------------------
add_line(modelName, 'w_ref_const/1', 'error_sum/1', 'autorouting','on');
add_line(modelName, 'error_sum/1', 'd_error/1', 'autorouting','on');
add_line(modelName, 'error_sum/1', 'i_error/1', 'autorouting','on');
add_line(modelName, 'error_sum/1', 'obs_mux/1', 'autorouting','on');
add_line(modelName, 'd_error/1', 'obs_mux/2', 'autorouting','on');
add_line(modelName, 'i_error/1', 'obs_mux/3', 'autorouting','on');

add_line(modelName, 'obs_mux/1', 'RL Agent/1', 'autorouting','on');
add_line(modelName, 'RL Agent/1', 'action_scale/1', 'autorouting','on');
add_line(modelName, 'action_scale/1', 'voltage_sat/1', 'autorouting','on');
add_line(modelName, 'voltage_sat/1', 'plant_in_mux/1', 'autorouting','on');
add_line(modelName, 'Tload_const/1', 'plant_in_mux/2', 'autorouting','on');
add_line(modelName, 'plant_in_mux/1', 'motor_plant/1', 'autorouting','on');

add_line(modelName, 'motor_plant/1', 'state_demux/1', 'autorouting','on');
add_line(modelName, 'state_demux/1', 'ia_out/1', 'autorouting','on');
add_line(modelName, 'state_demux/2', 'omega_out/1', 'autorouting','on');
add_line(modelName, 'state_demux/1', 'ia_log/1', 'autorouting','on');
add_line(modelName, 'state_demux/2', 'omega_log/1', 'autorouting','on');

add_line(modelName, 'state_demux/2', 'error_sum/2', 'autorouting','on');
add_line(modelName, 'state_demux/2', 'obs_mux/4', 'autorouting','on');

add_line(modelName, 'error_sum/1', 'reward_in_mux/1', 'autorouting','on');
add_line(modelName, 'd_error/1', 'reward_in_mux/2', 'autorouting','on');
add_line(modelName, 'voltage_sat/1', 'reward_in_mux/3', 'autorouting','on');
add_line(modelName, 'state_demux/1', 'reward_in_mux/4', 'autorouting','on');
add_line(modelName, 'reward_in_mux/1', 'reward_fcn/1', 'autorouting','on');

add_line(modelName, 'reward_fcn/1', 'RL Agent/2', 'autorouting','on');
add_line(modelName, 'reward_fcn/1', 'reward_log/1', 'autorouting','on');

add_line(modelName, 'state_demux/1', 'abs_ia/1', 'autorouting','on');
add_line(modelName, 'abs_ia/1', 'done_relop/1', 'autorouting','on');
add_line(modelName, 'done_limit/1', 'done_relop/2', 'autorouting','on');
add_line(modelName, 'done_relop/1', 'RL Agent/3', 'autorouting','on');

add_line(modelName, 'voltage_sat/1', 'u_log/1', 'autorouting','on');

save_system(modelName,modelFile);
open_system(modelName);

if hasRlAgentBlock
    fprintf('Created fully wired RL-ready model: %s\n', modelFile);
    fprintf(['Model uses observation [e; de; ie; omega], action in [-1,1], ', ...
             'voltage scaling by Vmax, and reward/isDone signals for RL Agent.\n']);
else
    fprintf('Created model with placeholder RL Agent: %s\n', modelFile);
    fprintf(['Install Reinforcement Learning Toolbox, then rerun this script ', ...
             'to replace placeholder with actual RL Agent block.\n']);
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

function localAddPlaceholderAgent(modelName, position)
% Create a 3-input, 1-output placeholder agent subsystem.
subPath = [modelName '/RL Agent'];
add_block('simulink/Ports & Subsystems/Subsystem', subPath, ...
    'Position', position);

% Normalize default ports.
if ~isempty(find_system(subPath,'SearchDepth',1,'Name','In1'))
    set_param([subPath '/In1'], 'Name', 'obs');
else
    add_block('simulink/Sources/In1', [subPath '/obs'], ...
        'Position',[35 38 65 52], 'Port','1');
end

if ~isempty(find_system(subPath,'SearchDepth',1,'Name','Out1'))
    set_param([subPath '/Out1'], 'Name', 'action');
else
    add_block('simulink/Sinks/Out1', [subPath '/action'], ...
        'Position',[285 93 315 107]);
end

set_param([subPath '/obs'], 'Position', [35 38 65 52]);
set_param([subPath '/action'], 'Position', [285 93 315 107]);

try
    delete_line(subPath,'obs/1','action/1');
catch
    % line may not exist
end

% Add missing input ports (reward, isDone).
add_block('simulink/Sources/In1', [subPath '/reward'], ...
    'Position',[35 88 65 102], 'Port','2');
add_block('simulink/Sources/In1', [subPath '/isDone'], ...
    'Position',[35 138 65 152], 'Port','3');

% Keep subsystem deterministic: output zero action.
add_block('simulink/Sources/Constant', [subPath '/zero_action'], ...
    'Position',[160 90 210 110], 'Value','0');
add_line(subPath,'zero_action/1','action/1','autorouting','on');

% Terminate unused ports to avoid warnings.
add_block('simulink/Sinks/Terminator', [subPath '/term_obs'], ...
    'Position',[120 38 140 52]);
add_block('simulink/Sinks/Terminator', [subPath '/term_reward'], ...
    'Position',[120 88 140 102]);
add_block('simulink/Sinks/Terminator', [subPath '/term_done'], ...
    'Position',[120 138 140 152]);
add_line(subPath,'obs/1','term_obs/1','autorouting','on');
add_line(subPath,'reward/1','term_reward/1','autorouting','on');
add_line(subPath,'isDone/1','term_done/1','autorouting','on');
end

function [rlAgentLibPath, hasRlAgentBlock, diagText] = localFindRlAgentBlockPath()
% Locate RL Agent block in RL Toolbox library across versions.
rlAgentLibPath = '';
hasRlAgentBlock = false;
diagText = '';

% Fast pre-check to avoid noisy errors when toolbox is missing.
hasRlFunctions = exist('rlSimulinkEnv','file') == 2 || ...
                 exist('rlTD3Agent','file') == 2 || ...
                 exist('rlDDPGAgent','file') == 2;
if ~hasRlFunctions
    diagText = 'No RL Toolbox functions detected on MATLAB path.';
    return;
end

% Attempt to load known library names (version-dependent).
knownLibs = {'rllib','rl_lib','rlLib','reinforcement_learning_lib','reinforcementlearninglib'};
loadedLibs = {};
for i = 1:numel(knownLibs)
    lib = knownLibs{i};
    try
        load_system(lib);
        loadedLibs{end+1} = lib; %#ok<AGROW>
    catch
        % continue
    end
end

% Direct probe for canonical path first.
if any(strcmp(loadedLibs,'rllib'))
    try
        if getSimulinkBlockHandle('rllib/RL Agent') ~= -1
            rlAgentLibPath = 'rllib/RL Agent';
            hasRlAgentBlock = true;
            diagText = 'Resolved directly via rllib/RL Agent.';
            return;
        end
    catch
        % continue to broad search
    end
end

% Search all currently loaded Simulink libraries for blocks named RL Agent.
allDiagrams = find_system('type','block_diagram');
candidateBlocks = {};
for i = 1:numel(allDiagrams)
    bd = allDiagrams{i};
    try
        bdType = get_param(bd,'BlockDiagramType');
    catch
        continue;
    end
    if ~strcmpi(bdType,'library')
        continue;
    end
    try
        blocks = find_system(bd, ...
            'LookUnderMasks','all', ...
            'FollowLinks','on', ...
            'RegExp','on', ...
            'Name','^RL Agent$');
        if ~isempty(blocks)
            candidateBlocks = [candidateBlocks; blocks(:)]; %#ok<AGROW>
        end
    catch
        % continue
    end
end

% Prefer block whose mask type indicates RL Agent.
if ~isempty(candidateBlocks)
    preferredIdx = 1;
    for i = 1:numel(candidateBlocks)
        blk = candidateBlocks{i};
        mType = '';
        try
            mType = get_param(blk,'MaskType');
        catch
        end
        if contains(lower(mType),'rl') && contains(lower(mType),'agent')
            preferredIdx = i;
            break;
        end
    end

    rlAgentLibPath = candidateBlocks{preferredIdx};
    hasRlAgentBlock = true;
    diagText = sprintf('Loaded libs checked: %s\nSelected RL Agent block: %s', ...
        strjoin(loadedLibs, ', '), rlAgentLibPath);
    return;
end

diagText = sprintf(['RL functions found but no RL Agent library block named "RL Agent" ', ...
    'was discovered in loaded libraries.\nLoaded known libs: %s'], strjoin(loadedLibs, ', '));
end
