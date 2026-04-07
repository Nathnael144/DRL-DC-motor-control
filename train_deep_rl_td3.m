function [agent, T] = train_deep_rl_td3(maxEpisodes, showPlots, resumeFromCheckpoint, runEvaluation)
%TRAIN_DEEP_RL_TD3 Deep RL (TD3) for DC motor speed control.
%   Uses a custom MATLAB environment — NO Simulink — for fast training.
%   Bipolar action [-Vmax, Vmax], 6D normalized observation, dual-Gaussian
%   reward with normalized error + control effort/smoothness penalties.
%
%   Observations (6D):
%     [e, ie, de_agent, omega, ia, w_ref]  — matches classical controller
%     information (full-state + reference) for fair comparison.
%
%   After training, generates rl_*.mat scenario data and runs the full
%   comparison pipeline against LQR and pole-placement controllers.
%
%   Usage:
%       train_deep_rl_td3             % 20000 episodes, training monitor ON
%       train_deep_rl_td3(30000)      % more episodes
%       train_deep_rl_td3(3000,false) % fewer episodes, no training monitor
%       train_deep_rl_td3(500,true,true,false) % resume + no evaluation
%
%   The trained agent is saved to:
%       trained_rl_agent_td3_deep.mat  (full checkpoint)
%       trained_rl_agent.mat           (compatibility with eval pipeline)
%   By default, if a checkpoint exists, training resumes from it.

if nargin < 1 || isempty(maxEpisodes), maxEpisodes = 20000; end
if nargin < 2 || isempty(showPlots),   showPlots = true;   end
if nargin < 3 || isempty(resumeFromCheckpoint), resumeFromCheckpoint = true; end
if nargin < 4 || isempty(runEvaluation),        runEvaluation = true;       end

thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir), thisDir = pwd; end
oldDir = pwd;
cleanupObj = onCleanup(@() cd(oldDir)); %#ok<NASGU>
cd(thisDir);
addpath(thisDir);
checkpointFile = fullfile(thisDir,'trained_rl_agent_td3_deep.mat');

% -------- Motor & simulation parameters --------
p    = dc_motor_params();
Vmax = 24;
Imax = 8;
Ts   = 1e-4;           % simulation timestep (0.1 ms)
ctrlInterval = 10;     % agent acts every 10 sim steps = 1 ms
maxSteps = 1000;       % 1000 agent decisions × 1 ms = 1.0 s episode

% -------- Reward shaping / tuning knobs --------
% KEY INSIGHT: max r_gauss ≈ 1.6/sub-step ≈ 16/agent-step.
% Penalties must be a meaningful fraction of this.
% Round-2 problem: ControlEnergy was 1.5-1.7x LQR because u^2 penalty at
% moderate voltages is ~0.  Fix: add |u| term + efficiency bonus.
rewardCfg = struct( ...
    'sigmaBroad', 50, ...
    'sigmaTight', 8, ...
    'wBroad', 0.4, ...
    'wTight', 1.2, ...
    'eScale', 150, ...
    'wIAE', 0.15, ...
    'wISE', 0.25, ...
    'wU2', 0.06, ...
    'wU1', 0.05, ...
    'wDU', 0.015, ...
    'wOvershoot', 0.6, ...
    'wTerminal', 15.0, ...
    'precisionThresh', 3, ...
    'wPrecision', 0.5, ...
    'wEfficiency', 0.4, ...
    'effErrThresh', 8);

% -------- Training scenario mix --------
scenarioCfg = struct( ...
    'stepProb', 0.5, ...
    'rampProb', 0.25, ...
    'sineProb', 0.25);

loadCfg = struct( ...
    'constProb', 0.5, ...
    'stepProb', 0.5, ...
    'constAmp', 0.015, ...
    'stepAmp', 0.02, ...
    'stepTime', 0.2);

fprintf('=== Deep RL TD3 Training (custom env, no Simulink) ===\n');
fprintf('Motor: La=%.3g  Ra=%.3g  J=%.3g  Kt=%.3g\n', p.La, p.Ra, p.J, p.Kt);
fprintf('Sim: Ts=%.1e  ctrlInterval=%d  maxSteps=%d  episodeDuration=%.2fs\n', ...
    Ts, ctrlInterval, maxSteps, maxSteps*ctrlInterval*Ts);

% -------- Environment --------
obsInfo = rlNumericSpec([6 1], ...
    'LowerLimit', -inf(6,1), 'UpperLimit', inf(6,1));
obsInfo.Name = 'observations';
obsInfo.Description = 'Normalized [e, ie, de_agent, omega, ia, w_ref]';

actInfo = rlNumericSpec([1 1], 'LowerLimit', -1, 'UpperLimit', 1);
actInfo.Name = 'action';
actInfo.Description = 'Normalized voltage [-1,1] -> [-Vmax,Vmax]';

env = rlFunctionEnv(obsInfo, actInfo, ...
    @(a,ls) localStepFcn(a, ls, p, Vmax, Imax, Ts, ctrlInterval, maxSteps, rewardCfg), ...
    @() localResetFcn(p, scenarioCfg, loadCfg));

fprintf('Custom rlFunctionEnv created.\n');

% -------- TD3 Agent --------
resumeUsed = false;
if resumeFromCheckpoint && exist(checkpointFile,'file') == 2
    try
        Sckpt = load(checkpointFile,'agent');
        if isfield(Sckpt,'agent')
            agent = Sckpt.agent;
            resumeUsed = true;
            fprintf('Loaded checkpoint agent: %s\n', checkpointFile);
        end
    catch ME
        warning('Checkpoint load failed (%s). Starting from scratch.', ME.message);
    end
end

% Verify checkpoint agent is compatible with current 6D observation spec
if resumeUsed
    try
        testObs = zeros(obsInfo.Dimension);
        getAction(agent, testObs);
    catch
        warning('Checkpoint agent incompatible with %dD obs. Starting fresh.', obsInfo.Dimension(1));
        resumeUsed = false;
    end
end

if ~resumeUsed
    rng(0, 'twister');
    initOpts = rlAgentInitializationOptions(NumHiddenUnit=512);
    agent = rlTD3Agent(obsInfo, actInfo, initOpts);
    fprintf('TD3 agent created from scratch.\n');
else
    fprintf('TD3 agent resumed from checkpoint.\n');
end

agent.AgentOptions.SampleTime              = Ts * ctrlInterval;
agent.AgentOptions.DiscountFactor          = 0.995;
agent.AgentOptions.TargetSmoothFactor      = 5e-3;
agent.AgentOptions.ExperienceBufferLength  = 2e6;
agent.AgentOptions.MiniBatchSize           = 512;

% Exploration noise (Gaussian, decaying)
agent.AgentOptions.ExplorationModel.StandardDeviation          = 0.3;
agent.AgentOptions.ExplorationModel.StandardDeviationDecayRate = 1e-5;
agent.AgentOptions.ExplorationModel.StandardDeviationMin       = 0.01;
agent.AgentOptions.ExplorationModel.LowerLimit = -1;
agent.AgentOptions.ExplorationModel.UpperLimit =  1;

% Target policy smoothing (TD3 regularisation)
if isprop(agent.AgentOptions, 'TargetPolicySmoothModel')
    agent.AgentOptions.TargetPolicySmoothModel.StandardDeviation = 0.2;
    agent.AgentOptions.TargetPolicySmoothModel.LowerLimit = -0.5;
    agent.AgentOptions.TargetPolicySmoothModel.UpperLimit =  0.5;
end

% Optimizer learning rates
agent.AgentOptions.ActorOptimizerOptions.LearnRate       = 1e-4;
agent.AgentOptions.ActorOptimizerOptions.GradientThreshold = 1;
for ci = 1:numel(agent.AgentOptions.CriticOptimizerOptions)
    agent.AgentOptions.CriticOptimizerOptions(ci).LearnRate       = 3e-4;
    agent.AgentOptions.CriticOptimizerOptions(ci).GradientThreshold = 1;
end

fprintf('TD3 agent configured: 512 hidden units, bipolar action, 6D obs.\n');

% -------- Training --------
plotMode = 'none';
if showPlots, plotMode = 'training-progress'; end

trainOpts = rlTrainingOptions( ...
    'MaxEpisodes',              maxEpisodes, ...
    'MaxStepsPerEpisode',       maxSteps, ...
    'StopTrainingCriteria',     'AverageReward', ...
    'StopTrainingValue',        14000, ...
    'ScoreAveragingWindowLength', 100, ...
    'Verbose',                  false, ...
    'Plots',                    plotMode);

if resumeUsed
    fprintf('Training resume: up to %d additional episodes (avg-reward stop at 14000).\n', maxEpisodes);
else
    fprintf('Training from scratch: up to %d episodes (avg-reward stop at 14000).\n', maxEpisodes);
end
fprintf('Reward range: ~0 to ~17000 depending on shaping/penalties.\n\n');

trainingStats = train(agent, env, trainOpts);

% -------- Save --------
algorithm = "td3"; %#ok<NASGU>
save(checkpointFile, ...
    'agent','trainingStats','trainOpts','algorithm','resumeUsed');
save(fullfile(thisDir,'trained_rl_agent.mat'), 'agent','algorithm');
fprintf('\nSaved: %s + trained_rl_agent.mat\n', checkpointFile);

% -------- Evaluation --------
T = [];
if runEvaluation
    fprintf('\nEvaluating on benchmark scenarios...\n');
    T = localGenerateScenarioData(agent, p, Vmax, Imax, Ts, ctrlInterval, thisDir);
    fprintf('\n=== RL Scenario Metrics ===\n');
    disp(T);

    fprintf('Running comparison vs LQR / PP...\n');
    try
        evalin('base', sprintf('run(''%s'');', ...
            strrep(fullfile(thisDir,'compare_controllers.m'),'''','''''')));
    catch ME
        warning('compare_controllers failed: %s', ME.message);
    end
    try
        f = fullfile(thisDir,'create_graph_table_comparison.m');
        if exist(f,'file') == 2
            evalin('base', sprintf('run(''%s'');', strrep(f,'''','''''')));
        end
    catch ME
        warning('create_graph_table_comparison failed: %s', ME.message);
    end
else
    fprintf('\nSkipping evaluation (runEvaluation=false).\n');
end

fprintf('\n=== Deep RL TD3 pipeline complete ===\n');
end

%% =====================================================================
%  Environment: step function
%  =====================================================================
function [obs, reward, isDone, ls] = localStepFcn(action, ls, p, Vmax, Imax, Ts, ctrlInterval, maxSteps, rewardCfg)
% Apply the same voltage for ctrlInterval simulation sub-steps.
% 6D observation: [e, ie, de_agent, omega, ia, w_ref] — gives the agent
% the same information available to classical controllers (full state +
% reference) for a fair comparison.

V = double(action(1)) * Vmax;
V = max(-Vmax, min(Vmax, V));

if ~isfield(ls, 'prevV')
    ls.prevV = 0;
end
dV = V - ls.prevV;
u_norm  = V / Vmax;
du_norm = dV / Vmax;

% Save error at start of this agent step (for agent-rate derivative)
e_at_start = ls.e_prev;

reward = 0;
for sub = 1:ctrlInterval
    % Reference and load for current time
    ls.w_ref = localReference(ls.t, ls.refType, ls.refParams);
    Tload = localLoad(ls.t, ls);

    % RK4 integration
    [ls.ia, ls.omega] = localRK4Step(ls.ia, ls.omega, V, Tload, p, Ts);

    % Error signals
    e  = ls.w_ref - ls.omega;
    ls.ie     = ls.ie + e * Ts;
    ls.e_prev = e;

    % Dual-Gaussian reward + error/control penalties + efficiency bonus
    e_norm = e / rewardCfg.eScale;
    r_gauss = rewardCfg.wBroad*exp(-(e/rewardCfg.sigmaBroad)^2) + ...
              rewardCfg.wTight*exp(-(e/rewardCfg.sigmaTight)^2);

    % Control penalty: |u| + u^2 + du^2.
    % The |u| term creates a gradient even at moderate voltages
    % (u^2 alone was ~0 for u_norm<0.3, so agent didn't bother reducing).
    r_pen = rewardCfg.wIAE*abs(e_norm) + rewardCfg.wISE*(e_norm^2) + ...
            rewardCfg.wU2*(u_norm^2) + rewardCfg.wU1*abs(u_norm) + ...
            (rewardCfg.wDU/ctrlInterval)*(du_norm^2);

    % Overshoot penalty: when speed exceeds reference (e < 0)
    r_overshoot = 0;
    if e < 0
        r_overshoot = rewardCfg.wOvershoot * (e_norm^2);
    end

    % Precision bonus: tighter threshold, scales up over episode
    r_prec = 0;
    if abs(e) < rewardCfg.precisionThresh
        tFrac = min(ls.step / maxSteps, 1);
        r_prec = rewardCfg.wPrecision * (1 + tFrac);
    end

    % Efficiency bonus: reward low voltage WHILE tracking well.
    % This is the key signal for learning steady-state behaviour.
    % When |e|<8 and |u_norm|<0.3 (≈7V), agent gets up to 0.4/sub-step.
    r_eff = 0;
    if abs(e) < rewardCfg.effErrThresh
        r_eff = rewardCfg.wEfficiency * max(0, 1 - abs(u_norm)/0.3);
    end

    r_sub = r_gauss - r_pen - r_overshoot + r_prec + r_eff;
    reward = reward + r_sub;

    ls.t = ls.t + Ts;
end

ls.prevV = V;
ls.step = ls.step + 1;
isDone  = (ls.step >= maxSteps) || (abs(ls.ia) > 3.5*Imax);

% Stronger terminal SSE penalty
e_final = ls.e_prev;
if ls.step >= maxSteps
    e_norm_end = e_final / rewardCfg.eScale;
    reward = reward - rewardCfg.wTerminal*abs(e_norm_end);
end

% Agent-rate derivative (smooth, interpretable — unlike noisy sim-rate de)
de_agent = (e_final - e_at_start) / (Ts * ctrlInterval);

% 6D normalised observation:
%   e      — tracking error   (tanh, scale 80 — better gradient than /120)
%   ie     — integral error   (tanh, scale 3  — was /60, fixing dead channel!)
%   de_agt — agent-rate deriv (tanh, scale 50k — clean signal)
%   omega  — motor speed      (linear /150)
%   ia     — armature current (linear /Imax)  [NEW — enables torque awareness]
%   w_ref  — reference signal (linear /150)   [NEW — enables feedforward]
obs = [tanh(e_final / 80); ...
       tanh(ls.ie / 3); ...
       tanh(de_agent / 50000); ...
       ls.omega / 150; ...
       ls.ia / Imax; ...
       ls.w_ref / 150];
end

%% =====================================================================
%  Environment: reset function
%  =====================================================================
function [obs, ls] = localResetFcn(p, scenarioCfg, loadCfg) %#ok<INUSD>
ls = struct();
ls.ia      = 0;
ls.omega   = 0;
ls.ie      = 0;
ls.e_prev  = 0;
ls.step    = 0;
ls.prevV   = 0;
ls.t      = 0;

% Reference scenario (step / ramp / sine) — wider randomisation for
% robustness.  Includes the exact benchmark values in the range.
r = rand;
if r < scenarioCfg.stepProb
    ls.refType = 'step';
    ls.refParams = struct('value', 40 + 110*rand); % 40..150 rad/s (wider)
elseif r < scenarioCfg.stepProb + scenarioCfg.rampProb
    ls.refType = 'ramp';
    rampSlope = 150 + 100*rand;  % 150..250 rad/s/s (randomised)
    rampMax   = 80 + 60*rand;    % 80..140 rad/s
    ls.refParams = struct('slope', rampSlope, 'max', rampMax);
else
    ls.refType = 'sine';
    sineBias = 80 + 40*rand;     % 80..120 rad/s
    sineAmp  = 10 + 20*rand;     % 10..30 rad/s
    sineFreq = 1 + 3*rand;       % 1..4 Hz
    ls.refParams = struct('bias', sineBias, 'amp', sineAmp, 'freq', sineFreq);
end

% Load profile (constant or step) — randomised amplitude
r = rand;
if r < loadCfg.constProb
    ls.loadType = 'const';
    ls.TloadBase = loadCfg.constAmp*(2*rand - 1);
    ls.TloadStep = 0;
else
    ls.loadType = 'step';
    ls.TloadBase = 0;
    ls.TloadStep = loadCfg.stepAmp * (0.5 + rand); % 0.5x..1.5x
end
ls.TloadStepTime = loadCfg.stepTime;

% Initial reference and error
ls.w_ref = localReference(ls.t, ls.refType, ls.refParams);
e = ls.w_ref - ls.omega;   % initial error = w_ref - 0
ls.e_prev = e;

% 6D observation: [e, ie, de_agent, omega, ia, w_ref]
obs = [tanh(e / 80); 0; 0; 0; 0; ls.w_ref / 150];
end

function w_ref = localReference(t, refType, refParams)
switch refType
    case 'step'
        w_ref = refParams.value;
    case 'ramp'
        w_ref = min(refParams.slope * t, refParams.max);
    case 'sine'
        w_ref = refParams.bias + refParams.amp * sin(2*pi*refParams.freq*t);
    otherwise
        w_ref = 0;
end
end

function tau = localLoad(t, ls)
tau = ls.TloadBase;
if strcmp(ls.loadType,'step') && t >= ls.TloadStepTime
    tau = tau + ls.TloadStep;
end
end

%% =====================================================================
%  Motor dynamics helpers
%  =====================================================================
function [ia_next, omega_next] = localRK4Step(ia, omega, V, Tload, p, Ts)
x = [ia; omega];
f = @(xx) localMotorDyn(xx, V, Tload, p);
k1 = f(x);
k2 = f(x + 0.5*Ts*k1);
k3 = f(x + 0.5*Ts*k2);
k4 = f(x + Ts*k3);
x_next = x + (Ts/6)*(k1 + 2*k2 + 2*k3 + k4);
ia_next    = x_next(1);
omega_next = x_next(2);
end

function dx = localMotorDyn(x, V, Tload, p)
ia    = x(1);
omega = x(2);
dia    = -(p.Ra/p.La)*ia - (p.Ke/p.La)*omega + (1/p.La)*V;
domega =  (p.Kt/p.J)*ia  - (p.Bm/p.J)*omega - (1/p.J)*Tload;
dx = [dia; domega];
end

%% =====================================================================
%  Evaluation: generate rl_*.mat scenario data
%  =====================================================================
function T = localGenerateScenarioData(agent, p, Vmax, Imax, Ts, ctrlInterval, thisDir)
scenarios = {'step_nominal','step_load_disturbance','ramp','sine'};
Tend = 0.5;
t = (0:Ts:Tend)';
N = numel(t);

records = struct('Scenario',{},'IAE',{},'ISE',{},'SSE',{}, ...
    'ControlEnergy',{},'RiseTime',{},'SettlingTime',{},'Overshoot',{});

for i = 1:numel(scenarios)
    sName = scenarios{i};
    [r, tauLoad] = benchmark_scenario_signals(sName, t);

    % Simulate RL policy
    ia = 0; omega = 0; ie = 0; e_prev = r(1); e_prev_agent = r(1);
    y = zeros(N,1);
    u = zeros(N,1);
    xLog = zeros(N,2);
    V = 0;

    for k = 1:N
        y(k) = omega;
        xLog(k,:) = [ia, omega];

        % Agent decides every ctrlInterval simulation steps
        if mod(k-1, ctrlInterval) == 0
            e  = r(k) - omega;
            de_agent = (e - e_prev_agent) / (Ts * ctrlInterval);
            obs = [tanh(e/80); tanh(ie/3); tanh(de_agent/50000); omega/150; ia/Imax; r(k)/150];
            e_prev_agent = e;
            actionCell = getAction(agent, obs);
            if iscell(actionCell)
                a = double(actionCell{1}(1));
            else
                a = double(actionCell(1));
            end
            V = a * Vmax;
            V = max(-Vmax, min(Vmax, V));
        end

        u(k) = V;

        % RK4 step
        [ia, omega] = localRK4Step(ia, omega, V, tauLoad(k), p, Ts);

        e  = r(k) - omega;
        ie = ie + e * Ts;
        e_prev = e;
    end

    % Save scenario data (same format as generate_rl_scenario_data.m)
    t_rl   = t;       %#ok<NASGU>
    y_rl   = y;       %#ok<NASGU>
    u_rl   = u;       %#ok<NASGU>
    r_rl   = r;       %#ok<NASGU>
    tau_rl = tauLoad; %#ok<NASGU>
    x_rl   = xLog;   %#ok<NASGU>
    outFile = fullfile(thisDir, ['rl_' sName '.mat']);
    save(outFile, 't_rl','y_rl','u_rl','r_rl','tau_rl','x_rl');
    fprintf('  Saved %s\n', outFile);

    m = calcMetrics(t, y, r, u);
    rec.Scenario     = sName;
    rec.IAE          = m.IAE;
    rec.ISE          = m.ISE;
    rec.SSE          = m.SSE;
    rec.ControlEnergy = m.ControlEnergy;
    rec.RiseTime     = m.RiseTime;
    rec.SettlingTime = m.SettlingTime;
    rec.Overshoot    = m.Overshoot;
    records(end+1)   = rec; %#ok<AGROW>
end

T = struct2table(records);
save(fullfile(thisDir,'rl_scenario_metrics.mat'), 'T','records');
end

%% =====================================================================
%  Scenario signal generators (identical to compare_controllers.m)
%  =====================================================================
 
