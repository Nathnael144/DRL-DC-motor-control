function [agent, T] = train_deep_rl_td3_tuned(maxEpisodes, showPlots, resumeFromCheckpoint, runEvaluation)
%TRAIN_DEEP_RL_TD3_TUNED  Tuned TD3 for DC motor speed control.
%
%   Key improvements over train_deep_rl_td3:
%     1. Composite reward with linear + quadratic error penalty, integral
%        error penalty, control cost, and terminal SSE bonus.
%     2. Wider integral-error observation range (tanh(ie/500) vs ie/60).
%     3. 512 hidden-unit networks.
%     4. Wider training distribution (load ± 0.03 Nm, ref 40-160 rad/s).
%     5. Longer training (5000 eps), higher stop threshold.
%
%   Usage:
%       train_deep_rl_td3_tuned           % defaults
%       train_deep_rl_td3_tuned(8000)     % more episodes
%       train_deep_rl_td3_tuned(5000,true,false)  % fresh start
%
%   Saves:  trained_rl_agent_td3_tuned.mat  (checkpoint)
%           trained_rl_agent.mat            (eval pipeline compat)

if nargin < 1 || isempty(maxEpisodes), maxEpisodes = 1000; end
if nargin < 2 || isempty(showPlots),   showPlots = true;   end
if nargin < 3 || isempty(resumeFromCheckpoint), resumeFromCheckpoint = false; end
if nargin < 4 || isempty(runEvaluation),        runEvaluation = true;         end

thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir), thisDir = pwd; end
oldDir = pwd;
cleanupObj = onCleanup(@() cd(oldDir)); %#ok<NASGU>
cd(thisDir);
addpath(thisDir);
checkpointFile = fullfile(thisDir,'trained_rl_agent_td3_tuned.mat');


% -------- TD3 Agent --------
resumeUsed = false;
if resumeFromCheckpoint && exist(checkpointFile,'file') == 2
    try
        Sckpt = load(checkpointFile,'agent');
        if isfield(Sckpt,'agent')
            agent = Sckpt.agent;
            resumeUsed = true;
            fprintf('Loaded checkpoint: %s\n', checkpointFile);
        end
    catch ME
        warning('Checkpoint load failed (%s). Starting fresh.', ME.message);
    end
end

if ~resumeUsed
    rng(42, 'twister');
    initOpts = rlAgentInitializationOptions(NumHiddenUnit=512);
    agent = rlTD3Agent(obsInfo, actInfo, initOpts);
    fprintf('TD3 agent created (512 hidden units, fresh).\n');
else
    fprintf('TD3 agent resumed from checkpoint.\n');
end

agent.AgentOptions.SampleTime              = Ts * ctrlInterval;
agent.AgentOptions.DiscountFactor          = 0.995;   % longer horizon
agent.AgentOptions.TargetSmoothFactor      = 5e-3;
agent.AgentOptions.ExperienceBufferLength  = 2e6;
agent.AgentOptions.MiniBatchSize           = 512;

% Exploration noise — start higher, decay slower
agent.AgentOptions.ExplorationModel.StandardDeviation          = 0.4;
agent.AgentOptions.ExplorationModel.StandardDeviationDecayRate = 5e-6;
agent.AgentOptions.ExplorationModel.StandardDeviationMin       = 0.02;
agent.AgentOptions.ExplorationModel.LowerLimit = -1;
agent.AgentOptions.ExplorationModel.UpperLimit =  1;

% Target policy smoothing
if isprop(agent.AgentOptions, 'TargetPolicySmoothModel')
    agent.AgentOptions.TargetPolicySmoothModel.StandardDeviation = 0.2;
    agent.AgentOptions.TargetPolicySmoothModel.LowerLimit = -0.5;
    agent.AgentOptions.TargetPolicySmoothModel.UpperLimit =  0.5;
end

% Learning rates — slightly larger for critic, smaller for actor
agent.AgentOptions.ActorOptimizerOptions.LearnRate         = 1e-4;
agent.AgentOptions.ActorOptimizerOptions.GradientThreshold = 1;
for ci = 1:numel(agent.AgentOptions.CriticOptimizerOptions)
    agent.AgentOptions.CriticOptimizerOptions(ci).LearnRate         = 8e-4;
    agent.AgentOptions.CriticOptimizerOptions(ci).GradientThreshold = 1;
end

    agent.AgentOptions.CriticOptimizerOptions(ci).GradientThreshold = 1;

% -------- Training: 20 chunks of 50 episodes --------
plotMode = 'none';
if showPlots, plotMode = 'training-progress'; end

chunkSize = 50;
totalEpisodes = 1000;
numChunks = totalEpisodes / chunkSize;
allStats = cell(numChunks,1);
episodesDone = 0;

for chunk = 1:numChunks
    fprintf('\n--- Training chunk %d/%d: episodes %d to %d ---\n', chunk, numChunks, episodesDone+1, episodesDone+chunkSize);
    trainOpts = rlTrainingOptions( ...
        'MaxEpisodes', chunkSize, ...
        'MaxStepsPerEpisode', maxSteps, ...
        'StopTrainingCriteria', 'EpisodeCount', ...
        'StopTrainingValue', chunkSize, ...
        'ScoreAveragingWindowLength', 100, ...
        'Verbose', false, ...
        'Plots', plotMode);
    trainingStatsChunk = train(agent, env, trainOpts);
    allStats{chunk} = trainingStatsChunk;
    episodesDone = episodesDone + chunkSize;
    algorithm = "td3_tuned"; %#ok<NASGU>
    save(checkpointFile, ...
        'agent','allStats','trainOpts','algorithm','resumeUsed','episodesDone','chunkSize');
    save(fullfile(thisDir,'trained_rl_agent.mat'), 'agent','algorithm');
    fprintf('Checkpoint saved after %d episodes.\n', episodesDone);
    % Optionally print summary stats for this chunk
    if isprop(trainingStatsChunk, 'AverageReward')
        avgR = trainingStatsChunk.AverageReward(end);
    elseif isfield(trainingStatsChunk, 'AverageReward')
        avgR = trainingStatsChunk.AverageReward(end);
    else
        avgR = NaN;
    end
    fprintf('Chunk %d complete. Last avg reward: %.2f\n', chunk, avgR);
end

trainingStats = allStats;

fprintf('Saved: %s + trained_rl_agent.mat\n', checkpointFile);
end

fprintf('Agent configured: 512 HU, gamma=0.995, actor LR=5e-5.\n');

% -------- Training --------
plotMode = 'none';
if showPlots, plotMode = 'training-progress'; end

trainOpts = rlTrainingOptions( ...
    'MaxEpisodes',              100, ... % will be overridden per chunk
    'MaxStepsPerEpisode',       maxSteps, ...
    'StopTrainingCriteria',     'AverageReward', ...
    'StopTrainingValue',        3500, ...
    'ScoreAveragingWindowLength', 100, ...
    'Verbose',                  false, ...
    'Plots',                    plotMode);

chunkSize = 100;
fprintf('Training: up to %d episodes in chunks of %d (stop at avg reward 3500).\n', maxEpisodes, chunkSize);
fprintf('Reward: reshaped for tracking precision + smoothness + disturbance recovery.\n\n');

allStats = cell(0,1);
episodesDone = 0;
chunkIdx = 0;
while episodesDone < maxEpisodes
    chunkIdx = chunkIdx + 1;
    thisChunk = min(chunkSize, maxEpisodes - episodesDone);
    trainOptsChunk = trainOpts;
    trainOptsChunk.MaxEpisodes = thisChunk;

    fprintf('--- Chunk %d: episodes %d..%d ---\n', chunkIdx, episodesDone+1, episodesDone+thisChunk);
    trainingStatsChunk = train(agent, env, trainOptsChunk);
    allStats{end+1,1} = trainingStatsChunk; %#ok<AGROW>
    episodesDone = episodesDone + thisChunk;

    % Save checkpoint each chunk (safe stop/resume)
    algorithm = "td3_tuned"; %#ok<NASGU>
    save(checkpointFile, ...
        'agent','allStats','trainOpts','algorithm','resumeUsed','episodesDone','chunkSize');
    save(fullfile(thisDir,'trained_rl_agent.mat'), 'agent','algorithm');
    fprintf('Checkpoint saved after %d episodes.\n\n', episodesDone);

    % Optional early stop if criteria met (if supported by training stats)
    try
        if isprop(trainingStatsChunk, 'AverageReward')
            avgR = trainingStatsChunk.AverageReward(end);
        elseif isfield(trainingStatsChunk, 'AverageReward')
            avgR = trainingStatsChunk.AverageReward(end);
        else
            avgR = NaN;
        end
        if isfinite(avgR) && avgR >= trainOpts.StopTrainingValue
            fprintf('Early stop: average reward %.3f >= %.3f\n', avgR, trainOpts.StopTrainingValue);
            break;
        end
    catch
        % ignore and continue
    end
end

% -------- Save --------
algorithm = "td3_tuned"; %#ok<NASGU>
save(checkpointFile, ...
    'agent','allStats','trainOpts','algorithm','resumeUsed','episodesDone','chunkSize');
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

fprintf('\n=== Tuned Deep RL TD3 pipeline complete ===\n');

%% =====================================================================
%  Environment: step function  — REDESIGNED REWARD
%  =====================================================================

function [obs, reward, isDone, ls] = stepFcn_td3(action, ls, p, Vmax, Imax, Ts, ctrlInterval, maxSteps)
V = double(action(1)) * Vmax;
% Small action delay / actuator noise to learn robustness
V = V + 0.02 * Vmax * randn;
V = max(-Vmax, min(Vmax, V));

% Update state
[ls.ia, ls.omega] = localRK4Step(ls.ia, ls.omega, V, ls.Tload, p, Ts);

% Python-style reward: reward = c * abs(delta_omega_next)
c = -0.95;
omega_bound = 1250;
delta_omega_next = (ls.w_ref / omega_bound) - (ls.omega / omega_bound);
reward = c * abs(delta_omega_next);

ls.step = ls.step + 1;
isDone  = (ls.step >= maxSteps) || (abs(ls.ia) > 3.5*Imax);

% Normalised observation (keep as before)
e = ls.w_ref - ls.omega;
de_ctrl = 0; % If you want to keep derivative info, update as needed
obs = [tanh(ls.ie / 500);     % WIDER range (was /60) — preserves integral info
       tanh(e / 120);
       tanh(de_ctrl / 4000);
       ls.omega / 200;
       tanh(ls.ia / Imax);
       ls.w_ref / 200];
end

%% =====================================================================
%  Environment: reset function — WIDER TRAINING DISTRIBUTION
%  =====================================================================
function [obs, ls] = localResetFcn(p) %#ok<INUSD>
ls = struct();
ls.ia      = 0;
ls.omega   = 0;
ls.ie      = 0;
ls.e_prev  = 0;
ls.e_prev_ctrl = 0;
ls.de_ctrl = 0;
ls.step    = 0;
ls.prevV   = 0;
ls.w_ref   = 20 + 180*rand;          % 20 .. 200 rad/s
ls.Tload   = 0.05*(2*rand - 1);      % +/- 0.05 Nm

e = ls.w_ref;
ls.e_prev = e;
ls.e_prev_ctrl = e;

obs = [0; tanh(e/120); 0; 0; 0; ls.w_ref/200];
end

%% =====================================================================
%  Motor dynamics
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

    ia = 0; omega = 0; ie = 0; e_prev = r(1);
    y = zeros(N,1);
    u = zeros(N,1);
    xLog = zeros(N,2);
    V = 0;

    for k = 1:N
        y(k) = omega;
        xLog(k,:) = [ia, omega];

        if mod(k-1, ctrlInterval) == 0
            e  = r(k) - omega;
            de_ctrl = (e - e_prev) / (Ts*ctrlInterval);
            obs = [tanh(ie/500); tanh(e/120); tanh(de_ctrl/4000); omega/200; tanh(ia/Imax); r(k)/200];
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
        [ia, omega] = localRK4Step(ia, omega, V, tauLoad(k), p, Ts);

        e  = r(k) - omega;
        ie = ie + e * Ts;
        e_prev = e;
    end

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
%  Scenario signals (same as compare_controllers.m)
%  =====================================================================
 
