%% post_training_check.m
% Quick post-training metric comparison: RL vs LQR vs PP.
% Loads the trained agent, re-evaluates on all 4 benchmark scenarios,
% and prints a clear scorecard showing where RL wins/loses.
%
% Usage:  run('post_training_check.m')
%         (or just F5 in the editor)

clear; clc; close all;

thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir), thisDir = pwd; end
addpath(thisDir);
cd(thisDir);

%% ---- Load trained RL agent ----
agentFile = fullfile(thisDir,'trained_rl_agent_td3_deep.mat');
if exist(agentFile,'file') ~= 2
    agentFile = fullfile(thisDir,'trained_rl_agent.mat');
end
if exist(agentFile,'file') ~= 2
    error('No trained agent found. Run train_deep_rl_td3 first.');
end
S = load(agentFile,'agent');
agent = S.agent;
fprintf('Loaded agent from: %s\n\n', agentFile);

%% ---- Motor parameters ----
p    = dc_motor_params();
Vmax = 24;
Imax = 8;
Ts   = 1e-4;
ctrlInterval = 10;

%% ---- Classical controllers ----
lqrOut = lqr_controller();
ppOut  = pole_placement_controller();

%% ---- Benchmark scenarios ----
scenarios = {'step_nominal','step_load_disturbance','ramp','sine'};
Tend = 0.5;
t = (0:Ts:Tend)';
N = numel(t);

metricNames  = {'IAE','ISE','SSE','ControlEnergy','RiseTime','SettlingTime','Overshoot'};
compareNames = {'IAE','ISE','SSE','ControlEnergy'};  % lower is better

allRows = {};
wins = struct('RL',0,'LQR',0,'PP',0,'tie',0);

for si = 1:numel(scenarios)
    sName = scenarios{si};
    [r, tauLoad] = benchmark_scenario_signals(sName, t);

    % --- LQR ---
    [yLQR, uLQR] = simClassical(p, lqrOut, t, r, tauLoad, Vmax);
    mLQR = calcMetrics(t, yLQR, r, uLQR);

    % --- PP ---
    [yPP, uPP] = simClassical(p, ppOut, t, r, tauLoad, Vmax);
    mPP = calcMetrics(t, yPP, r, uPP);

    % --- RL ---
    [yRL, uRL] = simRL(agent, p, Vmax, Imax, Ts, ctrlInterval, t, r, tauLoad);
    mRL = calcMetrics(t, yRL, r, uRL);

    % --- Print scenario ---
    fprintf('============================================================\n');
    fprintf(' Scenario: %s\n', sName);
    fprintf('------------------------------------------------------------\n');
    fprintf('%-16s %12s %12s %12s   Winner\n', 'Metric', 'LQR', 'PP', 'RL');
    fprintf('%-16s %12s %12s %12s   ------\n', '------', '---', '--', '--');

    for mi = 1:numel(compareNames)
        mn = compareNames{mi};
        vL = mLQR.(mn);
        vP = mPP.(mn);
        vR = mRL.(mn);
        best = min([vL vP vR]);

        if     vR == best && vR < vL && vR < vP,  w = 'RL ***';  wins.RL  = wins.RL + 1;
        elseif vL == best && vL < vR && vL < vP,  w = 'LQR';     wins.LQR = wins.LQR + 1;
        elseif vP == best && vP < vR && vP < vL,  w = 'PP';      wins.PP  = wins.PP + 1;
        else,                                       w = 'tie';     wins.tie = wins.tie + 1;
        end

        fprintf('%-16s %12.4f %12.4f %12.4f   %s\n', mn, vL, vP, vR, w);
    end

    % Step metrics (only meaningful for step scenarios)
    if isfinite(mLQR.RiseTime)
        fprintf('%-16s %12.5f %12.5f %12.5f\n', 'RiseTime(s)', mLQR.RiseTime, mPP.RiseTime, mRL.RiseTime);
        fprintf('%-16s %12.5f %12.5f %12.5f\n', 'SettlingTime(s)', mLQR.SettlingTime, mPP.SettlingTime, mRL.SettlingTime);
        fprintf('%-16s %12.2f %12.2f %12.2f\n', 'Overshoot(%)', mLQR.Overshoot, mPP.Overshoot, mRL.Overshoot);
    end
    fprintf('\n');
end

%% ---- Scorecard ----
totalComparisons = wins.RL + wins.LQR + wins.PP + wins.tie;
fprintf('============================================================\n');
fprintf('                   OVERALL SCORECARD\n');
fprintf('============================================================\n');
fprintf('  RL wins:   %d / %d  (%.0f%%)\n', wins.RL, totalComparisons, 100*wins.RL/totalComparisons);
fprintf('  LQR wins:  %d / %d\n', wins.LQR, totalComparisons);
fprintf('  PP wins:   %d / %d\n', wins.PP, totalComparisons);
fprintf('  Ties:      %d / %d\n', wins.tie, totalComparisons);
fprintf('============================================================\n');

if wins.RL > wins.LQR && wins.RL > wins.PP
    fprintf('  >> RL is the OVERALL WINNER <<\n');
elseif wins.RL == 0
    fprintf('  >> RL lost all comparisons — needs more training or tuning <<\n');
else
    fprintf('  >> Mixed results — RL competitive but not dominant <<\n');
end
fprintf('============================================================\n');

%% ---- Helper: simulate classical controller with saturation ----
function [y, u] = simClassical(p, ctrl, t, r, tauLoad, Vmax)
N = numel(t);
Ts = t(2) - t(1);
x = zeros(2, 1);
y = zeros(N, 1);
u = zeros(N, 1);
for k = 1:N
    y(k) = x(2);
    uUnsat = -ctrl.K*x + ctrl.Nbar*r(k);
    u(k) = max(-Vmax, min(Vmax, uUnsat));
    if k < N
        dx = [-(p.Ra/p.La)*x(1) - (p.Ke/p.La)*x(2) + (1/p.La)*u(k);
              (p.Kt/p.J)*x(1) - (p.Bm/p.J)*x(2) - (1/p.J)*tauLoad(k)];
        x = x + Ts*dx;
        % RK4 for consistency
        f = @(xx) [-(p.Ra/p.La)*xx(1) - (p.Ke/p.La)*xx(2) + (1/p.La)*u(k);
                    (p.Kt/p.J)*xx(1) - (p.Bm/p.J)*xx(2) - (1/p.J)*tauLoad(k)];
        xk = x - Ts*dx;  % undo Euler
        k1 = f(xk);
        k2 = f(xk + 0.5*Ts*k1);
        k3 = f(xk + 0.5*Ts*k2);
        k4 = f(xk + Ts*k3);
        x = xk + (Ts/6)*(k1 + 2*k2 + 2*k3 + k4);
    end
end
end

%% ---- Helper: simulate RL agent ----
function [y, u] = simRL(agent, p, Vmax, Imax, Ts, ctrlInterval, t, r, tauLoad)
N = numel(t);
ia = 0; omega = 0; ie = 0; e_prev_agent = r(1);
y = zeros(N, 1);
u = zeros(N, 1);
V = 0;
for k = 1:N
    y(k) = omega;
    if mod(k-1, ctrlInterval) == 0
        e = r(k) - omega;
        de_agent = (e - e_prev_agent) / (Ts * ctrlInterval);
        obs = [tanh(e/80); tanh(ie/3); tanh(de_agent/50000); omega/150; ia/Imax; r(k)/150];
        e_prev_agent = e;
        actionCell = getAction(agent, obs);
        if iscell(actionCell)
            a = double(actionCell{1}(1));
        else
            a = double(actionCell(1));
        end
        V = max(-Vmax, min(Vmax, a * Vmax));
    end
    u(k) = V;
    % RK4 step
    f = @(xx) [-(p.Ra/p.La)*xx(1) - (p.Ke/p.La)*xx(2) + (1/p.La)*V;
                (p.Kt/p.J)*xx(1) - (p.Bm/p.J)*xx(2) - (1/p.J)*tauLoad(k)];
    xk = [ia; omega];
    k1 = f(xk);
    k2 = f(xk + 0.5*Ts*k1);
    k3 = f(xk + 0.5*Ts*k2);
    k4 = f(xk + Ts*k3);
    xn = xk + (Ts/6)*(k1 + 2*k2 + 2*k3 + k4);
    ia = xn(1);
    omega = xn(2);
    e = r(k) - omega;
    ie = ie + e * Ts;
end
end
