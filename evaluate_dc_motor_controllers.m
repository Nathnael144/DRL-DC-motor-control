%% evaluate_dc_motor_controllers
% Compare LQR, pole-placement, and RL controllers on dc_motor_rl model.
%
% ASSUMPTIONS:
% - The Simulink model 'dc_motor_rl' contains a mechanism (e.g. variant
%   subsystem or controller selector parameter) to switch between
%   controller types:
%       * "LQR"
%       * "PolePlacement"
%       * "DDPG"
%       * "TD3"
%       * (optionally) "SAC"
% - The same plant, reference, and disturbance configuration is used for
%   all controllers, and key signals are logged to workspace variables or
%   a logsout object:
%       * reference speed: ref (or from logsout)
%       * motor speed:     y
%       * control effort:  u
%
% You may need to adapt:
%   - controller selection variable name (ctrlMode below),
%   - simulation time, and
%   - how outputs are accessed from the Simulink model results.

clear; clc; close all;

%% Configuration
mdl          = 'dc_motor_rl';
simTime      = 2.0;               % seconds
refProfile   = "step";            % match training, if used
controllers  = ["LQR","PolePlacement","DDPG","TD3"]; % add "SAC" if available

% Where trained RL agents were saved by train_dc_motor_rl.m
agentDir     = "savedAgents";

% Name of base workspace variable / mask parameter that selects controller
ctrlModeVar  = "ctrlMode"; % adjust to your model (e.g. mask parameter)

%% Load model and agents
load_system(mdl);

ddpgPath = fullfile(agentDir,"ddpgAgent_dc_motor.mat");
td3Path  = fullfile(agentDir,"td3Agent_dc_motor.mat");
sacPath  = fullfile(agentDir,"sacAgent_dc_motor.mat");

if isfile(ddpgPath)
    tmp = load(ddpgPath,"ddpgAgent");
    ddpgAgent = tmp.ddpgAgent;
else
    warning("DDPG agent file not found at %s", ddpgPath);
end

if isfile(td3Path)
    tmp = load(td3Path,"td3Agent");
    td3Agent = tmp.td3Agent;
else
    warning("TD3 agent file not found at %s", td3Path);
end

if isfile(sacPath)
    tmp = load(sacPath,"sacAgent");
    sacAgent = tmp.sacAgent;
    if ~ismember("SAC",controllers)
        controllers(end+1) = "SAC";
    end
else
    sacAgent = [];
end

% Put agents into base workspace so Simulink blocks can reference them
if exist("ddpgAgent","var"); assignin("base","ddpgAgent",ddpgAgent); end
if exist("td3Agent","var");  assignin("base","td3Agent",td3Agent);   end
if ~isempty(sacAgent);       assignin("base","sacAgent",sacAgent);   end

% Provide reference profile and sim time to model if it expects them
assignin("base","refProfileType",refProfile);
assignin("base","simTime",simTime);

%% Run simulations for each controller
results = struct();

for k = 1:numel(controllers)
    ctrl = controllers(k);
    fprintf("\nSimulating controller: %s\n", ctrl);

    % Set controller selection variable in base workspace
    assignin("base",ctrlModeVar,char(ctrl));

    % Run Simulink simulation
    out = sim(mdl,"StopTime",num2str(simTime));

    % Extract signals: adapt this section to your logging setup
    [t, ref, y, u] = extractSignals(out);

    % Compute metrics
    metrics = computePerformanceMetrics(t, ref, y, u);

    % Store
    results.(char(ctrl)).t       = t;
    results.(char(ctrl)).ref     = ref;
    results.(char(ctrl)).y       = y;
    results.(char(ctrl)).u       = u;
    results.(char(ctrl)).metrics = metrics;
end

%% Plot comparisons

% Step response (speed)
figure("Name","Speed Response Comparison");
for k = 1:numel(controllers)
    ctrl = controllers(k);
    r = results.(char(ctrl));
    plot(r.t, r.y, "DisplayName", ctrl); hold on;
end
plot(r.t, r.ref, "k--", "DisplayName","Reference");
xlabel("Time [s]");
ylabel("Speed");
grid on;
legend("Location","best");
title("Motor speed response for different controllers");

% Control effort
figure("Name","Control Effort Comparison");
for k = 1:numel(controllers)
    ctrl = controllers(k);
    r = results.(char(ctrl));
    plot(r.t, r.u, "DisplayName", ctrl); hold on;
end
xlabel("Time [s]");
ylabel("Control effort");
grid on;
legend("Location","best");
title("Control effort for different controllers");

% Bar charts for scalar metrics
metricNames = ["Overshoot","RiseTime","SettlingTime","IAE","ISE","ControlEnergy"];

for m = 1:numel(metricNames)
    name = metricNames(m);
    values = zeros(1,numel(controllers));
    for k = 1:numel(controllers)
        ctrl = controllers(k);
        values(k) = results.(char(ctrl)).metrics.(char(name));
    end
    figure("Name",sprintf("%s Comparison",name));
    bar(categorical(controllers), values);
    ylabel(char(name));
    title(sprintf("%s comparison",name));
    grid on;
end

fprintf("\nEvaluation completed.\n");

%% Local helpers

function [t, ref, y, u] = extractSignals(out)
%EXTRACTSIGNALS Extract reference, output, and control from simulation result.
% Adapt this to match how your model logs data.

% Example using logsout:
if isprop(out,"logsout") && ~isempty(out.logsout)
    l = out.logsout;
    t   = l.getElement("y").Values.Time;
    y   = l.getElement("y").Values.Data;
    ref = l.getElement("ref").Values.Data;
    u   = l.getElement("u").Values.Data;
else
    % Fallback: try workspace variables
    if isfield(out,"y")
        t = out.y.Time;
        y = out.y.Data;
    else
        error("Could not find signal 'y' in simulation outputs.");
    end
    if isfield(out,"ref")
        ref = out.ref.Data;
    else
        ref = zeros(size(y));
    end
    if isfield(out,"u")
        u = out.u.Data;
    else
        u = zeros(size(y));
    end
end
end

function metrics = computePerformanceMetrics(t, ref, y, u)
%COMPUTEPERFORMANCEMETRICS Basic step-response metrics and integral errors.

dt = mean(diff(t));

% Assume single-input single-output for metrics
refSig = ref(:);
ySig   = y(:);
uSig   = u(:);

% Overshoot, rise time, settling time (simple approximations)
finalValue = refSig(end);
errorSig   = refSig - ySig;

overshoot = (max(ySig) - finalValue)/max(finalValue,eps) * 100;

% Rise time: 10-90% of final value
idxRiseStart = find(ySig >= 0.1*finalValue, 1, "first");
idxRiseEnd   = find(ySig >= 0.9*finalValue, 1, "first");
if ~isempty(idxRiseStart) && ~isempty(idxRiseEnd)
    riseTime = t(idxRiseEnd) - t(idxRiseStart);
else
    riseTime = NaN;
end

% Settling time: within 2% band
idxSettle = find(abs(errorSig) <= 0.02*max(abs(finalValue),eps), 1, "last");
if ~isempty(idxSettle)
    settlingTime = t(idxSettle);
else
    settlingTime = NaN;
end

% Integral absolute error (IAE) and integral squared error (ISE)
IAE = trapz(t, abs(errorSig));
ISE = trapz(t, errorSig.^2);

% Control effort metric (energy)
controlEnergy = trapz(t, uSig.^2);

metrics = struct();
metrics.Overshoot      = overshoot;
metrics.RiseTime       = riseTime;
metrics.SettlingTime   = settlingTime;
metrics.IAE            = IAE;
metrics.ISE            = ISE;
metrics.ControlEnergy  = controlEnergy;
end

