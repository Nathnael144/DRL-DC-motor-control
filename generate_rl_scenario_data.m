function T = generate_rl_scenario_data(agentFile)
%GENERATE_RL_SCENARIO_DATA Evaluate trained RL agent on benchmark scenarios.
%   T = generate_rl_scenario_data() simulates the trained RL policy on
%   four scenarios and writes files expected by compare_controllers.m:
%       rl_step_nominal.mat
%       rl_step_load_disturbance.mat
%       rl_ramp.mat
%       rl_sine.mat
%
%   Output table T contains key metrics (IAE, ISE, SSE, control energy,
%   rise/settling/overshoot where applicable).

thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir)
    thisDir = pwd;
end
oldDir = pwd;
cleanupObj = onCleanup(@() cd(oldDir)); %#ok<NASGU>
cd(thisDir);
addpath(thisDir);

if nargin < 1 || isempty(agentFile)
    candidates = { ...
        fullfile(thisDir,'trained_rl_agent_tuned.mat'), ...
        fullfile(thisDir,'trained_rl_agent.mat')};
    trainedFile = '';
    for i = 1:numel(candidates)
        if exist(candidates{i},'file') == 2
            trainedFile = candidates{i};
            break;
        end
    end
else
    trainedFile = agentFile;
end

if isempty(trainedFile) || exist(trainedFile,'file') ~= 2
    error(['Missing trained agent file. Provide agentFile or create one of:\n', ...
           '  %s\n  %s'], ...
           fullfile(thisDir,'trained_rl_agent_tuned.mat'), ...
           fullfile(thisDir,'trained_rl_agent.mat'));
end

S = load(trainedFile,'agent');
agent = S.agent;
fprintf('Using RL agent file: %s\n', trainedFile);

obsDim = 4;
try
    obsDim = agent.ObservationInfo.Dimension(1);
catch
end

obsMode = localDetectObsMode(agent, obsDim);

actLower = -1;
actUpper = 1;
try
    al = agent.ActionInfo.LowerLimit;
    au = agent.ActionInfo.UpperLimit;
    if ~isempty(al), actLower = double(al(1)); end
    if ~isempty(au), actUpper = double(au(1)); end
catch
end

p = dc_motor_params();
Vmax = 24;
Ts = 1e-4;
Tend = 0.5;
t = (0:Ts:Tend)';

scenarioNames = {'step_nominal','step_load_disturbance','ramp','sine'};
records = struct('Scenario',{},'IAE',{},'ISE',{},'SSE',{}, ...
    'ControlEnergy',{},'RiseTime',{},'SettlingTime',{},'Overshoot',{});

for i = 1:numel(scenarioNames)
    sName = scenarioNames{i};
    [r, tauLoad] = benchmark_scenario_signals(sName, t);

    agent = localResetAgent(agent);
    [y, u, x] = localSimulateRLPolicy(agent,p,t,r,tauLoad,Vmax,Ts,obsDim,actLower,actUpper,obsMode);

    t_rl = t; %#ok<NASGU>
    y_rl = y; %#ok<NASGU>
    u_rl = u; %#ok<NASGU>
    r_rl = r; %#ok<NASGU>
    tau_rl = tauLoad; %#ok<NASGU>
    x_rl = x; %#ok<NASGU>

    outFile = fullfile(thisDir, ['rl_' sName '.mat']);
    save(outFile,'t_rl','y_rl','u_rl','r_rl','tau_rl','x_rl');

    m = calcMetrics(t,y,r,u);
    records(end+1) = localBuildRecord(sName,m); %#ok<SAGROW>

    fprintf('Saved %s\n', outFile);
end

T = struct2table(records);
disp(T);

save(fullfile(thisDir,'rl_scenario_metrics.mat'),'T','records');
fprintf('Saved RL metrics to %s\n', fullfile(thisDir,'rl_scenario_metrics.mat'));

end

function rec = localBuildRecord(sName,m)
rec.Scenario = sName;
rec.IAE = m.IAE;
rec.ISE = m.ISE;
rec.SSE = m.SSE;
rec.ControlEnergy = m.ControlEnergy;
rec.RiseTime = m.RiseTime;
rec.SettlingTime = m.SettlingTime;
rec.Overshoot = m.Overshoot;
end

function agentOut = localResetAgent(agentIn)
agentOut = agentIn;
try
    tmp = reset(agentIn);
    if ~isempty(tmp)
        agentOut = tmp;
    end
catch
    try
        reset(agentOut);
    catch
        % ignore if no reset method
    end
end
end

function [y,u,x] = localSimulateRLPolicy(agent,p,t,r,tauLoad,Vmax,Ts,obsDim,actLower,actUpper,obsMode)
N = numel(t);
x = zeros(2,N);  % [ia; omega]
y = zeros(N,1);
u = zeros(N,1);

ie = 0;
ePrev = r(1) - x(2,1);
ePrevCtrl = ePrev;
deCtrl = 0;
ctrlInterval = 10; % match training: Ts=1e-4 and agent acts every 1 ms
uk = 0;

for k = 1:N
    xk = x(:,k);
    ia = xk(1);
    omega = xk(2);
    y(k) = omega;

    e = r(k) - omega;
    ie = ie + e*Ts;

    % Act only every ctrlInterval (hold u in between), matching training.
    if mod(k-1, ctrlInterval) == 0
        deCtrl = (e - ePrevCtrl) / (Ts * ctrlInterval);
        ePrevCtrl = e;

        if strcmp(obsMode,'legacy3')
            % GitHub-style / DDPG-style compact observation
            obs = [ie; e; omega];
        elseif strcmp(obsMode,'norm4')
            % TD3 deep: normalized [ie, e, de, omega]
            obs = [tanh(ie/60); tanh(e/120); tanh(deCtrl/3000); omega/150];
        elseif strcmp(obsMode,'legacy4')
            % Legacy 4D observation [e; de; ie; omega]
            obs = [e; deCtrl; ie; omega];
        else
            % Current tuned TD3 observation (6 dims):
            % [tanh(ie/500); tanh(e/120); tanh(de_ctrl/4000); omega/200; tanh(ia/Imax); w_ref/200]
            Imax = 8; % keep consistent with training script defaults
            obs = [tanh(ie/500); tanh(e/120); tanh(deCtrl/4000); omega/200; tanh(ia/Imax); r(k)/200];
        end

        a = localGetAgentAction(agent,obs);
        a = min(max(a,actLower),actUpper);

        if actLower >= 0
            % Positive-only action agents (e.g., [0,1] -> [0,Vmax])
            denom = (actUpper - actLower);
            if abs(denom) < eps
                aNorm = 0;
            else
                aNorm = (a - actLower)/denom;
            end
            aNorm = min(max(aNorm,0),1);
            uk = Vmax*aNorm;
        else
            % Symmetric action agents (e.g., [-1,1] -> [-Vmax,Vmax])
            uk = Vmax*a;
        end
    end
    u(k) = uk;

    if k < N
        x(:,k+1) = localRK4Step(xk,uk,tauLoad(k),p,Ts);
    end

    ePrev = e;
end

function obsMode = localDetectObsMode(agent, obsDim)
obsMode = 'legacy4';
if obsDim == 3
    obsMode = 'legacy3';
    return;
end
if obsDim >= 6
    obsMode = 'tuned6';
    return;
end

desc = '';
try
    desc = agent.ObservationInfo.Description;
catch
end
descLower = lower(string(desc));

if obsDim == 4 && (contains(descLower,'normalized') || contains(descLower,'tanh') || ...
        contains(descLower,'ie, e, de, omega'))
    obsMode = 'norm4';
else
    obsMode = 'legacy4';
end
end

end

function xNext = localRK4Step(x,u,tau,p,Ts)
f = @(xx) localMotorDyn(xx,u,tau,p);
k1 = f(x);
k2 = f(x + 0.5*Ts*k1);
k3 = f(x + 0.5*Ts*k2);
k4 = f(x + Ts*k3);
xNext = x + (Ts/6)*(k1 + 2*k2 + 2*k3 + k4);
end

function dx = localMotorDyn(x,u,tau,p)
ia = x(1);
omega = x(2);

dia = -(p.Ra/p.La)*ia - (p.Ke/p.La)*omega + (1/p.La)*u;
domega = (p.Kt/p.J)*ia - (p.Bm/p.J)*omega - (1/p.J)*tau;

dx = [dia; domega];
end

function a = localGetAgentAction(agent,obs)
try
    raw = getAction(agent,obs);
catch
    raw = getAction(agent,{obs});
end

if iscell(raw)
    raw = raw{1};
end

if isstruct(raw)
    fn = fieldnames(raw);
    raw = raw.(fn{1});
    if iscell(raw)
        raw = raw{1};
    end
end

if isa(raw,'dlarray')
    raw = extractdata(raw);
end

if isa(raw,'gpuArray')
    raw = gather(raw);
end

raw = double(raw);
if isempty(raw) || ~isfinite(raw(1))
    a = 0;
else
    a = raw(1);
end
end
