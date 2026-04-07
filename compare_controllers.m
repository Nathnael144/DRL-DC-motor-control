clear; clc; close all;

thisDir = fileparts(mfilename('fullpath'));
if ~isempty(thisDir)
    addpath(thisDir);
end

fprintf('--- Phase 5: Comparative simulation ---\n');

p = dc_motor_params();
lqrOut = lqr_controller();
ppOut  = pole_placement_controller();
Vmax = 24;
saturateClassical = true;

Ts = 1e-4;
Tend = 0.5;
t = (0:Ts:Tend)';

scenarioNames = {'step_nominal','step_load_disturbance','ramp','sine'};
results = struct();

for i = 1:numel(scenarioNames)
    sName = scenarioNames{i};

    [r, tauLoad] = benchmark_scenario_signals(sName, t);

    [yLQR,uLQR] = localSimulateController(p,lqrOut,t,r,tauLoad,Vmax,saturateClassical);
    [yPP,uPP]   = localSimulateController(p,ppOut,t,r,tauLoad,Vmax,saturateClassical);

    results.(sName).t = t;
    results.(sName).r = r;
    results.(sName).tauLoad = tauLoad;
    results.(sName).settings.Vmax = Vmax;
    results.(sName).settings.saturateClassical = saturateClassical;

    results.(sName).LQR.y = yLQR;
    results.(sName).LQR.u = uLQR;
    results.(sName).LQR.metrics = calcMetrics(t,yLQR,r,uLQR);

    results.(sName).PP.y = yPP;
    results.(sName).PP.u = uPP;
    results.(sName).PP.metrics = calcMetrics(t,yPP,r,uPP);

    % Optional RL data file per scenario: rl_<scenario>.mat
    % Require r_rl and tau_rl so we can verify the RL scenario matches.
    rlFile = ['rl_' sName '.mat'];
    if exist(rlFile,'file') == 2
        rlData = load(rlFile);
        req = {'t_rl','y_rl','u_rl','r_rl','tau_rl'};
        if all(isfield(rlData, req))
            tRL = rlData.t_rl(:);
            yRL = interp1(tRL, rlData.y_rl(:), t, 'linear', 'extrap');
            uRL = interp1(tRL, rlData.u_rl(:), t, 'linear', 'extrap');
            rRL = interp1(tRL, rlData.r_rl(:), t, 'linear', 'extrap');
            tauRL = interp1(tRL, rlData.tau_rl(:), t, 'linear', 'extrap');

            [matchOK, matchMsg] = localScenarioMatch(r, tauLoad, rRL, tauRL);
            if ~matchOK
                warning(['RL file %s scenario mismatch: %s. ' ...
                         'Regenerate RL scenario data for strict comparability.'], ...
                         rlFile, matchMsg);
            else
                results.(sName).RL.y = yRL;
                results.(sName).RL.u = uRL;
                results.(sName).RL.metrics = calcMetrics(t,yRL,r,uRL);
            end
        else
            warning('File %s missing required fields (t_rl/y_rl/u_rl/r_rl/tau_rl). Skipping RL for this scenario.', rlFile);
        end
    end
end

save('phase5_comparison_results.mat','results');
plot_results(results);

fprintf('Saved comparison results to phase5_comparison_results.mat\n');

% Print quick summary for nominal step
disp('Nominal step metrics:');
disp(results.step_nominal);

function [y,u] = localSimulateController(p,ctrl,t,r,tauLoad,Vmax,saturateControl)
if nargin < 7
    saturateControl = false;
end

if ~saturateControl
    Acl = p.A - p.B*ctrl.K;
    E = [0; -1/p.J];
    Bcl = [p.B*ctrl.Nbar, E];
    Ccl = p.C;
    Dcl = [0 0];
    sys_cl = ss(Acl,Bcl,Ccl,Dcl);

    inputs = [r(:), tauLoad(:)];
    [y,~,x] = lsim(sys_cl,inputs,t);
    u = (-ctrl.K*x.' + ctrl.Nbar*r.').';
    return;
end

N = numel(t);
Ts = t(2) - t(1);
x = zeros(2,N); % [ia; omega]
y = zeros(N,1);
u = zeros(N,1);

for k = 1:N
    xk = x(:,k);
    y(k) = xk(2);

    uUnsat = -ctrl.K*xk + ctrl.Nbar*r(k);
    uk = min(max(uUnsat,-Vmax),Vmax);
    u(k) = uk;

    if k < N
        x(:,k+1) = localRK4Step(xk,uk,tauLoad(k),p,Ts);
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

function [ok,msg] = localScenarioMatch(r, tauLoad, rRL, tauRL)
tol = 1e-6;
dr = max(abs(r - rRL));
dtau = max(abs(tauLoad - tauRL));

ok = (dr <= tol) && (dtau <= tol);

if ok
    msg = '';
elseif dr > tol && dtau > tol
    msg = sprintf('reference and disturbance differ (max |dr|=%.3g, max |dtau|=%.3g)', dr, dtau);
elseif dr > tol
    msg = sprintf('reference differs (max |dr|=%.3g)', dr);
else
    msg = sprintf('disturbance differs (max |dtau|=%.3g)', dtau);
end
end
