clear; clc; close all;

thisDir = fileparts(mfilename('fullpath'));
if ~isempty(thisDir)
    addpath(thisDir);
end

fprintf('--- Phase 6: Robustness tests under parameter uncertainty ---\n');

p0 = dc_motor_params();
lqrOut = lqr_controller();
ppOut  = pole_placement_controller();
Vmax = 24;
saturateClassical = true;

Ts = 1e-4;
Tend = 0.5;
t = (0:Ts:Tend)';
[r, tauLoad] = benchmark_scenario_signals('step_load_disturbance', t);

cases = {
    'J',  [0.8 1.0 1.2];
    'Ra', [0.8 1.0 1.2];
    'Kt', [0.9 1.0 1.1];
};

records = struct('Parameter',{},'Multiplier',{},'Controller',{}, ...
    'IAE',{},'ISE',{},'SSE',{},'ControlEnergy',{},'RiseTime',{}, ...
    'SettlingTime',{},'Overshoot',{});

for c = 1:size(cases,1)
    pName = cases{c,1};
    multipliers = cases{c,2};

    for k = 1:numel(multipliers)
        mul = multipliers(k);
        p = localPerturbParam(p0,pName,mul);

        [yLQR,uLQR] = localSimulateController(p,lqrOut,t,r,tauLoad,Vmax,saturateClassical);
        mLQR = calcMetrics(t,yLQR,r,uLQR);
        records(end+1) = localBuildRecord(pName,mul,'LQR',mLQR); %#ok<SAGROW>

        [yPP,uPP] = localSimulateController(p,ppOut,t,r,tauLoad,Vmax,saturateClassical);
        mPP = calcMetrics(t,yPP,r,uPP);
        records(end+1) = localBuildRecord(pName,mul,'PolePlacement',mPP); %#ok<SAGROW>
    end
end

T = struct2table(records);

disp(T);
settings.Vmax = Vmax;
settings.saturateClassical = saturateClassical;
save('phase6_robustness_results.mat','T','records','settings');
fprintf('Saved robustness results to phase6_robustness_results.mat\n');

function rec = localBuildRecord(pName,mul,controller,m)
rec.Parameter = pName;
rec.Multiplier = mul;
rec.Controller = controller;
rec.IAE = m.IAE;
rec.ISE = m.ISE;
rec.SSE = m.SSE;
rec.ControlEnergy = m.ControlEnergy;
rec.RiseTime = m.RiseTime;
rec.SettlingTime = m.SettlingTime;
rec.Overshoot = m.Overshoot;
end

function p = localPerturbParam(p,pName,mul)
p.(pName) = p.(pName) * mul;

if strcmpi(pName,'Kt')
    p.Ke = p.Kt; % keep Ke = Kt in SI
end

p.A = [-p.Ra/p.La, -p.Ke/p.La;
        p.Kt/p.J,  -p.Bm/p.J];
p.B = [1/p.La; 0];
p.C = [0 1];
p.D = 0;
end

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
Ts = t(2)-t(1);
x = zeros(2,N);
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
