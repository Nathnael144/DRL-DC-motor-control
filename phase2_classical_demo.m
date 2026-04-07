clear; clc; close all;

fprintf('--- Phase 2: LQR vs Pole Placement ---\n');

w_ref = 100; % rad/s
Ts = 1e-4;
Tend = 0.3;
t = (0:Ts:Tend)';
r = w_ref*ones(size(t));

lqrOut = lqr_controller();
ppOut  = pole_placement_controller();

[yLQR,~,xLQR] = lsim(lqrOut.sys_cl,r,t);
uLQR = (-lqrOut.K*xLQR.' + lqrOut.Nbar*r.').';

[yPP,~,xPP] = lsim(ppOut.sys_cl,r,t);
uPP = (-ppOut.K*xPP.' + ppOut.Nbar*r.').';

mLQR = calcMetrics(t,yLQR,r,uLQR);
mPP  = calcMetrics(t,yPP,r,uPP);

figure('Name','Phase 2 - Speed response');
plot(t,yLQR,'b','LineWidth',1.6); hold on;
plot(t,yPP,'r--','LineWidth',1.6);
plot(t,r,'k:','LineWidth',1.2);
grid on; xlabel('Time (s)'); ylabel('\\omega (rad/s)');
legend('LQR','Pole Placement','Reference','Location','best');
title('Phase 2: Closed-loop speed comparison');

figure('Name','Phase 2 - Control effort');
plot(t,uLQR,'b','LineWidth',1.4); hold on;
plot(t,uPP,'r--','LineWidth',1.4);
grid on; xlabel('Time (s)'); ylabel('Armature voltage (V)');
legend('LQR','Pole Placement','Location','best');
title('Control effort comparison');

disp('LQR metrics:'); disp(mLQR);
disp('Pole Placement metrics:'); disp(mPP);

results.phase2.t = t;
results.phase2.r = r;
results.phase2.LQR.y = yLQR;
results.phase2.LQR.u = uLQR;
results.phase2.LQR.metrics = mLQR;
results.phase2.LQR.K = lqrOut.K;
results.phase2.PP.y = yPP;
results.phase2.PP.u = uPP;
results.phase2.PP.metrics = mPP;
results.phase2.PP.K = ppOut.K;

save('phase2_results.mat','results','lqrOut','ppOut');
fprintf('Saved Phase 2 results to phase2_results.mat\n');
