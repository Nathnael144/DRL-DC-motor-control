clear; clc; close all;

p = dc_motor_params();
A = p.A; B = p.B; C = p.C; D = p.D;
sys = ss(A,B,C,D);

fprintf('--- Phase 1: DC motor model validation ---\n');
disp('A ='); disp(A);
disp('B ='); disp(B);
fprintf('Controllability rank = %d (expected 2)\n', rank(ctrb(A,B)));
disp('Open-loop poles ='); disp(pole(sys));

% Open-loop response to 1V step
Ts = 1e-4;
Tend = 0.5;
t = (0:Ts:Tend)';
u = ones(size(t));
omega = lsim(sys,u,t);

figure('Name','Phase 1 - Open-loop');
plot(t,omega,'LineWidth',1.6); grid on;
xlabel('Time (s)'); ylabel('\\omega (rad/s)');
title('DC motor open-loop speed response (1V step)');

save('phase1_nominal_model.mat','p','A','B','C','D');
fprintf('Saved nominal model to phase1_nominal_model.mat\n');
