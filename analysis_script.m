% analysis_script.m - Diagnose RL agent weaknesses
p = dc_motor_params();
fprintf('=== Motor dynamics analysis ===\n');
fprintf('Electrical time constant (La/Ra): %.6f ms\n', 1000*p.La/p.Ra);
fprintf('Mechanical time constant (J/Bm): %.3f s\n', p.J/p.Bm);
fprintf('Electromechanical coupling (Kt*Ke/(Ra*J)): %.3f\n', p.Kt*p.Ke/(p.Ra*p.J));
eigs_ol = eig(p.A);
fprintf('Open-loop poles: %.3f, %.3f\n', eigs_ol(1), eigs_ol(2));

out = lqr_controller();
fprintf('\nLQR gain K = [%.6f  %.6f]\n', out.K(1), out.K(2));
fprintf('LQR Nbar = %.6f\n', out.Nbar);
eigs_cl = eig(p.A - p.B*out.K);
fprintf('LQR closed-loop poles: %.3f, %.3f\n', eigs_cl(1), eigs_cl(2));
bw_cl = max(abs(eigs_cl));
fprintf('Dominant CL pole magnitude: %.3f rad/s  (%.1f Hz)\n', bw_cl, bw_cl/(2*pi));

t_test = (0:1e-4:0.5)';
r_test = 100*ones(size(t_test));
[y_lqr,~,x_lqr] = lsim(out.sys_cl, r_test, t_test);
u_lqr = (-out.K*x_lqr.' + out.Nbar*r_test.').';
fprintf('\nLQR step response:\n');
fprintf('  y(end) = %.6f (target 100)\n', y_lqr(end));
fprintf('  Max voltage: %.3f V  (Vmax=24)\n', max(abs(u_lqr)));
fprintf('  Steady-state voltage: %.3f V\n', u_lqr(end));
i10 = find(y_lqr >= 10, 1);
i90 = find(y_lqr >= 90, 1);
if ~isempty(i10) && ~isempty(i90)
    fprintf('  Rise time (10-90%%): %.4f ms\n', 1000*(t_test(i90)-t_test(i10)));
end

fprintf('\n=== Reward function analysis ===\n');
errors = [0 1 2 5 10 20 50 100];
for idx = 1:numel(errors)
    e_val = errors(idx);
    r_broad = 0.5*exp(-(e_val/50)^2);
    r_tight = 0.5*exp(-(e_val/5)^2);
    r_total = r_broad + r_tight;
    fprintf('  e=%-4d -> broad=%.4f  tight=%.4f  total=%.4f\n', e_val, r_broad, r_tight, r_total);
end

fprintf('\nPer ctrl-step max reward (e=0, 10 sub-steps): %.1f\n', 10*1.0);
fprintf('Per episode max (500 steps x 10 sub): %.1f\n', 500*10*1.0);
fprintf('Observed best avg reward: 3034.6 -> per sub-step: %.4f of 1.0\n', 3034.6/(500*10));
fprintf('Observed final avg reward: 2355.6 -> per sub-step: %.4f of 1.0\n', 2355.6/(500*10));

fprintf('\n=== Key problems identified ===\n');
fprintf('1. SSE ~22 rad/s on step_nominal: agent reaches only ~78/100 rad/s\n');
fprintf('2. Gaussian reward has zero gradient for large errors AND near-zero errors\n');
fprintf('3. No explicit penalty for steady-state error (integral term ignored in reward)\n');
fprintf('4. Observation tanh(ie/60) saturates quickly - integral info is lost\n');
fprintf('5. Training ref range 60-140 but eval uses fixed 100 - may be fine\n');
fprintf('6. Load range +/-0.015 Nm but eval uses 0.02 Nm step - out of distribution\n');
