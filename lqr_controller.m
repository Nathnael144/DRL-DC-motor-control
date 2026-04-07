function out = lqr_controller(Q,R)
%LQR_CONTROLLER Design LQR state-feedback for the nominal DC motor.
%   out = lqr_controller(Q,R) returns gains, closed-loop model and
%   feedforward prefilter Nbar for speed tracking.

p = dc_motor_params();

if nargin < 1 || isempty(Q)
    Q = diag([0.05 40]); % [current speed]
end
if nargin < 2 || isempty(R)
    R = 0.01;
end

[K,S,e] = lqr(p.A,p.B,Q,R);
Acl = p.A - p.B*K;
Nbar = -1/(p.C*(Acl\p.B));

out.name = 'LQR';
out.Q = Q;
out.R = R;
out.K = K;
out.S = S;
out.eigs = e;
out.Nbar = Nbar;
out.sys_cl = ss(Acl, p.B*Nbar, p.C, 0);

end
