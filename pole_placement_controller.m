function out = pole_placement_controller(desiredPoles)
%POLE_PLACEMENT_CONTROLLER Design pole-placement state-feedback controller.
%   out = pole_placement_controller(desiredPoles) returns K, Nbar and
%   closed-loop system for the nominal DC motor.

p = dc_motor_params();

if nargin < 1 || isempty(desiredPoles)
    desiredPoles = [-80 -1500];
end

K = place(p.A,p.B,desiredPoles);
Acl = p.A - p.B*K;
Nbar = -1/(p.C*(Acl\p.B));

out.name = 'PolePlacement';
out.desiredPoles = desiredPoles(:).';
out.K = K;
out.Nbar = Nbar;
out.eigs = eig(Acl);
out.sys_cl = ss(Acl, p.B*Nbar, p.C, 0);

end
