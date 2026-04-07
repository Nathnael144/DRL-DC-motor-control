function p = dc_motor_params()
%DC_MOTOR_PARAMS Motor parameters and nominal state-space matrices.
%   p = dc_motor_params() returns a struct with the motor parameters from
%   the provided specification table and the nominal state-space model:
%       x = [ia; omega], u = Va, y = omega.
%
%   Parameters:
%       La = 0.58e-3 H
%       Ra = 2.59 ohm
%       J  = 5.69e-4 kg.m^2
%       Bm = 1e-6 N.m.s/rad
%       Kt = 28.6e-3 N.m/A
%       Ke = Kt (SI units)

p.La = 0.58e-3;
p.Ra = 2.59;
p.J  = 5.69e-4;
p.Bm = 1e-6;
p.Kt = 28.6e-3;
p.Ke = p.Kt;

p.A = [-p.Ra/p.La, -p.Ke/p.La;
        p.Kt/p.J,  -p.Bm/p.J];
p.B = [1/p.La; 0];
p.C = [0 1];
p.D = 0;

end
