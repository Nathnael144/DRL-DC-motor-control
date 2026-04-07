function m = calcMetrics(t,y,r,u)
%CALCMETRICS Compute control performance metrics.

% Force column vectors
t = t(:);
y = y(:);
r = r(:);
u = u(:);

e = r - y;

m.IAE = trapz(t,abs(e));
m.ISE = trapz(t,e.^2);
m.ControlEnergy = trapz(t,u.^2);
m.SSE = abs(e(end));

if all(abs(r - r(1)) < 1e-9)
    info = stepinfo(y,t,r(1));
    m.RiseTime = info.RiseTime;
    m.SettlingTime = info.SettlingTime;
    m.Overshoot = info.Overshoot;
else
    m.RiseTime = NaN;
    m.SettlingTime = NaN;
    m.Overshoot = NaN;
end

end
