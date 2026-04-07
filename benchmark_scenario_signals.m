function [r, tauLoad] = benchmark_scenario_signals(sName, t)
%BENCHMARK_SCENARIO_SIGNALS Reference and load signals for benchmarks.
%   [r, tauLoad] = benchmark_scenario_signals(sName, t) returns the
%   reference speed r and load torque tauLoad for a given scenario name.

switch sName
    case 'step_nominal'
        r = 100*ones(size(t));
        tauLoad = zeros(size(t));

    case 'step_load_disturbance'
        r = 100*ones(size(t));
        tauLoad = zeros(size(t));
        tauLoad(t >= 0.2) = 0.02; % N.m load step

    case 'ramp'
        r = min(200*t,120); % up to 120 rad/s
        tauLoad = zeros(size(t));

    case 'sine'
        r = 100 + 20*sin(2*pi*2*t); % 2 Hz oscillation
        tauLoad = zeros(size(t));

    otherwise
        error('Unknown scenario: %s', sName);
end

end
