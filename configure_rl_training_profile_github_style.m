function configure_rl_training_profile_github_style(modelName)
%CONFIGURE_RL_TRAINING_PROFILE_GITHUB_STYLE
% Apply tuned RL profile:
% - full observation: [integral_error; error; d_error; speed]
% - bipolar control [-Vmax, Vmax] (can brake like LQR/PP)
% - reward emphasizing tracking + smooth control + current safety

if nargin < 1 || isempty(modelName)
    modelName = 'dc_motor_rl';
end

thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir)
    thisDir = pwd;
end
modelFile = fullfile(thisDir,[modelName '.slx']);

if ~bdIsLoaded(modelName)
    if exist(modelFile,'file') ~= 2
        error('Model file not found: %s. Build model first.', modelFile);
    end
    load_system(modelFile);
end

% Base variables
assignin('base','EobsNorm',120);     % rad/s
assignin('base','DEobsNorm',3000);    % rad/s^2 error derivative normalization
assignin('base','IEobsNorm',60);      % integrated error
assignin('base','WobsNorm',150);      % rad/s
assignin('base','ErewardNorm',100);   % rad/s  (wide norm so max-error penalty is bounded)
assignin('base','DErewardNorm',3000); % rad/s^2
assignin('base','IErewardNorm',50);   % integral-error norm
assignin('base','TrackTol',2.0);      % rad/s  (generous initially — agent can actually earn it)
assignin('base','TrackBonus',1.0);    % tracking step bonus (small — bulk reward from exp bonus)

obsNormBlk = [modelName '/obs_norm_fcn'];
if getSimulinkBlockHandle(obsNormBlk) == -1
    add_block('simulink/User-Defined Functions/MATLAB Fcn', obsNormBlk, ...
        'Position',[380 165 560 205]);
end

% obs_mux is [error; d_error; i_error; omega]
% full observation => [i_error; error; d_error; omega]  (4D — includes derivative for damping)
set_param(obsNormBlk,'MATLABFcn', ...
    '[tanh(u(3)/IEobsNorm); tanh(u(1)/EobsNorm); tanh(u(2)/DEobsNorm); tanh(u(4)/WobsNorm)]');

% Rewire observation path safely
try
    delete_line(modelName,'obs_mux/1','RL Agent/1');
catch
end
try
    add_line(modelName,'obs_mux/1','obs_norm_fcn/1','autorouting','on');
catch
end
try
    add_line(modelName,'obs_norm_fcn/1','RL Agent/1','autorouting','on');
catch
end

% Bipolar voltage saturation — allows braking like LQR/PP
if getSimulinkBlockHandle([modelName '/voltage_sat']) ~= -1
    set_param([modelName '/voltage_sat'], 'LowerLimit', '-Vmax', 'UpperLimit', 'Vmax');
end

% Add integral error as 5th input to reward mux
rewardMuxBlk = [modelName '/reward_in_mux'];
if getSimulinkBlockHandle(rewardMuxBlk) ~= -1
    set_param(rewardMuxBlk, 'Inputs', '5');
    try
        add_line(modelName, 'i_error/1', 'reward_in_mux/5', 'autorouting', 'on');
    catch
        % line may already exist
    end
end

% Reward uses reward_in_mux = [error, d_error, voltage, ia, integral_error]
rewardBlk = [modelName '/reward_fcn'];
if getSimulinkBlockHandle(rewardBlk) == -1
    error('Reward block not found: %s', rewardBlk);
end
% --- Reward design (bipolar action, learnable from random policy) ---
% u(1)=error  u(2)=d_error  u(3)=voltage  u(4)=ia  u(5)=integral_error
%
% Design constraint: per-step reward must be in [-1.2, +3.0] range so that
% over 5000 steps the episode reward is [-6000, +15000].
% Random policy (error~100): -(1.0 + 0.02 + 0.1) + 0.27 = -0.85/step = -4250
% Moderate tracking (error~20): -(0.04 + 0.02 + 0.1) + 1.64 = +1.48/step = +7400
% Perfect tracking (error~0): -(0 + 0 + 0.02 + 0) + 2.0 + 1.0 = +2.98/step = +14900
% This gives clear gradient: any reduction in error immediately improves reward.
rewardExpr = ['-( 1.0*(min(abs(u(1)),ErewardNorm)/ErewardNorm)^2 + ' ...
              '0.02*(min(abs(u(2)),DErewardNorm)/DErewardNorm)^2 + ' ...
              '0.05*(u(3)/Vmax)^2 + ' ...
              '0.5*(max(0,abs(u(4))-Imax)/Imax)^2 + ' ...
              '0.1*(min(abs(u(5)),IErewardNorm)/IErewardNorm)^2 ) + ' ...
              '2.0*exp(-3*(min(abs(u(1)),ErewardNorm)/ErewardNorm)) + ' ...
              'TrackBonus*(abs(u(1))<TrackTol)'];
set_param(rewardBlk,'MATLABFcn',rewardExpr);

% Relax done threshold for learning
if getSimulinkBlockHandle([modelName '/done_limit']) ~= -1
    set_param([modelName '/done_limit'],'Value','3.5*Imax');
end

save_system(modelName,modelFile,'OverwriteIfChangedOnDisk',true);
fprintf('Applied GitHub-style RL profile and saved model: %s\n', modelFile);

end
