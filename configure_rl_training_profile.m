function configure_rl_training_profile(modelName)
%CONFIGURE_RL_TRAINING_PROFILE Apply tuned RL reward/normalization settings.
%   configure_rl_training_profile('dc_motor_rl') updates model blocks to:
%   1) normalize observations before RL Agent (tanh scaling)
%   2) use a normalized/clipped reward expression
%   3) relax done-limit for current to reduce premature episode termination

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

% Base variables used in normalized observation and reward function
assignin('base','EobsNorm',120);    % rad/s error normalization
assignin('base','DEobsNorm',3000);  % rad/s^2 error-derivative normalization
assignin('base','IEobsNorm',60);    % integral-error normalization
assignin('base','WobsNorm',150);    % speed normalization

assignin('base','ErewardNorm',120);   % normalize error term
assignin('base','DErewardNorm',2500); % normalize derivative term
assignin('base','TrackTol',1.0);      % rad/s
assignin('base','TrackBonus',4.0);    % reward bonus near target

obsNormBlk = [modelName '/obs_norm_fcn'];
if getSimulinkBlockHandle(obsNormBlk) == -1
    add_block('simulink/User-Defined Functions/MATLAB Fcn', obsNormBlk, ...
        'Position',[380 165 560 205]);
end

set_param(obsNormBlk,'MATLABFcn', ...
    '[tanh(u(1)/EobsNorm); tanh(u(2)/DEobsNorm); tanh(u(3)/IEobsNorm); tanh(u(4)/WobsNorm)]');

% Rewire observation path: obs_mux -> obs_norm_fcn -> RL Agent
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

% Reward shaping (normalized + clipped penalties + tracking bonus)
rewardBlk = [modelName '/reward_fcn'];
if getSimulinkBlockHandle(rewardBlk) == -1
    error('Reward block not found: %s', rewardBlk);
end
rewardExpr = ['-( 3*(min(abs(u(1)),ErewardNorm)/ErewardNorm)^2 + ' ...
              '0.2*(min(abs(u(2)),DErewardNorm)/DErewardNorm)^2 + ' ...
              '1.5*(max(0,-u(1))/ErewardNorm)^2 + ' ...
              '0.25*(u(3)/Vmax)^2 + ' ...
              '1.2*(max(0,abs(u(4))-Imax)/Imax)^2 ) + ' ...
              'TrackBonus*(abs(u(1))<TrackTol)'];
set_param(rewardBlk,'MATLABFcn',rewardExpr);

% Relax early termination for current magnitude
if getSimulinkBlockHandle([modelName '/done_limit']) ~= -1
    set_param([modelName '/done_limit'],'Value','2.5*Imax');
end

save_system(modelName,modelFile,'OverwriteIfChangedOnDisk',true);
fprintf('Applied tuned RL profile and saved model: %s\n', modelFile);

end
