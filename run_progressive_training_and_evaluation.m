clear; clc; close all;

thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir)
    thisDir = pwd;
end
oldDir = pwd;
cleanupObj = onCleanup(@() cd(oldDir)); %#ok<NASGU>
cd(thisDir);
addpath(thisDir);

% -------- User-adjustable knobs (long training profile) --------
maxEpisodes = struct('ddpg',250,'td3',1000,'sac',2000);
showTrainingMonitor = true;
doPostEvaluation = true;
startFromTrainedDDPG = true;  % true: skip DDPG stage and start from TD3/SAC
startFromTrainedTD3 = true;   % true: skip DDPG+TD3 stages and start from SAC
algorithmOrder = ["ddpg","td3","sac"];
runSmokeCheck = false;  % set true only when debugging episode progression
smokeEpisodes = 10;
% --------------------------------------------------------------

if startFromTrainedTD3
    % Skip both DDPG and TD3, go straight to SAC.
    algorithmOrder = ["sac"];
elseif startFromTrainedDDPG
    ddpgWarmStartFile = fullfile(thisDir,'trained_rl_agent_github_style.mat');
    if exist(ddpgWarmStartFile,'file') == 2
        algorithmOrder = ["td3","sac"];
    else
        warning(['startFromTrainedDDPG=true, but no trained DDPG model found at\n  %s\n', ...
                 'Falling back to full order: DDPG -> TD3 -> SAC.'], ddpgWarmStartFile);
    end
end

if showTrainingMonitor && ~(usejava('desktop') && usejava('awt'))
    error(['This launcher is configured for non-headless training with visible ', ...
           'Training Monitor. Run it in MATLAB Desktop (GUI) and retry.']);
end

fprintf('=== One-click progressive RL training + evaluation ===\n');
fprintf('Episode plan: DDPG=%d, TD3=%d, SAC=%d\n', ...
    maxEpisodes.ddpg, maxEpisodes.td3, maxEpisodes.sac);
fprintf('Training monitor: %s\n', string(showTrainingMonitor));
fprintf('Post evaluation: %s\n', string(doPostEvaluation));
fprintf('Start from trained DDPG: %s\n', string(startFromTrainedDDPG));
fprintf('Start from trained TD3: %s\n', string(startFromTrainedTD3));
fprintf('Algorithm order: %s\n', strjoin(cellstr(upper(algorithmOrder)), ' -> '));
fprintf('Smoke check enabled: %s\n', string(runSmokeCheck));

if runSmokeCheck
    fprintf(['\nPre-check: running a short GitHub-style DDPG smoke test ', ...
             '(%d episodes) to confirm episode counter progression...\n'], smokeEpisodes);
    train_rl_agent_github_style(smokeEpisodes, showTrainingMonitor);
    fprintf('Pre-check complete. Starting full progressive run...\n');
end

results = train_rl_agent_progressive( ...
    maxEpisodes, showTrainingMonitor, doPostEvaluation, algorithmOrder);

fprintf('\n=== Progressive pipeline complete ===\n');
if ~(isstruct(results) && isfield(results,'bestAlgorithm') && isfield(results,'bestAgentFile'))
    summaryFile = fullfile(thisDir,'trained_rl_agent_progressive_summary.mat');
    if exist(summaryFile,'file') == 2
        S = load(summaryFile,'results');
        if isfield(S,'results') && isstruct(S.results) && ...
                isfield(S.results,'bestAlgorithm') && isfield(S.results,'bestAgentFile')
            results = S.results;
        end
    end
end

if isstruct(results) && isfield(results,'bestAlgorithm') && isfield(results,'bestAgentFile')
    fprintf('Best algorithm: %s\n', results.bestAlgorithm);
    fprintf('Best agent file: %s\n', results.bestAgentFile);
else
    warning(['Progressive run finished, but best-agent summary fields were not found in ', ...
             'returned results or summary file.']);
end
