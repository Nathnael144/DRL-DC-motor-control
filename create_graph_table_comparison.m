clear; clc; close all;

thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir)
    thisDir = pwd;
end
oldDir = pwd;
cleanupObj = onCleanup(@() cd(oldDir)); %#ok<NASGU>
cd(thisDir);
addpath(thisDir);

resultsFile = fullfile(thisDir,'phase5_comparison_results.mat');
if exist(resultsFile,'file') ~= 2
    fprintf('phase5_comparison_results.mat not found. Running compare_controllers.m first...\n');
    run('compare_controllers.m');
end

S = load(resultsFile);
if ~isfield(S,'results')
    error('File %s does not contain variable "results".', resultsFile);
end
results = S.results;

scenarioNames = fieldnames(results);
controllers = {'RL','LQR','PP'};
metrics = {'IAE','ISE','SSE','ControlEnergy','RiseTime','SettlingTime','Overshoot'};

records = struct('Scenario',{},'Controller',{}, ...
    'IAE',{},'ISE',{},'SSE',{},'ControlEnergy',{}, ...
    'RiseTime',{},'SettlingTime',{},'Overshoot',{});

for i = 1:numel(scenarioNames)
    sName = scenarioNames{i};
    data = results.(sName);

    if ~isfield(data,'t') || ~isfield(data,'r')
        continue
    end

    t = data.t(:);
    r = data.r(:);

    % --------- Graph 1: Speed tracking ---------
    f1 = figure('Name',['Speed_' sName],'Color','w');
    plot(t,r,'k:','LineWidth',1.5); hold on;

    legendEntries = {'Reference'};
    for c = 1:numel(controllers)
        cName = controllers{c};
        if isfield(data,cName) && isstruct(data.(cName)) && isfield(data.(cName),'y')
            plot(t,data.(cName).y(:),'LineWidth',1.5);
            legendEntries{end+1} = cName; %#ok<SAGROW>
        end
    end

    grid on;
    xlabel('Time (s)'); ylabel('\omega (rad/s)');
    title(['Speed Tracking - ' strrep(sName,'_','\_')]);
    legend(legendEntries,'Location','best');
    exportgraphics(f1, fullfile(thisDir, ['fig_speed_' sName '.png']), 'Resolution', 300);

    % --------- Graph 2: Control effort ---------
    f2 = figure('Name',['Control_' sName],'Color','w');
    hold on;
    legendEntries = {};

    for c = 1:numel(controllers)
        cName = controllers{c};
        if isfield(data,cName) && isstruct(data.(cName)) && isfield(data.(cName),'u')
            plot(t,data.(cName).u(:),'LineWidth',1.5);
            legendEntries{end+1} = cName; %#ok<SAGROW>
        end
    end

    grid on;
    xlabel('Time (s)'); ylabel('Armature voltage (V)');
    title(['Control Effort - ' strrep(sName,'_','\_')]);
    if ~isempty(legendEntries)
        legend(legendEntries,'Location','best');
    end
    exportgraphics(f2, fullfile(thisDir, ['fig_control_' sName '.png']), 'Resolution', 300);

    % --------- Metrics records ---------
    for c = 1:numel(controllers)
        cName = controllers{c};
        rec = localEmptyRecord(sName, cName);

        if isfield(data,cName) && isstruct(data.(cName)) && isfield(data.(cName),'metrics')
            m = data.(cName).metrics;
            for k = 1:numel(metrics)
                mName = metrics{k};
                if isfield(m,mName)
                    rec.(mName) = m.(mName);
                end
            end
        end

        records(end+1) = rec; %#ok<SAGROW>
    end
end

if isempty(records)
    error('No records were generated. Check phase5_comparison_results.mat structure.');
end

T_long = struct2table(records);
T_long = sortrows(T_long, {'Scenario','Controller'});
writetable(T_long, fullfile(thisDir,'comparison_metrics_long.csv'));

% --------- Wide table (easy for paper/report) ---------
scenarioList = unique(string(T_long.Scenario), 'stable');
T_wide = table(scenarioList, 'VariableNames', {'Scenario'});

for k = 1:numel(metrics)
    mName = metrics{k};
    for c = 1:numel(controllers)
        cName = controllers{c};
        col = nan(numel(scenarioList),1);
        for i = 1:numel(scenarioList)
            idx = strcmp(T_long.Scenario, scenarioList(i)) & strcmp(T_long.Controller, cName);
            if any(idx)
                col(i) = T_long{find(idx,1,'first'), mName};
            end
        end
        T_wide.([cName '_' mName]) = col;
    end
end

writetable(T_wide, fullfile(thisDir,'comparison_metrics_wide.csv'));

% --------- RL improvement table (lower metric is better) ---------
improveMetrics = {'IAE','ISE','SSE','ControlEnergy','RiseTime','SettlingTime','Overshoot'};
T_imp = table(scenarioList, 'VariableNames', {'Scenario'});
for k = 1:numel(improveMetrics)
    mName = improveMetrics{k};

    rl = T_wide.(['RL_' mName]);
    lq = T_wide.(['LQR_' mName]);
    pp = T_wide.(['PP_' mName]);

    T_imp.(['RL_vs_LQR_' mName '_pct']) = localPctImprove(lq, rl);
    T_imp.(['RL_vs_PP_'  mName '_pct']) = localPctImprove(pp, rl);
end
writetable(T_imp, fullfile(thisDir,'comparison_improvement_pct.csv'));

save(fullfile(thisDir,'comparison_tables.mat'),'T_long','T_wide','T_imp');

fprintf('\nGenerated comparison artifacts:\n');
fprintf('  - comparison_metrics_long.csv\n');
fprintf('  - comparison_metrics_wide.csv\n');
fprintf('  - comparison_improvement_pct.csv\n');
fprintf('  - comparison_tables.mat\n');
fprintf('  - fig_speed_<scenario>.png\n');
fprintf('  - fig_control_<scenario>.png\n\n');

disp('=== Long metrics table (first rows) ===');
disp(head(T_long));

function rec = localEmptyRecord(sName,cName)
rec.Scenario = char(sName);
rec.Controller = char(cName);
rec.IAE = NaN;
rec.ISE = NaN;
rec.SSE = NaN;
rec.ControlEnergy = NaN;
rec.RiseTime = NaN;
rec.SettlingTime = NaN;
rec.Overshoot = NaN;
end

function pct = localPctImprove(base, candidate)
% Positive means candidate (RL) is better (smaller metric) than base.
pct = nan(size(base));
valid = isfinite(base) & isfinite(candidate) & abs(base) > eps;
pct(valid) = 100*(base(valid) - candidate(valid))./base(valid);
end
