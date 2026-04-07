function plot_results(results)
%PLOT_RESULTS Plot scenario-wise controller comparison.

scenarioNames = fieldnames(results);
colors = lines(6);

for i = 1:numel(scenarioNames)
    sName = scenarioNames{i};
    data = results.(sName);

    if ~isfield(data,'t') || ~isfield(data,'r')
        continue
    end

    t = data.t;
    r = data.r;

    allNames = setdiff(fieldnames(data), {'t','r','tauLoad','settings'});
    controllerNames = {};
    for j = 1:numel(allNames)
        cname = allNames{j};
        if isstruct(data.(cname)) && (isfield(data.(cname),'y') || isfield(data.(cname),'u'))
            controllerNames{end+1} = cname; %#ok<AGROW>
        end
    end

    figure('Name', ['Speed - ' sName]);
    plot(t,r,'k:','LineWidth',1.4); hold on;

    cIdx = 1;
    for j = 1:numel(controllerNames)
        cname = controllerNames{j};
        if isstruct(data.(cname)) && isfield(data.(cname),'y')
            plot(t,data.(cname).y,'LineWidth',1.5,'Color',colors(cIdx,:));
            cIdx = cIdx + 1;
        end
    end

    grid on; xlabel('Time (s)'); ylabel('\\omega (rad/s)');
    legend([{'Reference'}, controllerNames],'Location','best');
    title(['Speed tracking: ' strrep(sName,'_','\\_')]);

    figure('Name', ['Control - ' sName]);
    cIdx = 1;
    for j = 1:numel(controllerNames)
        cname = controllerNames{j};
        if isstruct(data.(cname)) && isfield(data.(cname),'u')
            plot(t,data.(cname).u,'LineWidth',1.4,'Color',colors(cIdx,:)); hold on;
            cIdx = cIdx + 1;
        end
    end

    grid on; xlabel('Time (s)'); ylabel('Armature voltage (V)');
    if ~isempty(controllerNames)
        legend(controllerNames,'Location','best');
    end
    title(['Control effort: ' strrep(sName,'_','\\_')]);
end

end
