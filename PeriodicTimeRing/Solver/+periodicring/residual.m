function [flatResidual,diagnostics]=residual(compiled,flatState,parameters)
%RESIDUAL Assemble all local equations and optional global row replacements.

n=compiled.StateCount;grid=compiled.Grid;nodeCount=grid.NodeCount;
state=reshape(flatState,n,nodeCount);
derivative=state*grid.D.';
local=zeros(n,nodeCount,'like',state);
for node=1:nodeCount
    value=compiled.LocalResidual(grid.Time(node),state(:,node), ...
        derivative(:,node),parameters);
    value=value(:);
    if numel(value)~=n || (compiled.RuntimeChecks ...
            && (~isreal(value) || any(~isfinite(value))))
        error('periodicring:LocalResidualValue', ...
            'LocalResidual must return one finite real value per state.');
    end
    local(:,node)=value;
end
flatResidual=local(:);
globalValue=zeros(0,1);
if ~isempty(compiled.GlobalRows)
    globalValue=compiled.GlobalResidual(grid,state,parameters);
    globalValue=globalValue(:);
    if numel(globalValue)~=numel(compiled.GlobalRows) ...
            || (compiled.RuntimeChecks ...
            && (~isreal(globalValue) || any(~isfinite(globalValue))))
        error('periodicring:GlobalResidualValue', ...
            'GlobalResidual must return one finite value per GlobalRows entry.');
    end
    flatResidual(compiled.GlobalRows)=globalValue;
end
diagnostics=struct('RawLocalResidual',local, ...
    'GlobalResidual',globalValue,'State',state,'Derivative',derivative);
end
