function compiled=compile(problem,grid)
%COMPILE Validate a periodic residual problem and cache pattern coloring.

requiredGrid={'NodeCount','Period','Origin','Time','D','Weight'};
for index=1:numel(requiredGrid)
    if ~isfield(grid,requiredGrid{index})
        error('periodicring:Grid','Grid is missing %s.',requiredGrid{index});
    end
end
if ~isfield(problem,'StateCount')
    error('periodicring:StateCount','problem.StateCount is required.');
end
validateattributes(problem.StateCount,{'numeric'}, ...
    {'scalar','integer','positive','finite'});
stateCount=problem.StateCount;
if ~isfield(problem,'LocalResidual') || ~isa(problem.LocalResidual,'function_handle')
    error('periodicring:LocalResidual', ...
        'problem.LocalResidual must be a function handle.');
end
if ~isfield(problem,'StatePattern') || isempty(problem.StatePattern)
    statePattern=sparse(true(stateCount));
else
    statePattern=problem.StatePattern;
end
if ~isfield(problem,'DerivativePattern') || isempty(problem.DerivativePattern)
    derivativePattern=sparse(true(stateCount));
else
    derivativePattern=problem.DerivativePattern;
end
globalRows=zeros(0,1);globalPattern=[];globalResidual=[];
if isfield(problem,'GlobalRows'),globalRows=problem.GlobalRows(:);end
if isfield(problem,'GlobalPattern'),globalPattern=problem.GlobalPattern;end
if isfield(problem,'GlobalResidual'),globalResidual=problem.GlobalResidual;end
if ~isempty(globalRows) && ~isa(globalResidual,'function_handle')
    error('periodicring:GlobalResidual', ...
        'A GlobalResidual callback is required for GlobalRows.');
end
pattern=periodicring.buildJacobianPattern(statePattern, ...
    derivativePattern,grid,globalRows,globalPattern);
[colors,colorCount]=periodicring.greedyColoring(pattern);
unknownCount=stateCount*grid.NodeCount;

newtonDefaults=struct('Colors',colors);
mapping={'StateScale','VariableScale';'ResidualScale','ResidualScale'; ...
    'LowerBound','LowerBound';'UpperBound','UpperBound'};
for index=1:size(mapping,1)
    source=mapping{index,1};target=mapping{index,2};
    if isfield(problem,source) && ~isempty(problem.(source))
        newtonDefaults.(target)=expandRing(problem.(source), ...
            stateCount,grid.NodeCount,source);
    end
end
validator=[];
if isfield(problem,'ValidateState') && ~isempty(problem.ValidateState)
    if ~isa(problem.ValidateState,'function_handle')
        error('periodicring:ValidateState', ...
            'ValidateState must be a function handle.');
    end
    validator=problem.ValidateState;
end
parameters=[];if isfield(problem,'Parameters'),parameters=problem.Parameters;end
runtimeChecks=false;
if isfield(problem,'RuntimeChecks')
    validateattributes(problem.RuntimeChecks,{'logical','numeric'},{'scalar'});
    runtimeChecks=logical(problem.RuntimeChecks);
end
compiled=struct('StateCount',stateCount,'UnknownCount',unknownCount, ...
    'Grid',grid,'LocalResidual',problem.LocalResidual, ...
    'GlobalRows',globalRows,'GlobalResidual',globalResidual, ...
    'Pattern',pattern,'Colors',colors,'ColorCount',colorCount, ...
    'NewtonDefaults',newtonDefaults,'ValidateState',validator, ...
    'Parameters',parameters,'RuntimeChecks',runtimeChecks);
end

function expanded=expandRing(value,stateCount,nodeCount,name)
value=value(:);
if isscalar(value)
    expanded=repmat(value,stateCount*nodeCount,1);
elseif numel(value)==stateCount
    expanded=repmat(value,nodeCount,1);
elseif numel(value)==stateCount*nodeCount
    expanded=value;
else
    error('periodicring:ProblemOptionSize', ...
        '%s must be scalar, per-state, or per-unknown.',name);
end
end
