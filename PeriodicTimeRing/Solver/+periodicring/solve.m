function result=solve(compiled,initialGuess,options,parameters)
%SOLVE Solve a compiled periodic ODE/DAE residual problem.

if nargin<3,options=struct();end
if nargin<4,parameters=compiled.Parameters;end
grid=compiled.Grid;n=compiled.StateCount;nodeCount=grid.NodeCount;
if isa(initialGuess,'function_handle')
    initialGuess=initialGuess(grid.Time);
end
if isvector(initialGuess) && n==1 && numel(initialGuess)==nodeCount
    initialGuess=initialGuess(:).';
end
if ~isnumeric(initialGuess) || ~isequal(size(initialGuess),[n,nodeCount]) ...
        || ~isreal(initialGuess) || any(~isfinite(initialGuess),'all')
    error('periodicring:InitialGuess', ...
        'InitialGuess must be a finite real stateCount-by-nodeCount matrix.');
end
options=mergeOptions(compiled.NewtonDefaults,options);
if ~isempty(compiled.ValidateState)
    validator=compiled.ValidateState;
    options.Feasible=@(flat)all(validator( ...
        grid,reshape(flat,n,nodeCount),parameters));
end
residualFunction=@(flat)periodicring.residual(compiled,flat,parameters);
[flat,info]=periodicring.solveNewton(residualFunction, ...
    initialGuess(:),compiled.Pattern,options);
state=reshape(flat,n,nodeCount);
derivative=periodicring.differentiate(state,grid);
[flatResidual,residualDiagnostics]=periodicring.residual( ...
    compiled,flat,parameters);
residual=reshape(flatResidual,n,nodeCount);
result=struct('Grid',grid,'State',state,'Derivative',derivative, ...
    'Residual',residual,'Info',info,'Parameters',parameters, ...
    'RawLocalResidual',residualDiagnostics.RawLocalResidual, ...
    'Compiled',compiled);
end

function merged=mergeOptions(defaults,overrides)
merged=defaults;names=fieldnames(overrides);
for index=1:numel(names),merged.(names{index})=overrides.(names{index});end
end

