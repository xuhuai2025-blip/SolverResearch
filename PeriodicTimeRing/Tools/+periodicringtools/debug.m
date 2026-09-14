function report=debug(compiled,initialGuess,parameters,options)
%DEBUG Audit residual assembly and declared Jacobian sparsity.

if nargin<3,parameters=compiled.Parameters;end
if nargin<4,options=struct();end
if ~isfield(options,'FiniteDifferenceStep')
    options.FiniteDifferenceStep=sqrt(eps);
end
n=compiled.StateCount;N=compiled.Grid.NodeCount;
if isa(initialGuess,'function_handle')
    initialGuess=initialGuess(compiled.Grid.Time);
end
if isvector(initialGuess) && n==1,initialGuess=initialGuess(:).';end
if ~isequal(size(initialGuess),[n,N]) || any(~isfinite(initialGuess),'all')
    error('periodicringtools:InitialGuess', ...
        'Initial guess must be finite and stateCount-by-nodeCount.');
end
x=initialGuess(:);
residualFunction=@(value)periodicring.residual(compiled,value,parameters);
[residual,diagnostics]=periodicring.residual(compiled,x,parameters);
lower=-inf(size(x));upper=inf(size(x));
if isfield(compiled.NewtonDefaults,'LowerBound')
    lower=compiled.NewtonDefaults.LowerBound;
end
if isfield(compiled.NewtonDefaults,'UpperBound')
    upper=compiled.NewtonDefaults.UpperBound;
end
evaluations=periodicringtools.verifyJacobianPattern( ...
    residualFunction,x,residual,compiled.Pattern, ...
    options.FiniteDifferenceStep,lower,upper,@(~)true);
report=struct('Passed',true,'PatternAuditEvaluations',evaluations, ...
    'InitialResidualInf',norm(residual,inf), ...
    'RawLocalResidual',diagnostics.RawLocalResidual, ...
    'GlobalResidual',diagnostics.GlobalResidual);
end
