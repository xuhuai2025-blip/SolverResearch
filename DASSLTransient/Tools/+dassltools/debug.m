function diagnostics=debug(problem,timeSpan,initialState,initialDerivative,options)
%DEBUG Validate and run one instrumented transient solve.

if nargin<4,initialDerivative=[];end
if nargin<5,options=struct();end
if isfield(options,'Backend'),backend=options.Backend;else,backend="auto";end
validation=dassltools.validateProblem( ...
    problem,initialState,initialDerivative,backend,timeSpan(1));
options.Backend=backend;options.Stats=true;
solver=dassl.TransientSolver(options);
result=solver.solve(problem,timeSpan,initialState,initialDerivative);
diagnostics=struct('Validation',validation,'Result',result, ...
    'FinalResidualInf',result.FinalResidualInf, ...
    'InternalMeshPointCount',result.InternalMeshPointCount, ...
    'ElapsedSeconds',result.ElapsedSeconds);
end
