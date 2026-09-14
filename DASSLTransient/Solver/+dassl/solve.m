function result=solve(problem,timeSpan,initialState,options,initialDerivative)
%SOLVE Convenience functional interface to dassl.TransientSolver.

if nargin<4,options=struct();end
if nargin<5,initialDerivative=[];end
solver=dassl.TransientSolver(options);
result=solver.solve(problem,timeSpan,initialState,initialDerivative);
end

