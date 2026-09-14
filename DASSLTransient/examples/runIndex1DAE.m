function result=runIndex1DAE(doPlot)
%RUNINDEX1DAE Residual DAE with y1=exp(-t), y2=y1^2.

if nargin<1,doPlot=(nargout==0);end
root=fileparts(fileparts(mfilename('fullpath')));
solverRoot=fullfile(root,'Solver');
oldPath=path;
addpath(solverRoot);
cleanup=onCleanup(@()path(oldPath));
problem.Residual=@residual;
problem.FixedY0=[true;false];
problem.FixedYP0=[false;true];
solver=dassl.TransientSolver(struct('Backend',"ode15i", ...
    'RelTol',1e-7,'AbsTol',[1e-9;1e-9]));
result=solver.solve(problem,linspace(0,1,101),[1;.8],[-1;-2]);
exact=[exp(-result.Time);exp(-2*result.Time)];
result.MaximumError=max(abs(result.State-exact),[],'all');
if doPlot
    plot(result.Time,result.State,'o',result.Time,exact,'-')
    xlabel('t');ylabel('state');grid on
    title(sprintf('Index-1 DAE, max error %.3g',result.MaximumError))
end
end

function value=residual(~,state,derivative)
value=[derivative(1)+state(1);state(2)-state(1)^2];
end
