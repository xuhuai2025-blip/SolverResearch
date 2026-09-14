function result=runStiffODE(doPlot)
%RUNSTIFFODE Manufactured stiff problem with exact solution cos(t).

if nargin<1,doPlot=(nargout==0);end
root=fileparts(fileparts(mfilename('fullpath')));
solverRoot=fullfile(root,'Solver');
oldPath=path;
addpath(solverRoot);
cleanup=onCleanup(@()path(oldPath));
problem.RHS=@(t,y)-1000*(y-cos(t))-sin(t);
solver=dassl.TransientSolver(struct('Backend',"ode15s", ...
    'RelTol',1e-7,'AbsTol',1e-9,'BDF',true));
result=solver.solve(problem,linspace(0,1,101),1);
result.MaximumError=max(abs(result.State-cos(result.Time)),[],'all');
if doPlot
    plot(result.Time,result.State,'o',result.Time,cos(result.Time),'-')
    legend('computed','exact');xlabel('t');ylabel('y');grid on
    title(sprintf('Stiff ODE, max error %.3g',result.MaximumError))
end
end
