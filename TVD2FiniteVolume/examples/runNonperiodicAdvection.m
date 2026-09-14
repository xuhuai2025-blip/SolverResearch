function result=runNonperiodicAdvection(doPlot)
%RUNNONPERIODICADVECTION Unit-speed inflow/outflow transport example.

if nargin<1,doPlot=(nargout==0);end
root=fileparts(fileparts(mfilename('fullpath')));
solverRoot=fullfile(root,'Solver');
oldPath=path;
addpath(solverRoot);
cleanup=onCleanup(@()path(oldPath));
mesh=tvd2.makeMesh(linspace(0,1,101));
initial=zeros(1,mesh.CellCount);
problem=struct();
problem.Flux=@(U,x,t)U;
problem.MaxWaveSpeed=@(~,~,x,~)ones(size(x));
problem.Boundary=struct( ...
    'Left',tvd2.boundary("dirichlet",1), ...
    'Right',tvd2.boundary("outflow"));
options=struct('Limiter',"minmod",'Integrator',"ssp-rk2",'CFL',.45);
result=tvd2.solve(problem,mesh,initial,linspace(0,.5,6),options);
if doPlot
    plot(mesh.CellX,squeeze(result.State(1,:,:)),'LineWidth',1.2)
    xlabel('x');ylabel('u');title('Non-periodic inflow transport')
    grid on
end
end
