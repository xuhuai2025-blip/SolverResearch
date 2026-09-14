function result=runForcedLinear
%RUNFORCEDLINEAR Solve a manufactured scalar periodic ODE.

root=fileparts(fileparts(mfilename('fullpath')));
solverRoot=fullfile(root,'Solver');
oldPath=path;
addpath(solverRoot);
cleanup=onCleanup(@()path(oldPath));
period=3;origin=.2;grid=periodicring.uniformGrid(7,period,origin);
parameters=struct('Origin',origin,'Omega',2*pi/period);
problem.StateCount=1;
problem.LocalResidual=@localResidual;
problem.StatePattern=sparse(1);
problem.DerivativePattern=sparse(1);
compiled=periodicring.compile(problem,grid);
options=struct('Tolerance',1e-11,'Display',"iter");
result=periodicring.solve(compiled,zeros(1,grid.NodeCount),options,parameters);

exact=sin(parameters.Omega*(grid.Time-origin)).';
fprintf('maximum waveform error: %.3e\n',max(abs(result.State-exact),[],'all'));
end

function residual=localResidual(time,state,derivative,parameters)
phase=parameters.Omega*(time-parameters.Origin);
forcing=parameters.Omega*cos(phase)+sin(phase);
residual=derivative+state-forcing;
end
