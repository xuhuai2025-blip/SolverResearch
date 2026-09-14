function report=validateProblem(problem,grid,initialGuess,parameters)
%VALIDATEPROBLEM Perform optional structural checks before a production run.

if nargin<4,parameters=[];end
problem.RuntimeChecks=true;
compiled=periodicring.compile(problem,grid);
if nargin<4 && isfield(problem,'Parameters'),parameters=problem.Parameters;end
n=compiled.StateCount;N=grid.NodeCount;
if isa(initialGuess,'function_handle'),initialGuess=initialGuess(grid.Time);end
if isvector(initialGuess) && n==1,initialGuess=initialGuess(:).';end
assert(isequal(size(initialGuess),[n,N]) && all(isfinite(initialGuess),'all'), ...
    'periodicringtools:InitialGuess','Initial guess has the wrong size or values.');
[residual,diagnostics]=periodicring.residual( ...
    compiled,initialGuess(:),parameters);
zeroRows=find(sum(compiled.Pattern,2)==0);
zeroColumns=find(sum(compiled.Pattern,1)==0).';
coloringValid=true;
for color=1:compiled.ColorCount
    columns=find(compiled.Colors==color);
    coloringValid=coloringValid ...
        && all(full(sum(compiled.Pattern(:,columns),2))<=1);
end
report=struct('Passed',isempty(zeroRows) && isempty(zeroColumns) ...
    && coloringValid,'Compiled',compiled,'InitialResidualInf',norm(residual,inf), ...
    'ZeroRows',zeroRows,'ZeroColumns',zeroColumns, ...
    'ColoringValid',coloringValid,'RawLocalResidual', ...
    diagnostics.RawLocalResidual);
if ~report.Passed
    error('periodicringtools:ValidationFailed', ...
        'Pattern has zero rows/columns or invalid color groups.');
end
end
