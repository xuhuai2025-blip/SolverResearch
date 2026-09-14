function targetValues=resample(values,grid,targetTime)
%RESAMPLE Fourier-interpolate variables-by-node values at target times.

if ~isnumeric(values) || ~ismatrix(values) ...
        || size(values,2)~=grid.NodeCount || any(~isfinite(values),'all')
    error('periodicring:ValueSize', ...
        'Values must be a finite nVariable-by-nodeCount numeric matrix.');
end
matrix=periodicring.evaluationMatrix(grid,targetTime);
targetValues=values*matrix.';
end

