function integral=integrate(values,grid)
%INTEGRATE Integrate variables-by-node values over one complete period.

if ~isnumeric(values) || ~ismatrix(values) ...
        || size(values,2)~=grid.NodeCount || any(~isfinite(values),'all')
    error('periodicring:ValueSize', ...
        'Values must be a finite nVariable-by-nodeCount numeric matrix.');
end
integral=values*grid.Weight;
end

