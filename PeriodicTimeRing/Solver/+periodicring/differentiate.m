function derivative=differentiate(values,grid)
%DIFFERENTIATE Differentiate variables-by-node values on a time ring.

validateValues(values,grid);
derivative=values*grid.D.';
end

function validateValues(values,grid)
if ~isnumeric(values) || ~ismatrix(values) ...
        || size(values,2)~=grid.NodeCount || any(~isfinite(values),'all')
    error('periodicring:ValueSize', ...
        'Values must be a finite nVariable-by-nodeCount numeric matrix.');
end
end

