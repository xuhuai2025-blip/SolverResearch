function specification=boundary(type,value)
%BOUNDARY Create an explicit finite-volume boundary specification.
% Types: dirichlet, outflow, zero-gradient, neumann, periodic.

if nargin<2,value=[];end
type=lower(string(type));
allowed=["dirichlet","outflow","zero-gradient","neumann","periodic"];
if ~isscalar(type) || ~any(type==allowed)
    error('tvd2:BoundaryType','Unsupported boundary type: %s',type);
end
if any(type==["dirichlet","neumann"]) && isempty(value)
    error('tvd2:BoundaryValue','Boundary type %s requires a value.',type);
end
specification=struct('Type',type,'Value',value);
end

