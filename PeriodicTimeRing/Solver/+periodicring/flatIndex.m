function index=flatIndex(stateIndex,nodeIndex,stateCount)
%FLATINDEX Map state/node subscripts into the solver vector ordering.

validateattributes(stateCount,{'numeric'}, ...
    {'scalar','integer','positive','finite'});
validateattributes(stateIndex,{'numeric'}, ...
    {'integer','positive','finite','<=',stateCount});
validateattributes(nodeIndex,{'numeric'}, ...
    {'integer','positive','finite'});
if ~isscalar(stateIndex) && ~isscalar(nodeIndex) ...
        && ~isequal(size(stateIndex),size(nodeIndex))
    error('periodicring:IndexSize', ...
        'State and node indices must be scalar-expandable.');
end
index=stateIndex+(nodeIndex-1)*stateCount;
end

