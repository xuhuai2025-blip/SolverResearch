function pattern=buildJacobianPattern(statePattern,derivativePattern, ...
    grid,globalRows,globalPattern)
%BUILDJACOBIANPATTERN Assemble the complete space-time sparsity pattern.

if nargin<4 || isempty(globalRows),globalRows=zeros(0,1);end
if nargin<5,globalPattern=[];end
statePattern=spones(sparse(statePattern));
derivativePattern=spones(sparse(derivativePattern));
if size(statePattern,1)~=size(statePattern,2) ...
        || ~isequal(size(statePattern),size(derivativePattern))
    error('periodicring:LocalPattern', ...
        'State and derivative patterns must be equal-size square matrices.');
end
stateCount=size(statePattern,1);
unknownCount=stateCount*grid.NodeCount;
pattern=spones(kron(speye(grid.NodeCount),statePattern)+ ...
    kron(spones(grid.D),derivativePattern));

globalRows=globalRows(:);
if any(globalRows<1) || any(globalRows>unknownCount) ...
        || any(globalRows~=fix(globalRows)) || numel(unique(globalRows))~=numel(globalRows)
    error('periodicring:GlobalRows', ...
        'GlobalRows must be unique valid flat residual indices.');
end
if ~isempty(globalRows)
    if ~isequal(size(globalPattern),[numel(globalRows),unknownCount])
        error('periodicring:GlobalPattern', ...
            'GlobalPattern must have one row per replaced residual row.');
    end
    pattern(globalRows,:)=spones(sparse(globalPattern));
elseif ~isempty(globalPattern)
    error('periodicring:GlobalPattern', ...
        'GlobalPattern was supplied without GlobalRows.');
end
pattern=spones(pattern);
end

