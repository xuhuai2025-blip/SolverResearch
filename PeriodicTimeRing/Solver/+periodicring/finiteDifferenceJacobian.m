function [jacobian,evaluationCount]=finiteDifferenceJacobian( ...
    residualFunction,x,residual,pattern,colors,stepSize, ...
    lowerBound,upperBound,feasible)
%FINITEDIFFERENCEJACOBIAN Bound-aware colored numerical Jacobian.

if nargin<7 || isempty(lowerBound),lowerBound=-inf(size(x));end
if nargin<8 || isempty(upperBound),upperBound=inf(size(x));end
if nargin<9 || isempty(feasible),feasible=@(~)true;end
x=x(:);residual=residual(:);lowerBound=lowerBound(:);upperBound=upperBound(:);
if numel(colors)~=numel(x) || ~isequal(size(pattern), ...
        [numel(residual),numel(x)])
    error('periodicring:FiniteDifferenceSize', ...
        'Pattern, colors, state, and residual sizes are inconsistent.');
end
validateattributes(stepSize,{'numeric'}, ...
    {'scalar','real','finite','positive'});

nonzeroCount=nnz(pattern);
rows=zeros(nonzeroCount,1);
columns=zeros(nonzeroCount,1);
values=zeros(nonzeroCount,1);
cursor=0;evaluationCount=0;
for color=1:max(colors,[],'omitnan')
    selected=find(colors==color);
    perturbation=zeros(size(x));
    for local=1:numel(selected)
        column=selected(local);
        nominal=stepSize*max(1,abs(x(column)));
        plusRoom=upperBound(column)-x(column);
        minusRoom=x(column)-lowerBound(column);
        if plusRoom>=nominal
            increment=nominal;
        elseif minusRoom>=nominal
            increment=-nominal;
        elseif plusRoom>=minusRoom && plusRoom>0
            increment=.5*plusRoom;
        elseif minusRoom>0
            increment=-.5*minusRoom;
        else
            error('periodicring:FixedVariablePattern', ...
                'A structurally active variable cannot be perturbed inside its bounds.');
        end
        perturbation(column)=increment;
    end
    trialX=x+perturbation;
    if ~all(feasible(trialX))
        error('periodicring:FiniteDifferenceDomain', ...
            'A colored finite-difference perturbation violates the state domain.');
    end
    trial=residualFunction(trialX);
    evaluationCount=evaluationCount+1;
    difference=trial-residual;
    for local=1:numel(selected)
        column=selected(local);
        affected=find(pattern(:,column));
        count=numel(affected);
        range=cursor+(1:count);
        rows(range)=affected;
        columns(range)=column;
        values(range)=difference(affected)/perturbation(column);
        cursor=cursor+count;
    end
end
jacobian=sparse(rows(1:cursor),columns(1:cursor),values(1:cursor), ...
    numel(residual),numel(x));
end

