function evaluationCount=verifyJacobianPattern( ...
    residualFunction,x,residual,pattern,stepSize,lowerBound,upperBound,feasible)
%VERIFYJACOBIANPATTERN Detect numerical dependencies omitted from a pattern.

if nargin<6 || isempty(lowerBound),lowerBound=-inf(size(x));end
if nargin<7 || isempty(upperBound),upperBound=inf(size(x));end
if nargin<8 || isempty(feasible),feasible=@(~)true;end
x=x(:);residual=residual(:);lowerBound=lowerBound(:);upperBound=upperBound(:);
evaluationCount=0;
threshold=200*eps(max(1,norm(residual,inf)))/stepSize;
for column=1:numel(x)
    nominal=stepSize*max(1,abs(x(column)));
    if upperBound(column)-x(column)>=nominal
        increment=nominal;
    elseif x(column)-lowerBound(column)>=nominal
        increment=-nominal;
    else
        continue
    end
    trialX=x;trialX(column)=trialX(column)+increment;
    if ~all(feasible(trialX)),continue,end
    trial=residualFunction(trialX);
    evaluationCount=evaluationCount+1;
    numericalMagnitude=abs((trial-residual)/increment);
    omitted=find(~logical(pattern(:,column)) & numericalMagnitude>threshold,1);
    if ~isempty(omitted)
        error('periodicring:PatternUnderreported', ...
            ['Jacobian pattern omits a dependency at residual row %d, ' ...
             'unknown column %d.'],omitted,column);
    end
end
end

