function [x,info]=solveNewton(residualFunction,x0,pattern,options)
%SOLVENEWTON Solve a square nonlinear system by sparse damped Newton steps.

if nargin<4,options=struct();end
x0=x0(:);pattern=spones(sparse(pattern));
if ~isequal(size(pattern),[numel(x0),numel(x0)])
    error('periodicring:PatternSize', ...
        'The Jacobian pattern must be square and match x0.');
end
defaults=struct('VariableScale',ones(size(x0)), ...
    'ResidualScale',ones(size(x0)), ...
    'LowerBound',-inf(size(x0)),'UpperBound',inf(size(x0)), ...
    'Tolerance',1e-8,'MaxIterations',12, ...
    'FiniteDifferenceStep',sqrt(eps), ...
    'MaxScaledStepInf',2,'MinimumLineSearchStep',2^-14, ...
    'ArmijoCoefficient',1e-4,'Display',"none", ...
    'Colors',[],'ThrowOnFailure',true, ...
    'Feasible',[]);
options=mergeDefaults(options,defaults);
n=numel(x0);
variableScale=expandVector(options.VariableScale,n,'VariableScale');
residualScale=expandVector(options.ResidualScale,n,'ResidualScale');
lowerBound=expandVector(options.LowerBound,n,'LowerBound');
upperBound=expandVector(options.UpperBound,n,'UpperBound');
if any(variableScale<=0) || any(residualScale<=0) ...
        || any(lowerBound>upperBound)
    error('periodicring:ScaleOrBounds', ...
        'Scales must be positive and lower bounds cannot exceed upper bounds.');
end
if isempty(options.Feasible),physicalFeasible=@(~)true;
else,physicalFeasible=options.Feasible;end
lowerScaled=lowerBound./variableScale;
upperScaled=upperBound./variableScale;
feasible=@(w)all(w>=lowerScaled & w<=upperScaled) ...
    && all(physicalFeasible(w.*variableScale),'all');
scaledResidual=@(w)evaluateScaled( ...
    residualFunction,w,variableScale,residualScale,n);
w=x0./variableScale;
if ~feasible(w)
    error('periodicring:InitialBounds', ...
        'The initial state violates bounds or the state-domain validator.');
end

started=tic;colorStarted=tic;
if isempty(options.Colors)
    [colors,colorCount]=periodicring.greedyColoring(pattern);
else
    colors=options.Colors(:);
    if numel(colors)~=n || any(colors<0) || any(colors~=fix(colors))
        error('periodicring:Colors','Colors must be nonnegative integers per column.');
    end
    colorCount=max(colors,[],'omitnan');
end
activeColumns=full(sum(pattern,1)).'>0;
if any(activeColumns & colors==0)
    error('periodicring:Colors', ...
        'Every structurally active column must have a positive color.');
end
for color=1:colorCount
    selected=colors==color;
    if any(full(sum(pattern(:,selected),2))>1)
        error('periodicring:ColorConflict', ...
            'Columns of one color have overlapping residual-row support.');
    end
end
colorSeconds=toc(colorStarted);
r=scaledResidual(w);functionEvaluations=1;
history=repmat(struct('Iteration',0,'ResidualInf',NaN, ...
    'ResidualRms',NaN,'StepInf',NaN,'LineSearchAlpha',NaN, ...
    'JacobianSeconds',NaN,'LinearSolveSeconds',NaN),options.MaxIterations,1);
historyCount=0;
converged=false;

for iteration=0:options.MaxIterations
    residualInf=norm(r,inf);
    residualRms=norm(r)/sqrt(numel(r));
    if string(options.Display)~="none"
        fprintf('periodic Newton %d: |R|inf=%.6e, RMS=%.6e\n', ...
            iteration,residualInf,residualRms);
    end
    if residualInf<=options.Tolerance,converged=true;break,end
    if iteration==options.MaxIterations,break,end

    jacobianStarted=tic;
    [jacobian,evaluations]=periodicring.finiteDifferenceJacobian( ...
        scaledResidual,w,r,pattern,colors,options.FiniteDifferenceStep, ...
        lowerScaled,upperScaled,feasible);
    jacobianSeconds=toc(jacobianStarted);
    functionEvaluations=functionEvaluations+evaluations;
    if any(~isfinite(nonzeros(jacobian)))
        error('periodicring:NonfiniteJacobian', ...
            'The numerical Jacobian is nonfinite.');
    end
    linearStarted=tic;step=jacobian\(-r);linearSeconds=toc(linearStarted);
    if any(~isfinite(step))
        error('periodicring:LinearSolve','The sparse Newton step is nonfinite.');
    end
    stepInf=norm(step,inf);
    if stepInf>options.MaxScaledStepInf
        step=step*(options.MaxScaledStepInf/stepInf);
        stepInf=options.MaxScaledStepInf;
    end

    merit=.5*(r.'*r);alpha=1;accepted=false;
    bestMerit=Inf;bestW=w;bestR=r;bestAlpha=0;
    while alpha>=options.MinimumLineSearchStep
        trialW=w+alpha*step;
        if feasible(trialW)
            % Callback errors intentionally propagate: a programming error is
            % not equivalent to a merely rejected line-search point.
            trialR=scaledResidual(trialW);
            functionEvaluations=functionEvaluations+1;
            trialMerit=.5*(trialR.'*trialR);
            if trialMerit<bestMerit
                bestMerit=trialMerit;bestW=trialW;bestR=trialR;bestAlpha=alpha;
            end
            if isfinite(trialMerit) && ...
                    trialMerit<=(1-options.ArmijoCoefficient*alpha)*merit
                accepted=true;break
            end
        end
        alpha=alpha/2;
    end
    if accepted
        w=trialW;r=trialR;
    elseif bestMerit<merit
        w=bestW;r=bestR;alpha=bestAlpha;
    else
        error('periodicring:LineSearch', ...
            'No damped Newton step reduced the scaled residual.');
    end
    historyCount=historyCount+1;
    history(historyCount)=struct('Iteration',iteration+1, ...
        'ResidualInf',norm(r,inf),'ResidualRms',norm(r)/sqrt(numel(r)), ...
        'StepInf',stepInf,'LineSearchAlpha',alpha, ...
        'JacobianSeconds',jacobianSeconds, ...
        'LinearSolveSeconds',linearSeconds);
end

x=w.*variableScale;
history=history(1:historyCount);
info=struct('Converged',converged,'Iterations',historyCount, ...
    'FinalResidualInf',norm(r,inf), ...
    'FinalResidualRms',norm(r)/sqrt(numel(r)), ...
    'FunctionEvaluations',functionEvaluations,'ColorCount',colorCount, ...
    'ColoringSeconds',colorSeconds,'ElapsedSeconds',toc(started), ...
    'History',history,'PatternNonzeros',nnz(pattern),'UnknownCount',n);
if ~converged && options.ThrowOnFailure
    error('periodicring:NoConvergence', ...
        'Newton stopped at |R|inf=%.6e after %d iterations.', ...
        info.FinalResidualInf,options.MaxIterations);
end
end

function value=evaluateScaled(functionHandle,w,variableScale,residualScale,n)
value=functionHandle(w.*variableScale);
value=value(:);
if numel(value)~=n || ~isreal(value) || any(~isfinite(value))
    error('periodicring:ResidualValue', ...
        'Residual must be a finite real column with one value per unknown.');
end
value=value./residualScale;
end

function value=expandVector(value,count,name)
if isscalar(value),value=repmat(value,count,1);else,value=value(:);end
if numel(value)~=count || ~isreal(value) || any(isnan(value))
    error('periodicring:OptionSize','%s must be scalar or match x0.',name);
end
end

function value=mergeDefaults(value,defaults)
names=fieldnames(defaults);
unknown=setdiff(fieldnames(value),names);
if ~isempty(unknown)
    error('periodicring:UnknownOption','Unknown option: %s',unknown{1});
end
for index=1:numel(names)
    if ~isfield(value,names{index}),value.(names{index})=defaults.(names{index});end
end
end
