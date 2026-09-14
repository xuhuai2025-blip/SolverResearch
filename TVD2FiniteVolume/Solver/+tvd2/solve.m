function result=solve(problem,mesh,initialState,timeSpan,options)
%SOLVE Advance a one-dimensional conservation law from an initial state.

if nargin<5,options=struct();end
defaults=struct('Order',2,'Limiter',"minmod",'Integrator',"ssp-rk2", ...
    'CFL',0.45,'TimeStep',[],'MaximumSteps',1000000, ...
    'StepCallback',[],'RuntimeChecks',false);
names=fieldnames(defaults);
unknown=setdiff(fieldnames(options),names);
if ~isempty(unknown)
    error('tvd2:UnknownOption','Unknown option: %s',unknown{1});
end
for index=1:numel(names)
    if ~isfield(options,names{index}),options.(names{index})=defaults.(names{index});end
end
timeSpan=timeSpan(:).';
if numel(timeSpan)<2 || any(~isfinite(timeSpan)) || any(diff(timeSpan)<=0)
    error('tvd2:TimeSpan','timeSpan must be finite and strictly increasing.');
end
if isvector(initialState) && numel(initialState)==mesh.CellCount
    initialState=initialState(:).';
end
validateattributes(initialState,{'numeric'},{'2d','real','finite'});
if size(initialState,2)~=mesh.CellCount
    error('tvd2:InitialState','Initial state must have one column per cell.');
end
if isempty(options.TimeStep)
    validateattributes(options.CFL,{'numeric'},{'scalar','real','finite','positive'});
    measureTolerance=100*eps(max(1,max(mesh.CellWidth)));
    if any(abs(mesh.CellMeasure-mesh.CellWidth)>measureTolerance)
        error('tvd2:CFLMeasure', ...
            ['Automatic CFL stepping is defined only when CellMeasure ' ...
             'equals CellWidth. Supply options.TimeStep for a custom capacity.']);
    end
else
    validateattributes(options.TimeStep,{'numeric'}, ...
        {'scalar','real','finite','positive'});
end
integrator=lower(string(options.Integrator));
if ~any(integrator==["ssp-rk2","ssp-rk3"])
    error('tvd2:Integrator','Integrator must be ssp-rk2 or ssp-rk3.');
end

state=zeros([size(initialState),numel(timeSpan)],'like',initialState);
state(:,:,1)=initialState;
current=initialState;
time=timeSpan(1);
stepCount=0;
rejectedStepCount=0;
started=tic;
for outputIndex=2:numel(timeSpan)
    target=timeSpan(outputIndex);
    while time<target
        remaining=target-time;
        retryLimit=remaining;
        retryCount=0;
        while true
            [k1,meta1]=tvd2.spatialOperator(problem,time,current,mesh,options);
            if isempty(options.TimeStep)
                if ~isfinite(meta1.MaxWaveSpeed)
                    error('tvd2:CFL', ...
                        'Automatic CFL stepping requires problem.MaxWaveSpeed.');
                elseif meta1.MaxWaveSpeed==0
                    dt=min(remaining,retryLimit);
                else
                    dt=min([remaining,retryLimit,options.CFL* ...
                        min(mesh.CellWidth)/meta1.MaxWaveSpeed]);
                end
            else
                dt=min(remaining,options.TimeStep);
            end
            if ~(dt>0) || time+dt==time
                error('tvd2:TimeStepUnderflow', ...
                    'Time step underflow at t=%.17g.',time);
            end
            first=current+dt*k1;
            [k2,meta2]=tvd2.spatialOperator( ...
                problem,time+dt,first,mesh,options);
            if integrator=="ssp-rk2"
                candidate=.5*current+.5*(first+dt*k2);
                stageMax=max([meta1.MaxWaveSpeed,meta2.MaxWaveSpeed]);
            else
                second=.75*current+.25*(first+dt*k2);
                [k3,meta3]=tvd2.spatialOperator( ...
                    problem,time+.5*dt,second,mesh,options);
                candidate=(1/3)*current+(2/3)*(second+dt*k3);
                stageMax=max([meta1.MaxWaveSpeed,meta2.MaxWaveSpeed, ...
                    meta3.MaxWaveSpeed]);
            end
            if isempty(options.TimeStep) && stageMax>0
                stableStep=options.CFL*min(mesh.CellWidth)/stageMax;
                if dt>stableStep*(1+100*eps)
                    retryLimit=.98*stableStep;
                    retryCount=retryCount+1;
                    rejectedStepCount=rejectedStepCount+1;
                    if retryCount>25
                        error('tvd2:CFLRetry', ...
                            'Stage-speed CFL retry failed at t=%.17g.',time);
                    end
                    continue
                end
            end
            break
        end
        if any(~isfinite(candidate),'all')
            error('tvd2:NonfiniteState','The state became nonfinite at t=%.17g.',time+dt);
        end
        current=candidate;
        time=time+dt;
        stepCount=stepCount+1;
        if stepCount>options.MaximumSteps
            error('tvd2:MaximumSteps','MaximumSteps was exceeded.');
        end
        if ~isempty(options.StepCallback)
            options.StepCallback(time,current,stepCount);
        end
    end
    state(:,:,outputIndex)=current;
end
result=struct('Time',timeSpan,'State',state,'Mesh',mesh, ...
    'StepCount',stepCount,'RejectedStepCount',rejectedStepCount, ...
    'ElapsedSeconds',toc(started),'Options',options, ...
    'Method',"finite volume / order="+options.Order+" / "+integrator, ...
    'SpatialReconstruction',"bounded MUSCL / "+string(options.Limiter));
end
