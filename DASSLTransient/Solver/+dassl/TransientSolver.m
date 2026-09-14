classdef TransientSolver
    %TRANSIENTSOLVER Unified stiff ODE and index-1 DAE integration adapter.

    properties(SetAccess=private)
        Options
    end

    methods
        function obj=TransientSolver(options)
            if nargin<1 || isempty(options),options=struct();end
            if ~isstruct(options) || ~isscalar(options)
                error('dassl:Options','Options must be a scalar struct.');
            end
            defaults=struct('Backend',"auto",'RelTol',1e-6,'AbsTol',1e-8, ...
                'MaxStep',[],'InitialStep',[],'MaxOrder',5,'BDF',true, ...
                'Stats',false,'Refine',1,'OutputTimes',[], ...
                'ComputeConsistentInitialConditions',true, ...
                'RuntimeChecks',false,'ThrowOnNonfiniteResidual',false);
            names=fieldnames(defaults);
            unknown=setdiff(fieldnames(options),names);
            if ~isempty(unknown)
                error('dassl:UnknownOption','Unknown option: %s',unknown{1});
            end
            for index=1:numel(names)
                if ~isfield(options,names{index})
                    options.(names{index})=defaults.(names{index});
                end
            end
            backend=lower(string(options.Backend));
            if ~any(backend==["auto","ode15s","ode15i"])
                error('dassl:Backend','Backend must be auto, ode15s, or ode15i.');
            end
            options.Backend=backend;
            validateattributes(options.RelTol,{'numeric'}, ...
                {'scalar','real','finite','positive'});
            validateattributes(options.AbsTol,{'numeric'}, ...
                {'vector','real','finite','positive'});
            validateattributes(options.MaxOrder,{'numeric'}, ...
                {'scalar','integer','>=',1,'<=',5});
            obj.Options=options;
        end

        function result=solve(obj,problem,timeSpan,initialState,initialDerivative)
            if nargin<5,initialDerivative=[];end
            timeSpan=timeSpan(:).';
            if numel(timeSpan)<2 || any(~isfinite(timeSpan)) ...
                    || any(diff(timeSpan)<=0)
                error('dassl:TimeSpan', ...
                    'timeSpan must be finite and strictly increasing.');
            end
            requestedOutputTimes=obj.requestedOutputTimes(timeSpan);
            initialState=initialState(:);
            if any(~isfinite(initialState))
                error('dassl:InitialState','Initial state must be finite.');
            end
            if ~isempty(initialDerivative)
                initialDerivative=initialDerivative(:);
                if numel(initialDerivative)~=numel(initialState) ...
                        || any(~isfinite(initialDerivative))
                    error('dassl:InitialDerivative', ...
                        'Initial derivative must be finite and match the state.');
                end
            end
            backend=obj.chooseBackend(problem);
            odeOptions=obj.makeOdeOptions(problem,backend,initialDerivative);
            t0=timeSpan(1);tf=timeSpan(end);
            inputY0=initialState;inputYP0=initialDerivative;
            y0=inputY0;yp0=inputYP0;
            totalStarted=tic;
            consistentInitializationSeconds=0;
            switch backend
                case "ode15i"
                    if ~isfield(problem,'Residual') || isempty(problem.Residual)
                        error('dassl:Residual','ode15i requires problem.Residual.');
                    end
                    if isempty(yp0),yp0=zeros(size(y0));end
                    if obj.Options.ComputeConsistentInitialConditions
                        initializationStarted=tic;
                        fixedY=false(size(y0));fixedYP=false(size(y0));
                        if isfield(problem,'FixedY0') && ~isempty(problem.FixedY0)
                            fixedY=logical(problem.FixedY0(:));
                        end
                        if isfield(problem,'FixedYP0') && ~isempty(problem.FixedYP0)
                            fixedYP=logical(problem.FixedYP0(:));
                        end
                        if numel(fixedY)~=numel(y0) || numel(fixedYP)~=numel(y0)
                            error('dassl:InitialMask', ...
                                'FixedY0 and FixedYP0 must match the state size.');
                        end
                        [y0,yp0]=decic(@(t,y,yp)obj.callResidual( ...
                            problem,t,y,yp),t0,y0,fixedY,yp0,fixedYP,odeOptions);
                        consistentInitializationSeconds=toc(initializationStarted);
                    else
                        obj.callResidual(problem,t0,y0,yp0);
                    end
                    integrationStarted=tic;
                    solution=ode15i(@(t,y,yp)obj.callResidual( ...
                        problem,t,y,yp),[t0 tf],y0,yp0,odeOptions);
                case "ode15s"
                    if ~isfield(problem,'RHS') || isempty(problem.RHS)
                        error('dassl:RHS','ode15s requires problem.RHS.');
                    end
                    integrationStarted=tic;
                    solution=ode15s(@(t,y)obj.callRHS(problem,t,y), ...
                        [t0 tf],y0,odeOptions);
                otherwise
                    error('dassl:Backend','Unexpected backend.');
            end
            integrationElapsed=toc(integrationStarted);
            [actualY0,actualYP0]=deval(solution,t0);
            if backend=="ode15s" ...
                    && (~isfield(problem,'Mass') || isempty(problem.Mass))
                actualYP0=problem.RHS(t0,actualY0);
                actualYP0=actualYP0(:);
            end
            query=obj.outputTimes(requestedOutputTimes,solution.x(end));
            [state,derivative]=deval(solution,query);
            finalResidual=obj.auditResidual( ...
                problem,backend,query(end),state(:,end),derivative(:,end));
            totalElapsed=toc(totalStarted);
            result=struct('Solution',solution,'Time',query,'State',state, ...
                'Derivative',derivative,'InputInitialState',inputY0, ...
                'InputInitialDerivative',inputYP0, ...
                'InitialState',actualY0,'InitialDerivative',actualYP0, ...
                'FinalState',state(:,end), ...
                'FinalDerivative',derivative(:,end), ...
                'FinalResidualInf',norm(finalResidual,inf), ...
                'FinalResidual',finalResidual,'Backend',backend, ...
                'InternalMeshPointCount',numel(solution.x), ...
                'ConsistentInitializationElapsedSeconds', ...
                consistentInitializationSeconds, ...
                'IntegrationElapsedSeconds',integrationElapsed, ...
                'TotalElapsedSeconds',totalElapsed, ...
                'ElapsedSeconds',totalElapsed,'Options',obj.Options, ...
                'Evaluate',@(time)deval(solution,time));
        end
    end

    methods(Access=private)
        function backend=chooseBackend(obj,problem)
            backend=obj.Options.Backend;
            if backend=="auto"
                if isfield(problem,'Residual') && ~isempty(problem.Residual)
                    backend="ode15i";
                elseif isfield(problem,'RHS') && ~isempty(problem.RHS)
                    backend="ode15s";
                else
                    error('dassl:Problem', ...
                        'Problem requires either Residual or RHS.');
                end
            end
        end

        function options=makeOdeOptions(obj,problem,backend,initialDerivative)
            stats="off";if obj.Options.Stats,stats="on";end
            options=odeset('RelTol',obj.Options.RelTol, ...
                'AbsTol',obj.Options.AbsTol,'Stats',char(stats), ...
                'Refine',obj.Options.Refine,'MaxOrder',obj.Options.MaxOrder);
            if isfield(problem,'JacobianPattern') ...
                    && ~isempty(problem.JacobianPattern)
                if backend=="ode15i" && (~iscell(problem.JacobianPattern) ...
                        || numel(problem.JacobianPattern)~=2)
                    error('dassl:JacobianPattern', ...
                        'ode15i JacobianPattern must be {dF/dy,dF/dyp}.');
                elseif backend=="ode15s" && iscell(problem.JacobianPattern)
                    error('dassl:JacobianPattern', ...
                        'ode15s JacobianPattern must be one numeric pattern.');
                end
            end
            if isfield(problem,'Jacobian') && ~isempty(problem.Jacobian) ...
                    && backend=="ode15i" ...
                    && ~isa(problem.Jacobian,'function_handle') ...
                    && (~iscell(problem.Jacobian) || numel(problem.Jacobian)~=2)
                error('dassl:Jacobian', ...
                    ['ode15i Jacobian must be a two-output function or ' ...
                     '{dF/dy,dF/dyp}.']);
            elseif isfield(problem,'Jacobian') && ~isempty(problem.Jacobian) ...
                    && backend=="ode15s" && iscell(problem.Jacobian)
                error('dassl:Jacobian', ...
                    'ode15s Jacobian must be one matrix or function handle.');
            end
            if isfield(problem,'NonNegative') && ~isempty(problem.NonNegative) ...
                    && (backend~="ode15s" || ...
                    (isfield(problem,'Mass') && ~isempty(problem.Mass)))
                error('dassl:NonNegativeUnsupported', ...
                    'NonNegative is supported only for explicit ode15s problems.');
            end
            if ~isempty(obj.Options.MaxStep)
                options=odeset(options,'MaxStep',obj.Options.MaxStep);
            end
            if ~isempty(obj.Options.InitialStep)
                options=odeset(options,'InitialStep',obj.Options.InitialStep);
            end
            if backend=="ode15s"
                bdf="off";if obj.Options.BDF,bdf="on";end
                options=odeset(options,'BDF',char(bdf));
                if ~isempty(initialDerivative)
                    options=odeset(options,'InitialSlope',initialDerivative);
                end
            end
            mapping={ ...
                'Jacobian','Jacobian'; ...
                'JacobianPattern','JPattern'; ...
                'Events','Events'; ...
                'OutputFcn','OutputFcn'; ...
                'NonNegative','NonNegative'};
            for index=1:size(mapping,1)
                source=mapping{index,1};target=mapping{index,2};
                if isfield(problem,source) && ~isempty(problem.(source))
                    options=odeset(options,target,problem.(source));
                end
            end
            if backend=="ode15s" && isfield(problem,'Mass') ...
                    && ~isempty(problem.Mass)
                options=odeset(options,'Mass',problem.Mass);
                if isfield(problem,'MassStateDependence')
                    options=odeset(options,'MStateDependence', ...
                        char(string(problem.MassStateDependence)));
                end
                if isfield(problem,'MassVectorPattern')
                    options=odeset(options,'MvPattern',problem.MassVectorPattern);
                end
                if isfield(problem,'MassSingular')
                    options=odeset(options,'MassSingular', ...
                        char(string(problem.MassSingular)));
                end
            end
        end

        function value=callRHS(obj,problem,time,state)
            obj.validateState(problem,time,state);
            value=problem.RHS(time,state);
            value=value(:);
            if numel(value)~=numel(state) ...
                    || (obj.Options.RuntimeChecks && any(~isfinite(value)))
                error('dassl:RHSValue','RHS must be finite and match the state.');
            end
        end

        function value=callResidual(obj,problem,time,state,derivative)
            obj.validateState(problem,time,state);
            value=problem.Residual(time,state,derivative);
            value=value(:);
            if numel(value)~=numel(state)
                error('dassl:ResidualSize', ...
                    'Residual must be square and match the state size.');
            end
            if (obj.Options.RuntimeChecks ...
                    || obj.Options.ThrowOnNonfiniteResidual) && any(~isfinite(value))
                error('dassl:ResidualValue','Residual must be finite.');
            end
        end

        function validateState(obj,problem,time,state)
            if obj.Options.RuntimeChecks && any(~isfinite(state))
                error('dassl:StateValue','State must be finite.');
            end
            if isfield(problem,'ValidateState') && ~isempty(problem.ValidateState)
                accepted=problem.ValidateState(time,state);
                if ~isempty(accepted) && ~all(accepted)
                    error('dassl:StateDomain', ...
                        'ValidateState rejected the trial state.');
                end
            end
        end

        function query=requestedOutputTimes(obj,timeSpan)
            if isempty(obj.Options.OutputTimes)
                query=timeSpan;
            else
                query=obj.Options.OutputTimes(:).';
                if isempty(query) || any(~isfinite(query)) ...
                        || any(diff(query)<0) || query(1)<timeSpan(1) ...
                        || query(end)>timeSpan(end)
                    error('dassl:OutputTimes', ...
                        'OutputTimes must be sorted and lie inside timeSpan.');
                end
            end
        end

        function query=outputTimes(~,query,actualEnd)
            query=query(query<=actualEnd+10*eps(max(1,abs(actualEnd))));
            if isempty(query) || query(end)<actualEnd-10*eps(max(1,abs(actualEnd)))
                query=[query,actualEnd];
            else
                query(end)=actualEnd;
            end
        end

        function residual=auditResidual(~,problem,backend,time,state,derivative)
            if backend=="ode15i"
                residual=problem.Residual(time,state,derivative);
            else
                rhs=problem.RHS(time,state);
                if isfield(problem,'Mass') && ~isempty(problem.Mass)
                    if isa(problem.Mass,'function_handle')
                        dependence="weak";
                        if isfield(problem,'MassStateDependence')
                            dependence=lower(string(problem.MassStateDependence));
                        end
                        if dependence=="none"
                            mass=problem.Mass(time);
                        else
                            mass=problem.Mass(time,state);
                        end
                    else
                        mass=problem.Mass;
                    end
                    residual=mass*derivative-rhs;
                else
                    residual=derivative-rhs;
                end
            end
            residual=residual(:);
        end
    end
end
