function report=validateProblem( ...
    problem,initialState,initialDerivative,backend,sampleTime)
%VALIDATEPROBLEM Probe a stiff ODE/DAE problem without integrating it.

if nargin<3,initialDerivative=[];end
if nargin<4 || strlength(string(backend))==0,backend="auto";end
if nargin<5,sampleTime=0;end
backend=lower(string(backend));
if backend=="auto"
    if isfield(problem,'Residual') && ~isempty(problem.Residual)
        backend="ode15i";
    else
        backend="ode15s";
    end
end
validateattributes(sampleTime,{'numeric'},{'scalar','real','finite'});
y=initialState(:);n=numel(y);
assert(n>0 && all(isfinite(y)), ...
    'dassltools:InitialState','Initial state must be finite and nonempty.');
if backend=="ode15i"
    assert(isfield(problem,'Residual') && isa(problem.Residual,'function_handle'), ...
        'dassltools:Residual','ode15i requires problem.Residual.');
    if isempty(initialDerivative),initialDerivative=zeros(n,1);end
    assert(numel(initialDerivative)==n && all(isfinite(initialDerivative)), ...
        'dassltools:InitialDerivative','Initial derivative has the wrong size.');
    value=problem.Residual(sampleTime,y,initialDerivative(:));
    if isfield(problem,'JacobianPattern') && ~isempty(problem.JacobianPattern)
        assert(iscell(problem.JacobianPattern) ...
            && numel(problem.JacobianPattern)==2, ...
            'dassltools:JacobianPattern', ...
            'ode15i JacobianPattern must be {dF/dy,dF/dyp}.');
        assert(all(cellfun(@(p)isempty(p) || isequal(size(p),[n,n]), ...
            problem.JacobianPattern)), ...
            'dassltools:JacobianPattern', ...
            'Both ode15i Jacobian patterns must be n-by-n.');
    end
    if isfield(problem,'Jacobian') && ~isempty(problem.Jacobian)
        if isa(problem.Jacobian,'function_handle')
            [jy,jyp]=problem.Jacobian( ...
                sampleTime,y,initialDerivative(:));
        else
            assert(iscell(problem.Jacobian) && numel(problem.Jacobian)==2, ...
                'dassltools:Jacobian', ...
                'ode15i Jacobian must be a two-output function or two matrices.');
            jy=problem.Jacobian{1};jyp=problem.Jacobian{2};
        end
        assert((isempty(jy) || isequal(size(jy),[n,n])) ...
            && (isempty(jyp) || isequal(size(jyp),[n,n])), ...
            'dassltools:Jacobian','Both ode15i Jacobians must be n-by-n.');
    end
    form="implicit residual F(t,y,yp)=0";
elseif backend=="ode15s"
    assert(isfield(problem,'RHS') && isa(problem.RHS,'function_handle'), ...
        'dassltools:RHS','ode15s requires problem.RHS.');
    value=problem.RHS(sampleTime,y);
    if isfield(problem,'JacobianPattern') && ~isempty(problem.JacobianPattern)
        assert(~iscell(problem.JacobianPattern) ...
            && isequal(size(problem.JacobianPattern),[n,n]), ...
            'dassltools:JacobianPattern', ...
            'ode15s JacobianPattern must be an n-by-n numeric pattern.');
    end
    if isfield(problem,'Jacobian') && ~isempty(problem.Jacobian)
        if isa(problem.Jacobian,'function_handle')
            jacobian=problem.Jacobian(sampleTime,y);
        else
            jacobian=problem.Jacobian;
        end
        assert(~iscell(jacobian) && isequal(size(jacobian),[n,n]), ...
            'dassltools:Jacobian','ode15s Jacobian must evaluate to n-by-n.');
    end
    if isfield(problem,'Mass') && ~isempty(problem.Mass)
        mass=problem.Mass;
        if isa(mass,'function_handle')
            dependence="weak";
            if isfield(problem,'MassStateDependence')
                dependence=lower(string(problem.MassStateDependence));
            end
            if dependence=="none",mass=mass(sampleTime);
            else,mass=mass(sampleTime,y);end
        end
        assert(isequal(size(mass),[n,n]) && all(isfinite(mass),'all'), ...
            'dassltools:Mass','Mass must evaluate to a finite n-by-n matrix.');
    end
    form="explicit or mass-matrix M*yp=f";
else
    error('dassltools:Backend','Backend must be ode15s or ode15i.');
end
value=value(:);
assert(numel(value)==n && all(isfinite(value)), ...
    'dassltools:CallbackValue','The sampled callback must return n finite values.');
if isfield(problem,'NonNegative') && ~isempty(problem.NonNegative) ...
        && (backend~="ode15s" || ...
        (isfield(problem,'Mass') && ~isempty(problem.Mass)))
    error('dassltools:NonNegativeUnsupported', ...
        'NonNegative is supported only for explicit ode15s problems.');
end
report=struct('Passed',true,'Backend',backend,'EquationForm',form, ...
    'StateCount',n,'SampleTime',sampleTime, ...
    'SampleCallbackInf',norm(value,inf));
end
