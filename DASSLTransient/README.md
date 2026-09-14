# DASSLTransient

A small class-based MATLAB adapter for stiff initial-value problems. It offers
one interface for:

- explicit stiff ODEs with `ode15s`;
- mass-matrix systems `M(t,y)*yp = f(t,y)` with `ode15s`;
- fully implicit residual systems `F(t,y,yp) = 0` with `ode15i`.

The name describes the DASSL-style problem class (implicit variable-order
backward differentiation), not a bundled copy of the LLNL DASSL source. MATLAB
`ode15s` normally uses NDF formulas; this wrapper enables its BDF option by
default. Fully implicit residual problems use `ode15i` and optional `decic`
consistent-initial-condition calculation.

## Explicit or mass-matrix form

```matlab
problem.RHS = @(t,y) [-1000*(y(1)-cos(t))-sin(t)];

solver = dassl.TransientSolver(struct( ...
    "Backend","ode15s", "RelTol",1e-7, "AbsTol",1e-9));
result = solver.solve(problem,linspace(0,1,101),1);
```

Add only `fullfile(root,"Solver")` to the MATLAB path for normal use.

For `M*yp=f`, additionally set `problem.Mass` to a numeric matrix or function
handle. Optional problem fields include `Jacobian`, `JacobianPattern`,
`MassStateDependence`, `MassVectorPattern`, `MassSingular`, `Events`,
`OutputFcn`, and `ValidateState`.

For `ode15s`, `JacobianPattern` is the numeric `df/dy` sparsity matrix. For
`ode15i`, MATLAB requires the cell pair `{dF/dy pattern,dF/dyp pattern}`.
A mass callback is `M(t)` when `MassStateDependence="none"`, otherwise it is
`M(t,y)`.

Likewise, an `ode15s` Jacobian is one matrix or `J(t,y)`, whereas an `ode15i`
Jacobian is `{dF/dy,dF/dyp}` or a two-output function
`[dFdy,dFdyp]=J(t,y,yp)`. `NonNegative` is accepted only for an explicit
`ode15s` problem because MATLAB ignores it for residual DAEs and mass-matrix
systems.

`Events` uses MATLAB's backend signature: `(t,y)` for `ode15s` and
`(t,y,yp)` for `ode15i`. Output-time ranges are checked before integration.

## Residual form

```matlab
problem.Residual = @(t,y,yp) [yp(1)+y(1); y(2)-y(1)^2];
problem.FixedY0 = [true;false];
problem.FixedYP0 = [false;true];

solver = dassl.TransientSolver(struct("Backend","ode15i"));
result = solver.solve(problem,[0,1],[1;.8],[-1;-2]);
```

With `ComputeConsistentInitialConditions=true` (the default for `ode15i`),
`decic` adjusts the unfixed entries before integration.

## Output

The result contains the raw MATLAB solution structure, requested output times,
states and derivatives, solver-returned initial state and derivative, original input initial
values, elapsed time, internal mesh
point count, backend name, and a final residual audit. The core performs no
periodic iteration, file output, domain postprocessing, or physical closure.

Timing is split into consistent-initialization, integration, and total elapsed
seconds. `ElapsedSeconds` is retained as an alias of `TotalElapsedSeconds`.
The functional shortcut is
`dassl.solve(problem,timeSpan,y0,options,yp0)`; the final argument is optional.
Set `RuntimeChecks=true` only when per-callback finite-value checks are wanted.
For an explicit ODE, `InitialDerivative` is evaluated directly from the RHS;
for a mass-matrix or residual DAE it is the solver interpolant derivative.
`FinalResidualInf` is a dense-output diagnostic, not an independent proof of
the solver's internal nonlinear tolerance.

## Tests

```matlab
runDASSLTransientTests
```

## Optional validation layer

`Tools` is not needed by the solver. Add it only while integrating a model:

```matlab
addpath(fullfile(root,"Tools"))
report = dassltools.validateProblem(problem,y0,yp0,"ode15i");
diagnostics = dassltools.debug(problem,[0,1],y0,yp0,options);
```

The first call probes callback shapes and backend-specific contracts without
integrating. The second performs a diagnostic solve with statistics enabled.

The packaged tests were run with MATLAB R2026a.
