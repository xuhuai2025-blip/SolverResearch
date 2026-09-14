# PeriodicTimeRing

`PeriodicTimeRing` is a reusable MATLAB solver for periodic ODE and index-1
DAE boundary-value problems.  It represents one period on an odd, uniform
time ring, differentiates the complete periodic waveform spectrally, and
solves the simultaneous residual equations with a sparse damped Newton
method.

The library contains no machine topology, thermodynamic property model,
mean-pressure rule, or application post-processing.

## Core contract

Store state variables by row and time-ring nodes by column:

```matlab
addpath(fullfile(root,"Solver"))
grid = periodicring.uniformGrid(7, period, origin);

problem.StateCount = 2;
problem.LocalResidual = @(t,x,xdot,p) [ ... ];
problem.StatePattern = sparse([ ... ]);       % dR/dx structure
problem.DerivativePattern = sparse([ ... ]);  % dR/dxdot structure

compiled = periodicring.compile(problem, grid);
result = periodicring.solve(compiled, initialGuess, options, parameters);
```

At node `j`, `LocalResidual` must return a column vector
`F(t(j),X(:,j),Xdot(:,j),parameters)`.  The solver forms
`Xdot = X*grid.D.'` and solves all node equations together.

The two local sparsity patterns generate the space-time pattern

```text
kron(I, StatePattern) + kron(spones(D), DerivativePattern)
```

and graph coloring reduces the number of finite-difference residual calls.

## Global constraints

A periodic differential equation may have a free constant or a DAE may need
one integral/gauge condition.  Replace, rather than append, the same number
of local residual rows:

```matlab
problem.GlobalRows = periodicring.flatIndex(1, 1, problem.StateCount);
problem.GlobalResidual = @(grid,X,p) ...;     % one value here
problem.GlobalPattern = sparse(ones(1,problem.StateCount*grid.NodeCount));
```

This generic mechanism can express a prescribed mean, total inventory, phase
condition, or another application-owned closure without putting that closure
inside the solver.

## Files

- `uniformGrid`, `differentiate`, `integrate`, `resample`: time-ring tools.
- `compile`: validates the problem and caches the sparse pattern/coloring.
- `solve`: assembles and solves the periodic residual.
- `solveNewton`: reusable scaled, bounded, damped sparse Newton solver.
- `examples/runForcedLinear.m`: manufactured periodic ODE example.
- `tests/TestPeriodicTimeRing.m`: harmonic, ODE, DAE, global-constraint, and
  structural-pattern tests.

Run `runPeriodicTimeRingTests` from this directory.  No optional toolbox is
required beyond MATLAB functions used by the base numerical environment.

## Optional validation layer

Normal runs add only `Solver`. During model development, temporarily add
`Tools` and explicitly invoke the slower checks:

```matlab
addpath(fullfile(root,"Tools"))
report = periodicringtools.validateProblem(problem,grid,initialGuess,parameters);
audit = periodicringtools.debug(compiled,initialGuess,parameters);
```

`debug` compares finite-difference changes with the declared sparsity pattern,
so omitted dependencies are detected before colored differences are trusted.
Neither tool is called by the production solver.
The compiled problem defaults to `RuntimeChecks=false`; set the problem field
true only when per-node residual checks are useful during integration work.
For a coupled nonlinear feasibility region, provide an interior initial guess;
the finite-difference backend changes direction for box bounds but does not
construct tangent perturbations for an arbitrary coupled constraint surface.

## Scope and limits

- The node count must be odd and at least three; the endpoint is not repeated.
- Smooth periodic waveforms are the intended use.  Discontinuities converge
  poorly in a global Fourier representation.
- The nonlinear system must be square after row replacements.
- A converged algebraic residual is not by itself proof that the physical
  model is correct; conservation and application outputs still need separate
  acceptance tests.
