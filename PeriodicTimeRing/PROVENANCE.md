# Provenance and extraction boundary

This package was extracted from the periodic time-ring research implementation
in the `SimpleMoreResearch` workspace.  The reusable numerical ideas retained
here are:

- odd-node Fourier-exact periodic differentiation;
- Fourier interpolation between time rings;
- simultaneous residual assembly over all time nodes;
- structural Jacobian assembly from local state/derivative patterns;
- graph-colored finite differences and damped sparse Newton iteration.

The application-specific state layout, component names, pressure/inventory
closure, legacy transient seed, snapshot mapping, grid-study scripts, and
thermal/work post-processing were deliberately excluded.  A generic row
replacement interface is supplied for application-owned integral or gauge
conditions.

The extracted implementation also tightens several numerical contracts:

- arbitrary positive period and finite time origin;
- columnized scales and bounds (avoiding accidental implicit expansion);
- inclusive bounds and bound-aware finite differences;
- optional structural-pattern verification;
- residual callback errors are not silently converted into line-search
  rejection.

This is an independently reviewable solver library, not a claim of identity
with any proprietary solver.

