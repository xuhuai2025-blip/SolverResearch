# Reusable MATLAB solver kernels

This review bundle contains three independent numerical libraries. None of
them depends on a thermodynamic model, a machine topology, or a particular
application domain.

| Folder | Mathematical problem | Time treatment | Primary use |
|---|---|---|---|
| `TVD2FiniteVolume` | One-dimensional conservation laws | SSP-RK2/SSP-RK3 initial-value marching | Non-periodic transient transport |
| `DASSLTransient` | Stiff ODEs, mass-matrix DAEs, residual DAEs | MATLAB `ode15s`/`ode15i` | General stiff transient simulation |
| `PeriodicTimeRing` | Square periodic ODE/DAE residual systems | Whole-period collocation and sparse damped Newton | Direct periodic steady-state solution |

The libraries are intentionally separate, but may be composed:

- `TVD2FiniteVolume` can provide a conservative spatial residual to
  `DASSLTransient` when the semi-discrete system is stiff.
- The same spatial residual can be supplied to `PeriodicTimeRing` when the
  desired solution is periodic rather than an initial-value trajectory.

Each folder contains its own README, provenance note, examples, and tests.
No Git repository has been initialized; this directory is intended for review
before publication.

Every library has the same two-layer layout:

- `Solver`: the normal execution path and the only directory needed in daily
  use;
- `Tools`: optional `validateProblem` and `debug` entry points, kept off the
  MATLAB path unless deeper diagnosis is requested.

Tests temporarily add both layers. Examples add only `Solver`, demonstrating
that validation tooling is not on the production path by default.

See `VALIDATION.md` for the MATLAB R2026a test, example, and static-analysis
record produced before delivery.
