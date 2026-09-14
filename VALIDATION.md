# Validation record

Validation date: 2026-09-14  
Environment: MATLAB R2026a on Windows

## Automated tests

| Library | Passed | Failed | Incomplete |
|---|---:|---:|---:|
| TVD2FiniteVolume | 14 | 0 | 0 |
| DASSLTransient | 12 | 0 | 0 |
| PeriodicTimeRing | 11 | 0 | 0 |
| **Total** | **37** | **0** | **0** |

The tests include non-periodic stretched-grid second-order convergence,
strongly stretched-grid bounded/TVD transport for minmod, MC, and van Leer,
SSP stage-speed CFL retry, stiff ODEs, mass-matrix and residual DAEs,
backend-specific MATLAB contracts, Fourier differentiation/interpolation,
periodic ODE/DAE manufactured solutions, a global mean constraint, and sparse
Jacobian-pattern auditing.

## Examples

| Example | Audit value |
|---|---:|
| TVD non-periodic advection | 115 accepted time steps |
| Stiff ODE | maximum error `4.358e-08` |
| Index-1 DAE | maximum error `3.845e-08` |
| Periodic time ring | residual infinity norm `8.882e-16`; waveform error `7.772e-16` |

## Static analysis

MATLAB Code Analyzer reported zero messages across all solver, tool, example,
test, and runner `.m` files.

## Production/debug separation

Examples add only each library's `Solver` directory. Test runners temporarily
add `Solver` and `Tools`, then restore the caller's original MATLAB path.
Optional `validateProblem` and `debug` routines are therefore outside the
normal calculation path by default.
