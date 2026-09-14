# Provenance and extraction boundary

The reviewed source used MATLAB `ode15s` inside application-specific cycle
solvers:

```text
D:\VibeCodingWorkSpace\SIMPLE模型开发\SIMPLE_Experimental\+physics\+solvers\solveCoupledCooler.m
D:\VibeCodingWorkSpace\SIMPLE模型开发\SIMPLE_Experimental\+physics\+solvers\solveCoupledHeatPump.m
```

At extraction time these files called `ode15s(@rhs,...)` without a mass matrix,
`ode15i`, `decic`, or a separately implemented DASSL/DASPK kernel. Pressure,
flow, material laws, periodic shooting, Anderson acceleration, and result
accounting were embedded in the application functions and are intentionally
excluded here.

This package is a new adapter around MATLAB solvers. No application RHS code
and no DASSL kernel were copied into it.

Source audit snapshot (SHA-256, 2026-09-14):

```text
solveCoupledCooler.m
F13A2D0CDFF4BF8335E6A739F96C5A1423723B0F3DB578EF69C742488E79A2CA

solveCoupledHeatPump.m
31CA0E882AF32FEA6CD59A5DFD4A1D45EF4286E3C4FC5D12ADA1CB97A459ECCD
```

The source files were untracked in the reviewed `experiment/local-direction`
checkout at commit `3eef560`; the hashes, not that commit, identify the audited
working copies.

Extraction date: 2026-09-14.
