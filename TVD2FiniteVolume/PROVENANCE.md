# Provenance and extraction boundary

The limiter and conservative one-face/one-flux principle were derived from the
reviewed MUSCL/minmod transport implementation in:

```text
D:\VibeCodingWorkSpace\回热器节点研究\+validation\faceEnthalpy.m
D:\VibeCodingWorkSpace\回热器节点研究\+validation\implicitResidual.m
```

The reusable library deliberately removes component groups, thermodynamic
enthalpy, graph topology, periodic convergence, and application-specific
source terms. It adds explicit boundary policies, nonuniform coordinates,
generic numerical flux callbacks, SSP time marching, diagnostics, and tests.

Extraction date: 2026-09-14.

