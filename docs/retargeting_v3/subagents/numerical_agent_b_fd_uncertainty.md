# Numerical Agent B Handoff: FD Uncertainty

Scope: backend-aware finite-difference validation, multi-scale estimates, and column classification.

Findings incorporated:

- The old h/h2 epsilon-halving gate was a one-strike hard failure.
- Newton/Warp float32 and MuJoCo float64 need different epsilon floors.
- Physical near-zero columns must not be treated as nonsmooth instability.

Integrated result:

- `coordinate_epsilon()` is backend-aware.
- `numerical_relative_jacobian()` evaluates `2h`, `h`, and `h/2`.
- Richardson estimates and per-column noise diagnostics are recorded.
- Column classifications include `stable_nonzero`, `numerically_zero`, `unstable_roundoff`, `unstable_nonsmooth`, and `nonfinite`.
- Existing compatibility fields remain in JSON, but new numerical gates use the richer diagnostics.
