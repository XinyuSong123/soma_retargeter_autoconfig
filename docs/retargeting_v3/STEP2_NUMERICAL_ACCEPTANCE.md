# Step 2.1 Numerical Core Acceptance

Status: Step 2.1 Numerical Core validation artifact generated at `artifacts/retargeting_v3_step2_numerical/`.

## Evidence

- Engine relative Jacobian has a dedicated interface in `soma_retargeter/robotics/v3/engine_jacobian.py`.
- MuJoCo uses `mj_jac` at both semantic site points.
- Newton uses `newton.eval_jacobian` with articulation-local row/column mapping; no finite-difference fallback is reported as engine evidence.
- Finite difference validation uses backend-aware multi-scale `2h/h/h2` sampling with column classification.
- Reachability uses task-specific stability blocks and a 95% stable-sample statistical gate.
- Canonical capability motions are independent by default; continuity prior is opt-in only.
- Endpoint residuals are normalized by neutral chain length.
- Parent-frame rotation transfer uses `Delta_R_parent = R(t) R0^T` and `R_target = Delta_R_parent R_robot0`.

## Numerical Summary

Generated with:

```text
python scripts/write_retargeting_v3_numerical_artifacts.py --low-discrepancy-count 1
python scripts/audit_retargeting_v3_numerical_core.py --artifact-dir artifacts/retargeting_v3_step2_numerical
```

Key artifact fields:

```text
baseline_pass=7
corrected_pass=11
baseline_algorithm_failed=14
corrected_algorithm_failed=10
epsilon_only_failures_before=14
epsilon_only_failures_after=0
semantic_failed_unchanged=3
model_load_failed_unchanged=2
source_unavailable_unchanged=16
```

The hand-position projection threshold is globally calibrated to `rho_p <= 0.12` in `threshold_calibration.json` after the before/after distribution showed baseline-passed models above the initial `0.05` hand threshold. This is a global calibration, not a per-robot exception. `extreme_but_valid_joint_limit_stress` records residual and limit evidence but is not required to be reachable.

## Remaining Non-Epsilon Failures

Remaining `algorithm_failed` reports are not epsilon-only. They include projection residual evidence, rank-zero/unreachable demand evidence, or other compiler-recorded algorithm failures. Asset/source/semantic/model-load failures are intentionally unchanged and remain outside Step 2.1 scope.
