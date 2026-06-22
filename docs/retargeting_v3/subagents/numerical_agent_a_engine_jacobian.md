# Numerical Agent A Handoff: Engine Jacobian

Scope: engine relative Jacobian interface and semantic-site formula.

Findings incorporated:

- Dedicated `engine_jacobian.py` was required; placing partial engine code inside FD code was not acceptable.
- MuJoCo must use `mj_jac` at both semantic site points.
- Newton must use `newton.eval_jacobian`; FD fallback must not be reported as engine evidence.
- Relative formula must support cross-branch tasks and reference/target local offsets.

Integrated result:

- Added `EngineRelativeJacobian` with `translation`, `rotation`, `scalar_dtype`, `source`, `finite`, and `convention`.
- Added MuJoCo and Newton engine paths.
- Added cross-branch/local-offset tests in `tests/v3/test_engine_jacobian.py`.
- Removed the old Newton crosscheck path that used `body_id * 6` and skipped cross-branch cases.
