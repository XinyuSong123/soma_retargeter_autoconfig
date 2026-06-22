# Numerical Agent A Handoff

Reasoning strength: xhigh

Scope: engine relative Jacobian and runtime twist conventions.

Final files:

- `soma_retargeter/robotics/v3/engine_jacobian.py`
- `soma_retargeter/robotics/v3/spatial.py`
- `tests/v3/test_engine_jacobian.py`

Validation:

- `PYTHONPATH=. pytest -q tests/v3/test_engine_jacobian.py`
- `PYTHONPATH=. python scripts/audit_retargeting_v3_numerical_core.py --artifact-dir artifacts/retargeting_v3_step2_numerical`

Result: PASS. Engine Jacobians use MuJoCo `mj_jac` and Newton `eval_jacobian`, include semantic site offsets, record backend/source/dtype/convention metadata, and are the primary profile/reachability evidence.

Related detailed note: `numerical_agent_a_engine_jacobian.md`.
