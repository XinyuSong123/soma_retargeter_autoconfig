# Step 3.1 Agent C Handoff

Scope: generic full-fleet runtime harness and smoke solver.

Files:

- `soma_retargeter/runtime/v3/fleet_harness.py`
- `soma_retargeter/runtime/v3/generic_smoke.py`
- `soma_retargeter/runtime/v3/runtime_status.py`

Result:

- Full humanoids enter generic runtime FK residual smoke.
- Partial humanoids run supported-semantics smoke and never full override.
- Negative controls run runtime load/rejection smoke and are not promoted.
- The solver path uses runtime model FK residual evaluation, not screenshot/video evidence and not production `NewtonPipeline` fake configs.

Validation is covered by full-fleet artifact schema, determinism, and audit tests.
