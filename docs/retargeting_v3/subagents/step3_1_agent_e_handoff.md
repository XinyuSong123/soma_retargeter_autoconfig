# Step 3.1 Agent E Handoff

Scope: orchestration, artifacts, and determinism.

Files:

- `soma_retargeter/tools/run_v3_full_fleet_runtime_quality.py`
- `scripts/run_step3_full_fleet_runtime_quality.sh`
- `tests/v3/test_step3_full_fleet_artifact_schema_runtime_quality.py`
- `tests/v3/test_step3_full_fleet_determinism_runtime_quality.py`

Result:

- Full artifact tree is generated under `artifacts/retargeting_v3_step3_runtime_quality`.
- `model_matrix.json` contains all 44 rows.
- Required per-model and per-clip evidence is written.
- Determinism records all 44 rows as compared/matched, with runtime duration treated as diagnostic timing rather than identity.
- Full repo pytest is classified explicitly instead of claimed by the artifact writer.
