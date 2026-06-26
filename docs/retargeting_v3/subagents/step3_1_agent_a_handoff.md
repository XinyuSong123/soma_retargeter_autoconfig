# Step 3.1 Agent A Handoff

Scope: fleet inventory, runtime source resolution, and runtime-local profile closure.

Files:

- `soma_retargeter/runtime/v3/fleet_inventory.py`
- `soma_retargeter/runtime/v3/runtime_local_profile.py`
- `soma_retargeter/tools/compile_runtime_local_profiles_v3.py`
- `tests/v3/test_step3_fleet_inventory_runtime_quality.py`
- `tests/v3/test_step3_runtime_local_profile_runtime_quality.py`

Result:

- 44 in-scope rows derived from lock/profile artifacts.
- Category counts are 32 full, 3 partial, 9 negative.
- Runtime sources resolve and are materialized.
- Runtime profile closure produces 32 `profile_match`, 3 `structured_partial_supported`, and 9 `negative_control_rejected`.
- No Step 2 artifacts are modified.

Validation:

```bash
PYTHONPATH=. python -m pytest -q \
  tests/v3/test_step3_fleet_inventory_runtime_quality.py \
  tests/v3/test_step3_runtime_local_profile_runtime_quality.py
```
