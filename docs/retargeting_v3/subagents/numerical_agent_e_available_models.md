# Numerical Agent E Handoff: Available Models

Scope: before/after artifact matrix for available models.

Findings incorporated:

- Old `artifacts/retargeting_v3_step2/` is only the baseline.
- Step 2.1 needs a separate artifact directory and explicit before/after counts.
- Semantic, model-load, source-unavailable, and license/source issues must remain unchanged.

Integrated result:

- Added `scripts/write_retargeting_v3_numerical_artifacts.py`.
- Generated `artifacts/retargeting_v3_step2_numerical/`.
- Added `baseline.json`, `before_after.json`, `threshold_calibration.json`, and numerical summary fields.
- Current default-sample artifact reports:
  - `baseline_pass=7`
  - `corrected_pass=16`
  - `baseline_algorithm_failed=14`
  - `corrected_algorithm_failed=5`
  - `epsilon_only_failures_before=14`
  - `epsilon_only_failures_after=0`
  - unchanged semantic/model-load/source-unavailable counts remain unchanged.
