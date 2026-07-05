# GOAL.md — Retargeting V3 Step 4.4 Super Goal

Date: 2026-07-05
Repository: `XinyuSong123/soma_retargeter_autoconfig`
Target branch: `retargeting-v3-step4-4-release-quality-v2-validation`
Base branch: `retargeting-v3-step4-3-normalized-residual-gate-reconciliation`
Project environment: `conda activate soma-retargeter-v2`
Do not use `env_isaaclab` for V3 tests.

This is the single execution goal for the next Codex window after `git pull`.

Codex must treat this as a fail-closed release-quality validation task. It is not a cosmetic cleanup task. It must patch every known weakness from Step 3.1.1 through Step 4.3, preserve honest evidence, validate the release-quality-v2 candidate gate, and either improve candidate coverage with global auditable methods or produce a complete honest blocked diagnosis.

---

## 0. Operating principle

The project must not become a green dashboard created by relabeling warnings. Every claim must be backed by artifacts, deterministic reruns, audits, tests, Git LFS checks, and final-head GitHub Actions evidence.

Correct outcomes are limited to:

- `PASS_RC` because a real global auditable improvement was made and all hard invariants passed.
- `BLOCKED_RELEASE_QUALITY_V2_VALIDATION` because no coverage improved but diagnostics are complete and honest.
- `BLOCKED_CANDIDATE_GATE_ROBUSTNESS` because candidate rows are unstable under required stress tests.
- `BLOCKED_CLIP_GENERALIZATION` because apparent improvements do not generalize across clips or hide worst-clip failures.
- `BLOCKED_PIPELINE_REGRESSION` because an invariant regressed.
- `BLOCKED_CI_OR_PROVENANCE` because final-head CI, LFS, or provenance evidence is incomplete.

Do not manufacture success by changing labels, weakening gates, deleting hard cases, or hiding raw residual regressions.

---

## 1. Current baseline truth from Step 4.3

Current completed stage: Step 4.3 normalized residual gate reconciliation.

Completed branch:

```text
retargeting-v3-step4-3-normalized-residual-gate-reconciliation
```

Recommended Step 4.4 branch:

```text
retargeting-v3-step4-4-release-quality-v2-validation
```

Step 4.3 artifact directory:

```text
artifacts/retargeting_v3_step4_3_normalized_residual_gate_reconciliation/
```

Step 4.3 handoff:

```text
docs/retargeting_v3/STEP4_3_NORMALIZED_RESIDUAL_GATE_RECONCILIATION_HANDOFF.md
```

Step 4.3 core baseline counts:

```text
full_fleet_rows = 44
full_humanoid_rows = 32
partial_profile_rows = 3
negative_control_rows = 9
solver_backed_smoke_attempted_count = 32
solver_backed_completed_count = 32
solver_backed_count = 32
residual_only_count = 0
runtime_quality_passed_count = 0
runtime_quality_warned_count = 32
runtime_quality_failed_count = 0
high_residual_warning_count = 32
partial_runtime_passed_count = 3
negative_control_runtime_passed_count = 9
trajectory_exports_count = 128
temporal_continuity_finite_count = 128
support_contact_diagnostic_count = 128
collision_proxy_diagnostic_count = 128
deterministic_compared_count = 44
deterministic_matched_count = 44
focused_tests = 63 passed
local_audit_status = PASS_RC
local_audit_blocking_count = 0
Git_LFS_fsck = OK
```

Step 4.3 release-quality-v2 candidate baseline:

```text
legacy_gates_unchanged = true
candidate_release_gates_defined = true
candidate_release_gates_active = true
production_default_changed = false
runtime_override_default_enabled = false
normalization_policy_selected = candidate_1_fixed_global_body_scale_normalization
gate_policy_selected = candidate_6_release_quality_gate_v2_candidate
selected_fixed_global_scale = 4.268140459384
selected_fixed_global_scale_source = closed_step4_1_raw_task_residual_p95_p95_distribution
raw_residual_regression_count = 0
denominator_inflation_detected = false
normalization_hides_raw_residual_regression = false
rows_below_legacy_warn_gate = 0
rows_below_candidate_warn_gate = 6
rows_below_candidate_pass_gate = 0
release_quality_candidate_passed_count = 0
release_quality_candidate_warned_count = 6
release_quality_candidate_blocked_count = 26
```

Step 4.3 candidate-warned rows:

```text
fourier_n1_mjcf
fourier_n1_urdf
robotis_op3_mjcf
toddlerbot_2xc_mjcf
toddlerbot_2xm_mjcf
toddlerbot_urdf
```

Important interpretation:

- Legacy runtime quality still has zero passes.
- All 32 full humanoids are still legacy warned.
- The v2 candidate gate is separate from legacy runtime quality.
- The six candidate-warned rows are not legacy passes.
- There are zero candidate-passed rows.
- Twenty-six full humanoid rows remain blocked.
- The Step 4.4 task is to validate, stress, and possibly expand candidate coverage, not to claim deployment readiness.

---

## 2. Historical problems that must now be patched or guarded

### 2.1 Step 3.1.1 problem — honest evidence but no solver-backed quality

Observed earlier problem:

```text
32 full / 3 partial / 9 negative
runtime_quality_passed_count = 0
runtime_quality_warned_count = 23
runtime_quality_failed_count = 9
solver_backed_count = 0
residual_only_count = 32
```

Patch requirement:

- Preserve honest labels.
- Never relabel warning or failure as pass.
- Never confuse evidence completeness with runtime quality success.
- Preserve full fleet rows and negative controls.
- Keep all claims tied to artifacts.

Step 4.4 guard:

- Audit must fail if any candidate warning is counted as a legacy runtime pass.
- Audit must fail if runtime quality counts are inconsistent with raw matrix rows.
- Audit must fail if closed Step 3.1.1 artifacts are modified.

### 2.2 Step 3.2 problem — solver-backed coverage added but quality still failed or warned

Observed earlier problem:

```text
solver_backed_count = 32
residual_only_count = 0
runtime_quality_passed_count = 0
runtime_quality_warned_count = 23
runtime_quality_failed_count = 9
```

Patch requirement:

- Solver-backed coverage must stay 32 of 32 for full humanoids.
- Residual-only fallback must remain 0.
- Solver-backed smoke coverage cannot be used to claim quality pass.

Step 4.4 guard:

- Audit fails if solver_backed_count regresses below 32.
- Audit fails if residual_only_count is nonzero.
- Audit fails if solver success fields are missing.

### 2.3 Step 3.3 problem — hard failures removed but high residual warnings remained

Observed earlier improvement:

```text
runtime_quality_failed_count: 9 -> 0
joint_limit_warning_count: 12 -> 1
joint_limit_smoke_warning_count: 10 -> 0
runtime_quality_passed_count = 0
runtime_quality_warned_count = 32
high_residual_warning_count = 32
```

Patch requirement:

- Hard runtime failures must remain 0.
- Joint limit regressions must not return.
- High residual warnings must be explained honestly, not hidden.

Step 4.4 guard:

- Audit fails if runtime_quality_failed_count is not 0.
- Audit fails if joint limit warnings regress without explanation.
- Audit fails if high residual warnings vanish only because thresholds changed.

### 2.4 Step 3.4 problem — coverage complete but residual quality still blocked

Observed earlier improvement:

```text
task_coverage_mean = 1.0
task_coverage_min = 1.0
anchor_reliability_mean = 0.994791666667
anchor_reliability_min = 0.833333333333
runtime_quality_passed_count = 0
runtime_quality_warned_count = 32
high_residual_warning_count = 32
```

Patch requirement:

- Coverage completeness must remain intact.
- Anchor reliability must not regress silently.
- Residual quality must be improved or diagnosed, not masked by coverage metrics.

Step 4.4 guard:

- Audit verifies task coverage and anchor reliability fields exist.
- Blocker taxonomy must distinguish coverage, anchor, residual, solver, temporal, contact, collision, and morphology blockers.

### 2.5 Step 4.0 problem — full pipeline scaffold complete but residual quality blocked

Observed earlier artifacts:

```text
full_pipeline_matrix.json
clip_matrix.json
trajectory_export_manifest.json
temporal_continuity_matrix.json
support_contact_diagnostics.json
collision_proxy_diagnostics.json
normalization_audit.json
orientation_residual_taxonomy.json
red_team_report.json
```

Observed status:

```text
release_candidate_status = BLOCKED_RESIDUAL_QUALITY
runtime_quality_passed_count = 0
runtime_quality_warned_count = 32
runtime_quality_failed_count = 0
trajectory_exports_count = 128
temporal_continuity_finite_count = 128
support_contact_diagnostic_count = 128
collision_proxy_diagnostic_count = 128
```

Patch requirement:

- Full pipeline artifacts must remain complete and finite.
- Diagnostics are necessary but not sufficient for pass.
- No visual or deployment readiness claim is allowed.

Step 4.4 guard:

- Audit fails if trajectory exports or temporal diagnostics are missing or non-finite.
- Audit fails if support/contact/collision proxy diagnostics are missing.
- Audit fails if any Step 4.4 artifact claims visual-ready or deployment-ready status.

### 2.6 Step 4.1 problem — orientation semantics improved but legacy runtime quality still warned

Observed earlier improvement:

```text
selected_policy = parent_relative_runtime_inv_target
Step_4_0_p95_rotation_residual_p95 = 2.90919255987585
Step_4_1_p95_rotation_residual_p95 = 2.0492009223205
p95_rotation_residual_p95_delta = -0.859991637555
raw_residual_regression_count = 0
normalization_hides_raw_residual_regression = false
robot_specific_tuning_used = false
production_default_changed = false
runtime_override_default_enabled = false
runtime_quality_passed_count = 0
runtime_quality_warned_count = 32
runtime_quality_failed_count = 0
```

Patch requirement:

- Keep global parent-relative orientation residual semantics.
- Do not change production defaults silently.
- Do not enable runtime override by default.
- Do not introduce robot-specific orientation math.

Step 4.4 guard:

- Audit verifies selected orientation policy is preserved when used for scoring.
- Audit fails on robot-specific math, robot-specific thresholds, whitelists, blacklists, or per-robot IK weights.

### 2.7 Step 4.2 problem — active orientation scoring improved residuals but gates still blocked

Observed earlier improvement:

```text
orientation_policy_active_for_scoring = true
p95_orientation_integrated_residual = 3.309012625858
p95_orientation_integrated_residual_delta_vs_step4_1 = -0.959127833526
p95_normalized_task_residual_p95_delta_vs_step4_1 = -0.006465520975
raw_residual_regression_count = 0
normalization_hides_raw_residual_regression = false
denominator_inflation_detected = false
rows_newly_passing = 0
rows_still_warned = 32
pass_gate_thresholds_unchanged = true
warn_gate_thresholds_unchanged = true
gates_weakened = false
high_task_residual = 32
normalized_task_residual_p95_above_pass_gate = 32
normalized_task_residual_p95_above_warn_gate = 32
solver_convergence_weak = 1
solver_success_fraction_below_one = 1
```

Patch requirement:

- Keep scoring improvement evidence separated from pass claims.
- Investigate the remaining solver convergence weakness.
- Preserve gate thresholds unless a new candidate gate is explicitly separate and audited.

Step 4.4 guard:

- Audit fails if legacy pass/warn thresholds are changed silently.
- Audit fails if candidate gate status leaks into legacy runtime_quality_passed_count.
- Blocker taxonomy must identify any solver convergence rows.

### 2.8 Step 4.3 problem — legacy row-local normalization explained but v2 candidate is not yet stable or promoted

Observed Step 4.3 finding:

```text
legacy_normalization_formula = residual / max(1.0, current_row_raw_residual_max)
denominator_scope = row_local_legacy_metric
normalized_max_all_rows_equal_one = true
rows_improve_raw_but_remain_legacy_normalized_blocked_count = 32
```

Observed Step 4.3 v2 candidate state:

```text
selected_fixed_global_scale = 4.268140459384
rows_below_candidate_warn_gate = 6
rows_below_candidate_pass_gate = 0
release_quality_candidate_passed_count = 0
release_quality_candidate_warned_count = 6
release_quality_candidate_blocked_count = 26
```

Patch requirement:

- Candidate v2 gate must be stress-tested before promotion.
- Fixed global scale must not become a hidden denominator inflation trick.
- Candidate-warned rows must be stable across clips.
- Candidate pass must require worst-clip and hard-safety evidence.
- Blocked rows must receive a complete taxonomy.

Step 4.4 guard:

- Audit fails if candidate denominator is row-local.
- Audit fails if denominator inflation is detected.
- Audit fails if candidate-warned deep audit is missing.
- Audit fails if stress tests are missing.
- Audit fails if promotion readiness is missing.

---

## 3. Absolute prohibitions

Do not do any of the following:

```text
modify Step 2 artifacts or thresholds
overwrite closed Step 3.1.1 artifact tree
overwrite closed Step 3.2 artifact tree
overwrite closed Step 3.3 artifact tree
overwrite closed Step 3.4 artifact tree
overwrite closed Step 4.0 artifact tree
overwrite closed Step 4.1 artifact tree
overwrite closed Step 4.2 artifact tree
overwrite closed Step 4.3 artifact tree
silently weaken legacy runtime gates
count release_quality_candidate_warned as legacy runtime_quality_passed
count release_quality_candidate_passed as legacy runtime_quality_passed unless legacy gates independently pass
use robot-specific math
use robot-specific thresholds
use robot-specific whitelists
use robot-specific blacklists
use per-robot IK weights
use per-clip whitelist behavior
remove hard robots to improve metrics
remove hard clips to improve metrics
hide raw residual regression
inflate candidate denominator
reintroduce row-local candidate denominator
change production defaults silently
enable runtime override by default
claim visual readiness
claim deployment readiness
start Step 4.5 automatically
```

---

## 4. Step 4.4 primary goal

Goal name:

```text
Step 4.4 — Release-Quality V2 Validation / Candidate Pass Expansion
```

Primary target:

```text
release_quality_candidate_passed_count > 0
OR rows_below_candidate_warn_gate > 6
OR release_quality_candidate_blocked_count < 26
OR produce honest BLOCKED_RELEASE_QUALITY_V2_VALIDATION with complete diagnostics.
```

Preferred target:

```text
release_quality_candidate_passed_count >= 2
OR rows_below_candidate_warn_gate >= 10
OR release_quality_candidate_blocked_count <= 22
```

Stretch target:

```text
release_quality_candidate_passed_count >= 4
OR rows_below_candidate_warn_gate >= 16
OR release_quality_candidate_blocked_count <= 16
```

Gold target:

```text
release_quality_candidate_passed_count >= 8
rows_below_candidate_warn_gate >= 20
runtime_quality_failed_count = 0
legacy gates unchanged
candidate gate separate from legacy runtime quality
```

Allowed global improvement directions:

```text
global clip aggregation policy with worst-clip guard
global task-class residual weighting
global solver retry or line-search refinement
global temporal-consistency residual penalty
global contact/support/collision guard integration
global release-quality-v2 score reconciliation
global normalization integrity validation
```

Not allowed improvement directions:

```text
robot-specific tuning
per-clip whitelist
threshold lowering
removing hard clips
removing hard robots
turning candidate_warned into candidate_passed by label change
mixing v2 candidate status into legacy runtime pass counts
```

---

## 5. Required artifact tree

Create a new artifact tree only:

```text
artifacts/retargeting_v3_step4_4_release_quality_v2_validation/
```

Required files:

```text
environment.json
commands.txt
quality_summary.json
acceptance_ledger.json
full_pipeline_matrix.json
clip_matrix.json
solver_smoke_matrix.json
generic_smoke_matrix.json
deterministic_rerun.json
quality_delta_vs_step4_3.json
release_quality_v2_validation_matrix.json
candidate_warned_deep_audit.json
release_quality_v2_blocker_taxonomy.json
release_quality_v2_stress_test.json
release_quality_v2_promotion_readiness.json
gate_reconciliation_v3_report.json
normalization_integrity_v2_report.json
trajectory_export_manifest.json
temporal_continuity_matrix.json
support_contact_diagnostics.json
collision_proxy_diagnostics.json
pipeline_config.json
solver_config.json
pipeline_controls_reference.json
red_team_report.json
test_results/pytest.txt
test_results/pytest_summary.json
test_results/junit.xml
```

Required artifact-level invariants:

```text
all JSON files are valid
schema version is recorded
source branch is recorded
source commit is recorded
baseline Step 4.3 artifact dir is recorded
Step 4.4 artifact dir is recorded
commands used to generate artifacts are recorded
closed artifact trees are not modified
row count is preserved at 44
full/partial/negative partition is preserved at 32/3/9
32 full humanoids remain solver-backed
residual_only_count remains 0
runtime_quality_failed_count remains 0
deterministic rerun remains 44/44
trajectory exports remain complete and finite
temporal diagnostics remain complete and finite
support/contact diagnostics remain complete
collision proxy diagnostics remain complete
legacy gates remain unchanged
candidate gate remains separate
normalization integrity passes
no raw residual regression hiding
no denominator inflation
no robot-specific tuning
no whitelist or blacklist
no visual/deployment readiness claim
```

---

## 6. Required new audit script

Create:

```text
scripts/audit_retargeting_v3_step4_4_release_quality_v2_validation.py
```

The audit must fail closed if any of the following is true:

```text
required artifacts are missing
required JSON is invalid
44 rows are not preserved
32/3/9 partition is not preserved
solver_backed_count regresses below 32
residual_only_count is not 0
runtime_quality_failed_count is not 0
legacy gates are changed silently
candidate release gates are mixed into legacy runtime_quality_passed_count
candidate_warned_deep_audit is missing
release_quality_v2_blocker_taxonomy is missing
release_quality_v2_stress_test is missing
release_quality_v2_promotion_readiness is missing
deterministic rerun is not 44/44
trajectory exports are missing or non-finite
temporal diagnostics are missing or non-finite
support/contact diagnostics are missing
collision proxy diagnostics are missing
robot-specific tuning is introduced
per-robot solver weights are introduced
thresholds are lowered silently
whitelists appear
blacklists appear
hard robots are removed
hard clips are removed
raw residual regression is hidden
candidate denominator is row-local
candidate denominator is inflated
closed artifact trees are modified
production default is changed silently
runtime override default is enabled
visual readiness is claimed
deployment readiness is claimed
final HEAD CI evidence is missing in strict mode
Git LFS fsck evidence is missing in strict mode
```

Audit output must include:

```text
status
blocking_count
finding_count
findings
quality_status
invariants
candidate_counts
legacy_counts
normalization_integrity
candidate_stability
blocker_taxonomy_summary
final_head_ci
lfs
```

---

## 7. Required tests

Create these tests:

```text
tests/v3/test_step4_4_release_quality_v2_validation_audit.py
tests/v3/test_step4_4_release_quality_v2_validation_artifact_schema.py
tests/v3/test_step4_4_release_quality_v2_validation_candidate_counts.py
tests/v3/test_step4_4_release_quality_v2_validation_stress.py
tests/v3/test_step4_4_release_quality_v2_validation_promotion_readiness.py
tests/v3/test_step4_4_release_quality_v2_validation_no_robot_specific_tuning.py
tests/v3/test_step4_4_release_quality_v2_validation_status_semantics.py
tests/v3/test_step4_4_release_quality_v2_validation_determinism.py
```

Minimum required coverage:

```text
missing validation matrix fails
missing deep audit fails
missing blocker taxonomy fails
missing stress test fails
missing promotion readiness fails
candidate warn/pass not counted as legacy runtime pass
legacy threshold lowering rejected
candidate threshold lowering rejected unless explicitly experimental and not promoted
robot-specific exceptions rejected
per-robot solver weights rejected
candidate pass without hard safety rejected
candidate pass without worst-clip guard rejected
candidate pass without temporal/support/contact/collision diagnostics rejected
candidate warned instability flagged
clip-specific improvement flagged
closed artifacts unchanged
deterministic mismatch rejected
row count regression rejected
partition regression rejected
solver-backed regression rejected
residual-only fallback rejected
normalization denominator inflation rejected
row-local candidate denominator rejected
raw residual regression hiding rejected
production default mutation rejected
runtime override default rejected
visual/deployment claim rejected
strict-mode missing final-head CI rejected
```

Tests must be semantic. Do not create tests that only assert file existence when deeper invariants can be checked.

---

## 8. Required GitHub Actions workflow

Create:

```text
.github/workflows/retargeting_v3_step4_4_release_quality_v2_validation.yml
```

Required jobs:

```text
step4-4-static-and-unit
step4-4-artifact-audit
step4-4-release-quality-v2-validation-smoke
step4-4-candidate-stability-smoke
step4-4-export-and-temporal-smoke
step4-4-lfs-and-snapshot-smoke
step4-4-pipeline-controls-reference
step4-4-final-head-ci-evidence
```

The final-head CI evidence job must print:

```text
workflow_run_id
head_sha
conclusion
job_conclusions
```

The final handoff must copy these values exactly.

---

## 9. Required local commands

Run focused tests:

```bash
PYTHONPATH=. python -m pytest -q \
  tests/v3/test_step4_4_release_quality_v2_validation_*.py \
  tests/v3/test_step4_3_normalized_residual_gate_reconciliation_*.py \
  tests/v3/test_step4_2_orientation_policy_runtime_scoring_*.py
```

Run pipeline:

```bash
PYTHONPATH=. python soma_retargeter/tools/run_v3_full_pipeline_acceptance.py \
  --artifact-dir artifacts/retargeting_v3_step4_4_release_quality_v2_validation \
  --baseline-step4-3-artifact-dir artifacts/retargeting_v3_step4_3_normalized_residual_gate_reconciliation \
  --enable-solver-backed-generic-smoke \
  --enable-global-solver-quality-hardening \
  --enable-global-residual-quality-hardening \
  --enable-parent-relative-orientation-runtime-scoring \
  --enable-normalized-residual-gate-reconciliation \
  --enable-release-quality-v2-validation \
  --enable-full-pipeline-exports \
  --deterministic-rerun
```

Run local audit:

```bash
PYTHONPATH=. python scripts/audit_retargeting_v3_step4_4_release_quality_v2_validation.py \
  --artifact-dir artifacts/retargeting_v3_step4_4_release_quality_v2_validation \
  --baseline-step4-3-artifact-dir artifacts/retargeting_v3_step4_3_normalized_residual_gate_reconciliation \
  --source-root .
```

Run strict audit after CI evidence exists:

```bash
PYTHONPATH=. python scripts/audit_retargeting_v3_step4_4_release_quality_v2_validation.py \
  --artifact-dir artifacts/retargeting_v3_step4_4_release_quality_v2_validation \
  --baseline-step4-3-artifact-dir artifacts/retargeting_v3_step4_3_normalized_residual_gate_reconciliation \
  --source-root . \
  --require-final-head-ci
```

Run LFS check:

```bash
git lfs fsck
```

Record all commands and results in:

```text
artifacts/retargeting_v3_step4_4_release_quality_v2_validation/commands.txt
artifacts/retargeting_v3_step4_4_release_quality_v2_validation/test_results/pytest.txt
artifacts/retargeting_v3_step4_4_release_quality_v2_validation/test_results/pytest_summary.json
```

---

## 10. Required implementation phases

### Phase 0 — Inspect repository and confirm baseline

1. Check branch and HEAD.
2. Confirm Step 4.3 artifact directory exists.
3. Confirm Step 4.3 handoff exists.
4. Check Step 4.3 final-head CI evidence and record if incomplete.
5. Load Step 4.3 quality summary, ledger, and gate reconciliation artifacts.
6. Confirm all baseline counts listed above.
7. Record discrepancies instead of silently patching them.

Deliverables:

```text
baseline section in acceptance_ledger.json
baseline section in final handoff
```

### Phase 1 — Add schemas and artifact writers

1. Add Step 4.4 artifact schema helpers.
2. Add deterministic JSON write helpers.
3. Add fail-closed placeholder support only for blocked status.
4. Ensure artifact writing never touches closed artifact trees.
5. Ensure commands.txt is recorded deterministically.

Deliverables:

```text
schema-valid required artifacts
tests for missing and bad artifacts
```

### Phase 2 — Implement release-quality-v2 validation matrix

1. Load Step 4.3 candidate gate data.
2. Recompute or validate release-quality-v2 scores in the Step 4.4 path.
3. Produce row-level candidate statuses.
4. Add clip-level details.
5. Add worst-clip guard.
6. Add task-class breakdown.
7. Add solver convergence status.
8. Add temporal/support/contact/collision guards.
9. Preserve legacy status fields separately.

Deliverable:

```text
release_quality_v2_validation_matrix.json
```

### Phase 3 — Deep audit candidate rows

1. Deep-audit the six Step 4.3 candidate-warned rows.
2. Deep-audit any new candidate-warned rows.
3. Deep-audit any candidate-passed rows.
4. Prove stability or mark instability for each row.
5. Identify worst clips and margins.
6. Record whether promotion is safe.

Deliverable:

```text
candidate_warned_deep_audit.json
```

### Phase 4 — Build blocker taxonomy for the 26 blocked rows

1. Classify every blocked row.
2. Identify primary and secondary blockers.
3. Link blockers to metrics.
4. Identify whether blockers are global algorithmic, task-class-specific, solver-related, temporal, contact/collision, morphology, or anchor related.
5. Recommend next global fix for each class.

Deliverable:

```text
release_quality_v2_blocker_taxonomy.json
```

### Phase 5 — Stress test candidate gate

1. Deterministic rerun stress.
2. Clip leave-one-out stress.
3. Worst-clip stress.
4. Candidate-warned stability stress.
5. Candidate-passed stability stress if any candidate pass exists.
6. Normalization denominator stability stress.
7. Task-class weighting stress if used.
8. Solver retry or line-search stress if used.
9. Temporal/support/contact/collision guard stress.

Deliverable:

```text
release_quality_v2_stress_test.json
```

### Phase 6 — Try global coverage expansion only if safe

Allowed attempts:

1. Global clip aggregation policy with worst-clip guard.
2. Global task-class residual weighting.
3. Global solver retry or line-search refinement.
4. Global temporal-consistency residual penalty.
5. Global candidate scoring reconciliation preserving raw residual integrity.

For each attempt record:

```text
attempt_name
global_or_not
config_changed
thresholds_changed
legacy_gates_changed
production_defaults_changed
rows_improved
rows_regressed
raw_residual_regression_count
candidate_passed_delta
candidate_warned_delta
blocked_delta
why_accepted_or_rejected
```

Deliverables:

```text
quality_delta_vs_step4_3.json
pipeline_config.json
solver_config.json
gate_reconciliation_v3_report.json
```

### Phase 7 — Promotion readiness decision

1. Decide whether the release-quality-v2 candidate gate is promotable.
2. If promotable, explain why with evidence.
3. If not promotable, explain exact blockers.
4. Never promote if clip stability, worst-clip guard, normalization integrity, final-head CI, or LFS evidence is missing.

Deliverable:

```text
release_quality_v2_promotion_readiness.json
```

### Phase 8 — Audit, tests, workflow, handoff

1. Create audit script.
2. Create tests.
3. Create GitHub Actions workflow.
4. Run local tests.
5. Run pipeline.
6. Run audit.
7. Run LFS fsck.
8. Push branch.
9. Verify final-head CI evidence.
10. Create final handoff.

Deliverables:

```text
complete Step 4.4 branch artifacts
docs/retargeting_v3/STEP4_4_RELEASE_QUALITY_V2_VALIDATION_HANDOFF.md
```

---

## 11. Quality status rules

Set `PASS_RC` only if at least one of these is true and all hard invariants pass:

```text
release_quality_candidate_passed_count > 0
OR rows_below_candidate_warn_gate > 6
OR release_quality_candidate_blocked_count < 26
OR promotion_readiness recommends promote_to_release_candidate_gate with complete evidence
```

Set `BLOCKED_RELEASE_QUALITY_V2_VALIDATION` if:

```text
candidate coverage does not improve
AND diagnostics are complete
AND invariants are preserved
AND candidate gate is not proven unstable
AND clip generalization is not specifically disproven
```

Set `BLOCKED_CANDIDATE_GATE_ROBUSTNESS` if:

```text
candidate-warned rows are unstable across clips
OR candidate-passed rows are unstable across clips
OR candidate status changes under deterministic or stress rerun without an accepted reason
```

Set `BLOCKED_CLIP_GENERALIZATION` if:

```text
apparent improvement is limited to specific clips
OR hard clips are responsible for blocker behavior
OR global aggregation hides worst-clip failures
```

Set `BLOCKED_PIPELINE_REGRESSION` if any invariant regresses:

```text
row count
32/3/9 partition
solver-backed count
residual-only count
runtime hard failure count
determinism
trajectory exports
temporal/support/contact/collision diagnostics
legacy gate separation
normalization integrity
closed artifact immutability
```

Set `BLOCKED_CI_OR_PROVENANCE` if:

```text
final-head CI evidence missing
workflow_run_id missing
head_sha missing
job conclusions missing
LFS fsck missing or failing
source/artifact provenance missing
```

---

## 12. Definition of done

The task is done only when all of these are true:

```text
branch is retargeting-v3-step4-4-release-quality-v2-validation or clearly equivalent
new artifact tree exists
old artifact trees are untouched
all required Step 4.4 artifacts exist and are schema-valid
44 rows are preserved
32/3/9 partition is preserved
32 of 32 full humanoids remain solver-backed
residual_only_count = 0
runtime_quality_failed_count = 0
legacy gates are unchanged
candidate gate remains separate from legacy runtime quality
normalization integrity passes
no denominator inflation
no raw residual regression hiding
no robot-specific tuning
no whitelist or blacklist
no hard robots or clips removed
no production default mutation
no runtime override default enablement
deterministic rerun is 44/44
trajectory exports are finite and complete
temporal diagnostics are finite and complete
support/contact diagnostics are complete
collision proxy diagnostics are complete
candidate-warned deep audit exists
blocker taxonomy exists
stress test exists
promotion readiness exists
audit script passes locally in non-strict mode
strict audit passes once final-head CI evidence is available, or final status is BLOCKED_CI_OR_PROVENANCE
tests pass or failures are explicitly captured as blockers
Git LFS fsck evidence is recorded
workflow exists and includes all required jobs
final handoff exists
final handoff records counts, deltas, candidate rows, blockers, CI evidence, LFS evidence, known limitations, and next direction
Step 4.5 has not been started
```

---

## 13. Final handoff required structure

Create:

```text
docs/retargeting_v3/STEP4_4_RELEASE_QUALITY_V2_VALIDATION_HANDOFF.md
```

Use this structure:

```markdown
# Step 4.4 Release-Quality V2 Validation Handoff

## 1. Branch and provenance
- Branch:
- Final HEAD:
- Baseline Step 4.3 branch:
- Baseline Step 4.3 artifact dir:
- Step 4.4 artifact dir:
- Workflow run id:
- Workflow conclusion:
- Job conclusions:
- LFS fsck result:

## 2. Final verdict
- release_candidate_status:
- verdict:
- reason:
- promotion readiness decision:

## 3. Baseline Step 4.3 truth
- legacy runtime counts:
- candidate v2 counts:
- candidate-warned rows:
- preserved invariants:

## 4. Step 4.4 counts
- legacy runtime counts:
- candidate v2 counts:
- candidate-passed rows:
- candidate-warned rows:
- candidate-blocked rows:
- deltas vs Step 4.3:

## 5. Global changes attempted
- change name:
- global or not:
- accepted or rejected:
- why:
- metrics before/after:

## 6. Candidate stability
- candidate-warned stability:
- candidate-passed stability:
- worst-clip guard:
- clip generalization:

## 7. Blocker taxonomy
- primary blocker categories:
- affected rows:
- recommended next global fix:

## 8. Stress tests
- deterministic rerun:
- clip leave-one-out:
- worst-clip:
- normalization integrity:
- temporal/support/contact/collision:

## 9. Audit and tests
- strict audit result:
- focused tests result:
- test files added:
- workflow jobs:

## 10. Prohibitions verified
- legacy gates unchanged:
- no candidate-to-legacy leakage:
- no robot-specific tuning:
- no threshold lowering:
- no whitelist/blacklist:
- no hard row/clip removal:
- no denominator inflation:
- no production default mutation:
- no visual/deployment claim:

## 11. Known limitations
- limitation 1:
- limitation 2:

## 12. Next direction
- Recommended Step 4.5 or next blocked investigation:
- Do not start automatically.
```

---

## 14. Red-team checklist

Before finalizing, Codex must ask and encode the answers in artifacts/tests:

```text
Did I improve metrics by changing labels?
Did I accidentally count candidate warnings as legacy passes?
Did I lower any threshold?
Did I change production defaults?
Did I enable runtime override by default?
Did I remove hard robots?
Did I remove hard clips?
Did I introduce robot-specific weights?
Did I introduce per-clip whitelist or blacklist behavior?
Did I hide raw residual regression with normalization?
Did I reintroduce row-local denominator behavior?
Did I inflate the candidate denominator?
Did I improve aggregate score while worsening worst clip?
Did I ignore temporal instability?
Did I ignore support/contact/collision proxy outliers?
Did I overwrite old artifacts?
Did I forget deterministic rerun?
Did I forget LFS fsck?
Did I forget final-head CI evidence?
Did I overclaim visual or deployment readiness?
```

Any unsafe answer blocks the release-quality-v2 validation and must be recorded.

---

## 15. Final response Codex should produce

At the end, Codex should summarize:

```text
branch used
files changed
whether candidate coverage improved
whether any candidate rows passed
whether the original six candidate-warned rows were stable
what blocked rows are blocked by
whether all invariants held
whether audit/tests/CI/LFS passed
where artifacts and handoff are
next direction
```

Do not say all humanoids are solved unless artifacts prove it. Do not say deployment ready or visual ready in Step 4.4.

---

## 16. Compact Codex prompt

```text
You are working on Retargeting V3 in repository XinyuSong123/soma_retargeter_autoconfig.

After git pull, read GOAL.md fully and inspect the repo state. Current completed stage is Step 4.3 normalized residual gate reconciliation. Use branch retargeting-v3-step4-4-release-quality-v2-validation.

Your job is Step 4.4: Release-Quality V2 Validation / Candidate Pass Expansion. Validate candidate gate robustness, stability, clip generalization, blocker taxonomy, stress tests, and promotion readiness. Try to improve candidate coverage only with global, auditable methods. Preserve honest evidence.

Baseline Step 4.3 truth:
legacy runtime_quality_passed_count = 0
legacy runtime_quality_warned_count = 32
legacy runtime_quality_failed_count = 0
legacy high_residual_warning_count = 32
release_quality_candidate_passed_count = 0
release_quality_candidate_warned_count = 6
release_quality_candidate_blocked_count = 26
rows_below_candidate_warn_gate = 6
rows_below_candidate_pass_gate = 0
legacy_gates_unchanged = true
candidate_release_gates_active = true
production_default_changed = false
runtime_override_default_enabled = false

Primary target:
release_quality_candidate_passed_count > 0 OR rows_below_candidate_warn_gate > 6 OR release_quality_candidate_blocked_count < 26 OR honest BLOCKED_RELEASE_QUALITY_V2_VALIDATION with complete diagnostics.

Do not tune per robot. Do not weaken gates. Do not count candidate warnings as legacy passes. Do not overwrite closed artifacts. Do not hide raw residual regression. Do not remove hard robots or clips. Do not claim visual/deployment readiness.

Deliver Step 4.4 artifacts, audit, tests, workflow, deterministic rerun, LFS evidence, final-head CI evidence, and final handoff. Do not start Step 4.5.
```
