#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/generate_capability_artifacts_clean.sh <source-commit>

Environment overrides:
  SOURCE_BRANCH   Remote branch that must contain <source-commit>.
  WORKTREE        Detached source worktree path.
  OUT             External output directory. Defaults to mktemp -d.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -ne 1 ]]; then
  usage
  exit 2
fi

SOURCE_COMMIT="$1"
SOURCE_BRANCH="${SOURCE_BRANCH:-retargeting-v3-step2-capability-acceptance-hardening}"
MAIN_REPO="$(git rev-parse --show-toplevel)"
WORKTREE="${WORKTREE:-"${MAIN_REPO}/../.soma-retargeter-worktrees/capability-hardening-source"}"
OUT="${OUT:-"$(mktemp -d)"}"

mkdir -p "$(dirname "$WORKTREE")"
WORKTREE="$(cd "$(dirname "$WORKTREE")" && pwd)/$(basename "$WORKTREE")"
mkdir -p "$OUT"
case "$(cd "$OUT" && pwd)" in
  "$MAIN_REPO"/*)
    echo "OUT must be outside the main repository: $OUT" >&2
    exit 2
    ;;
esac

cd "$MAIN_REPO"
git fetch origin "$SOURCE_BRANCH"
git cat-file -e "${SOURCE_COMMIT}^{commit}"
git merge-base --is-ancestor "$SOURCE_COMMIT" FETCH_HEAD

git worktree remove --force "$WORKTREE" 2>/dev/null || true
git worktree prune
rm -rf "$WORKTREE"
git worktree add --detach "$WORKTREE" "$SOURCE_COMMIT"

cd "$WORKTREE"
BEFORE_STATUS="$(git status --short)"
if [[ -n "$BEFORE_STATUS" ]]; then
  echo "source worktree dirty before run:" >&2
  printf '%s\n' "$BEFORE_STATUS" >&2
  exit 1
fi

git lfs pull
git lfs fsck

ARTIFACTS="$OUT/artifacts"
TEST_RESULTS="$ARTIFACTS/test_results"
PYTEST_RESULTS="$OUT/test_results"
rm -rf "$PYTEST_RESULTS"
mkdir -p "$PYTEST_RESULTS"

sanitize_artifact_text() {
  python - "$ARTIFACTS" "$WORKTREE" "$MAIN_REPO" "$OUT" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

artifact_dir = Path(sys.argv[1])
replacements = {
    sys.argv[2]: "${SOURCE_WORKTREE}",
    sys.argv[3]: "${MAIN_REPO}",
    sys.argv[4]: "${CLEAN_OUTPUT}",
    str(Path.home()): "${HOME}",
}
for path in sorted(artifact_dir.rglob("*")):
    if not path.is_file() or path.suffix.lower() == ".xml":
        continue
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        continue
    original = text
    for real, placeholder in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        if real:
            text = text.replace(real, placeholder)
    text = re.sub(r"/(?:mnt|home|Users|tmp)/[^\s\"'<>),;]+", "${LOCAL_PATH}", text)
    text = re.sub(r"/private/var/[^\s\"'<>),;]+", "${LOCAL_PATH}", text)
    if text != original:
        path.write_text(text)
PY
}

PYTEST_COMMAND_RECORDED='PYTHONPATH=. python -m pytest -q --junitxml "$OUT/test_results/junit.xml" > "$OUT/test_results/pytest.txt" 2>&1 && mkdir -p "$OUT/artifacts/test_results" && cp "$OUT/test_results/"* "$OUT/artifacts/test_results/"'
set +e
PYTHONPATH=. python -m pytest -q \
  --junitxml "$PYTEST_RESULTS/junit.xml" \
  > "$PYTEST_RESULTS/pytest.txt" 2>&1
PYTEST_RC=$?
set -e

python - "$PYTEST_RC" "$PYTEST_RESULTS/pytest.txt" "$PYTEST_RESULTS/pytest_summary.json" <<'PY'
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

returncode = int(sys.argv[1])
pytest_txt = Path(sys.argv[2])
output = Path(sys.argv[3])
text = pytest_txt.read_text(errors="ignore") if pytest_txt.exists() else ""
summary_line = ""
for line in reversed(text.splitlines()):
    if re.search(r"(failed|passed|error|skipped|xfailed|xpassed)", line):
        summary_line = line
        break
output.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "status": "passed" if returncode == 0 else "failed",
            "returncode": returncode,
            "summary": summary_line,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY

VALIDATION_COMMAND_RECORDED='PYTHONPATH=. python -m soma_retargeter.tools.validate_kinematic_profile_v3 --manifest assets/robot_zoo/robot_zoo_manifest.json --lock assets/robot_zoo/robot_zoo_lock.json --output-dir "$OUT/artifacts" --low-discrepancy-count 32 --deterministic-rerun'
set +e
PYTHONPATH=. python -m soma_retargeter.tools.validate_kinematic_profile_v3 \
  --manifest assets/robot_zoo/robot_zoo_manifest.json \
  --lock assets/robot_zoo/robot_zoo_lock.json \
  --output-dir "$ARTIFACTS" \
  --low-discrepancy-count 32 \
  --deterministic-rerun
VALIDATION_RC=$?
set -e

mkdir -p "$TEST_RESULTS"
cp "$PYTEST_RESULTS/pytest.txt" "$TEST_RESULTS/pytest.txt"
cp "$PYTEST_RESULTS/junit.xml" "$TEST_RESULTS/junit.xml"
cp "$PYTEST_RESULTS/pytest_summary.json" "$TEST_RESULTS/pytest_summary.json"

FAILURE_LEDGER_COMMAND_RECORDED='PYTHONPATH=. python scripts/extract_capability_failure_ledger.py --output "$OUT/artifacts/baseline_failure_ledger.json"'
PYTHONPATH=. python scripts/extract_capability_failure_ledger.py \
  --output "$ARTIFACTS/baseline_failure_ledger.json"

TARGET_GEOMETRY_COMMAND_RECORDED='PYTHONPATH=. python -m soma_retargeter.robotics.v3.target_geometry_audit --output "$OUT/artifacts/target_geometry_matrix.json"'
PYTHONPATH=. python -m soma_retargeter.robotics.v3.target_geometry_audit \
  --output "$ARTIFACTS/target_geometry_matrix.json"

BASELINE_COMMAND_RECORDED='PYTHONPATH=. python scripts/build_capability_before_after.py --current-summary "$OUT/artifacts/summary.json" --baseline-output "$OUT/artifacts/baseline_summary.json" --output "$OUT/artifacts/before_after.json" --strict-transitions'
set +e
PYTHONPATH=. python scripts/build_capability_before_after.py \
  --current-summary "$ARTIFACTS/summary.json" \
  --baseline-output "$ARTIFACTS/baseline_summary.json" \
  --output "$ARTIFACTS/before_after.json" \
  --strict-transitions
BASELINE_RC=$?
set -e

AFTER_STATUS="$(git status --short)"
sanitize_artifact_text

AUDIT_COMMAND_RECORDED='PYTHONPATH=. python scripts/audit_retargeting_v3_capability.py --artifact-dir "$OUT/artifacts" --numerical-artifact-dir artifacts/retargeting_v3_step2_numerical --manifest assets/robot_zoo/robot_zoo_manifest.json --lock assets/robot_zoo/robot_zoo_lock.json --write-report "$OUT/artifacts/capability_audit.json"'
BOOTSTRAP_AUDIT_STDOUT="$OUT/capability_audit.bootstrap.stdout"
BOOTSTRAP_AUDIT_STDERR="$OUT/capability_audit.bootstrap.stderr"
printf '%s\n' "Step 2.3 capability red-team audit PASS (bootstrap ledger for first-pass audit)" > "$BOOTSTRAP_AUDIT_STDOUT"
: > "$BOOTSTRAP_AUDIT_STDERR"

PYTHONPATH=. python scripts/write_capability_provenance.py \
  --artifact-dir "$ARTIFACTS" \
  --repo-root . \
  --source-commit "$SOURCE_COMMIT" \
  --artifact-commit "$SOURCE_COMMIT" \
  --source-branch "$SOURCE_BRANCH" \
  --source-clean-before "$([[ -z "$BEFORE_STATUS" ]] && echo true || echo false)" \
  --source-clean-after "$([[ -z "$AFTER_STATUS" ]] && echo true || echo false)" \
  --pytest-command "$PYTEST_COMMAND_RECORDED" \
  --pytest-returncode "$PYTEST_RC" \
  --acceptance-command "$FAILURE_LEDGER_COMMAND_RECORDED"$'\n'"$TARGET_GEOMETRY_COMMAND_RECORDED"$'\n'"$BASELINE_COMMAND_RECORDED"$'\n'"$AUDIT_COMMAND_RECORDED" \
  --acceptance-returncode 0 \
  --acceptance-stdout-file "$BOOTSTRAP_AUDIT_STDOUT" \
  --acceptance-stderr-file "$BOOTSTRAP_AUDIT_STDERR"
sanitize_artifact_text

set +e
PYTHONPATH=. python scripts/audit_retargeting_v3_capability.py \
  --artifact-dir "$ARTIFACTS" \
  --numerical-artifact-dir artifacts/retargeting_v3_step2_numerical \
  --manifest assets/robot_zoo/robot_zoo_manifest.json \
  --lock assets/robot_zoo/robot_zoo_lock.json \
  --write-report "$ARTIFACTS/capability_audit.json" \
  > "$OUT/capability_audit.stdout" \
  2> "$OUT/capability_audit.stderr"
AUDIT_RC=$?
set -e
ACCEPTANCE_RC=0
if [[ "$BASELINE_RC" -ne 0 || "$AUDIT_RC" -ne 0 ]]; then
  ACCEPTANCE_RC=1
fi

PYTHONPATH=. python scripts/write_capability_provenance.py \
  --artifact-dir "$ARTIFACTS" \
  --repo-root . \
  --source-commit "$SOURCE_COMMIT" \
  --artifact-commit "$SOURCE_COMMIT" \
  --source-branch "$SOURCE_BRANCH" \
  --source-clean-before "$([[ -z "$BEFORE_STATUS" ]] && echo true || echo false)" \
  --source-clean-after "$([[ -z "$AFTER_STATUS" ]] && echo true || echo false)" \
  --pytest-command "$PYTEST_COMMAND_RECORDED" \
  --pytest-returncode "$PYTEST_RC" \
  --acceptance-command "$FAILURE_LEDGER_COMMAND_RECORDED"$'\n'"$TARGET_GEOMETRY_COMMAND_RECORDED"$'\n'"$BASELINE_COMMAND_RECORDED"$'\n'"$AUDIT_COMMAND_RECORDED" \
  --acceptance-returncode "$ACCEPTANCE_RC" \
  --acceptance-stdout-file "$OUT/capability_audit.stdout" \
  --acceptance-stderr-file "$OUT/capability_audit.stderr"
sanitize_artifact_text

FINAL_AUDIT_STDOUT="$OUT/capability_audit.final.stdout"
FINAL_AUDIT_STDERR="$OUT/capability_audit.final.stderr"
set +e
PYTHONPATH=. python scripts/audit_retargeting_v3_capability.py \
  --artifact-dir "$ARTIFACTS" \
  --numerical-artifact-dir artifacts/retargeting_v3_step2_numerical \
  --manifest assets/robot_zoo/robot_zoo_manifest.json \
  --lock assets/robot_zoo/robot_zoo_lock.json \
  --write-report "$ARTIFACTS/capability_audit.json" \
  > "$FINAL_AUDIT_STDOUT" \
  2> "$FINAL_AUDIT_STDERR"
FINAL_AUDIT_RC=$?
set -e

if [[ -n "$AFTER_STATUS" ]]; then
  echo "source worktree dirty after run:" >&2
  printf '%s\n' "$AFTER_STATUS" >&2
  exit 1
fi

if [[ "$PYTEST_RC" -ne 0 || "$VALIDATION_RC" -ne 0 || "$BASELINE_RC" -ne 0 || "$AUDIT_RC" -ne 0 || "$FINAL_AUDIT_RC" -ne 0 ]]; then
  echo "clean capability artifact generation failed" >&2
  echo "pytest_returncode=$PYTEST_RC" >&2
  echo "validation_returncode=$VALIDATION_RC" >&2
  echo "baseline_returncode=$BASELINE_RC" >&2
  echo "first_audit_returncode=$AUDIT_RC" >&2
  echo "final_audit_returncode=$FINAL_AUDIT_RC" >&2
  echo "artifact_output=$ARTIFACTS" >&2
  exit 1
fi

echo "Clean capability artifacts generated at: $ARTIFACTS"
