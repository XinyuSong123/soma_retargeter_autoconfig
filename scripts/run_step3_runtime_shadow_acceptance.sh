#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/retargeting_v3_step3_runtime_shadow}"
PROFILE_ARTIFACT_ROOT="${PROFILE_ARTIFACT_ROOT:-artifacts/retargeting_v3_step2_capability}"
MAX_FRAMES="${MAX_FRAMES:-120}"
ROBOTS=(${ROBOTS:-roboparty_rpo unitree_g1})
CLIPS=(${CLIPS:-assets/motions/bvh/Neutral_walk_forward_002__A057.bvh assets/motions/bvh/wave_R_001__A428.bvh})
MODES=(${MODES:-disabled shadow override_experimental})

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate soma-retargeter-v2
fi

mkdir -p "$ARTIFACT_ROOT/test_results"

SMOKE_STDOUT="$ARTIFACT_ROOT/test_results/runtime_shadow_smoke.stdout.txt"
SMOKE_STDERR="$ARTIFACT_ROOT/test_results/runtime_shadow_smoke.stderr.txt"
TMP_SMOKE_STDOUT="$(mktemp "${TMPDIR:-/tmp}/step3_runtime_shadow_stdout.XXXXXX")"
TMP_SMOKE_STDERR="$(mktemp "${TMPDIR:-/tmp}/step3_runtime_shadow_stderr.XXXXXX")"
cleanup() {
  rm -f "$TMP_SMOKE_STDOUT" "$TMP_SMOKE_STDERR"
}
trap cleanup EXIT

set +e
PYTHONPATH=. python -m soma_retargeter.tools.run_v3_runtime_shadow_smoke \
  --artifact-root "$ARTIFACT_ROOT" \
  --profile-artifact-root "$PROFILE_ARTIFACT_ROOT" \
  --robots "${ROBOTS[@]}" \
  --clips "${CLIPS[@]}" \
  --modes "${MODES[@]}" \
  --max-frames "$MAX_FRAMES" \
  --fail-on-blocked \
  > "$TMP_SMOKE_STDOUT" \
  2> "$TMP_SMOKE_STDERR"
SMOKE_RC=$?
set -e
mkdir -p "$ARTIFACT_ROOT/test_results"
cp "$TMP_SMOKE_STDOUT" "$SMOKE_STDOUT"
cp "$TMP_SMOKE_STDERR" "$SMOKE_STDERR"

set +e
PYTHONPATH=. python -m pytest -q \
  tests/v3/test_step3_runtime_artifact_schema_*.py \
  tests/v3/test_step3_runtime_smoke_matrix_*.py \
  --junitxml "$ARTIFACT_ROOT/test_results/junit.xml" \
  > "$ARTIFACT_ROOT/test_results/pytest.txt" \
  2>&1
PYTEST_RC=$?
set -e

PYTHONPATH=. python - "$PYTEST_RC" "$ARTIFACT_ROOT/test_results/pytest.txt" "$ARTIFACT_ROOT/test_results/pytest_summary.json" <<'PY'
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

if [[ "$PYTEST_RC" -ne 0 || "$SMOKE_RC" -ne 0 ]]; then
  echo "Step 3 runtime shadow acceptance failed: smoke_rc=$SMOKE_RC pytest_rc=$PYTEST_RC" >&2
  exit 1
fi

echo "Step 3 runtime shadow acceptance passed"
