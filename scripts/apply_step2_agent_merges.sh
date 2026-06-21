#!/usr/bin/env bash
set -Eeuo pipefail

# Rebuild a local Step-2 integration branch from the six pushed agent branches.
# This script never touches main/dev and never pushes automatically.
#
# Usage:
#   ./scripts/apply_step2_agent_merges.sh
#   ./scripts/apply_step2_agent_merges.sh my-local-integration-branch
#
# Optional environment variables:
#   RUN_TESTS=0     Skip pytest after integration (default: 1)
#   FORCE_RESET=1   Recreate an existing local integration branch (default: 0)

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
  echo "error: run this script inside the soma_retargeter_autoconfig repository" >&2
  exit 2
fi
cd "${REPO_ROOT}"

BASE_BRANCH="retargeting-v3-step2-kinematic-core"
INTEGRATION_BRANCH="${1:-retargeting-v3-step2-integrated-local}"
RUN_TESTS="${RUN_TESTS:-1}"
FORCE_RESET="${FORCE_RESET:-0}"

# Pinned commits from the six pushed agent branches. Pinning prevents a later
# force-push or unrelated follow-up commit from silently changing integration.
A_CODE="ca4c19ca98477e94e176a7a5f038ebdfdd2ed2fa"
A_DOCS="dc560a6928c628ea8da9e67ac6646f5eee721488"
B_CODE="b449eb2a539e7076523898413777ef69676e7605"
B_DOCS="c2551e24d8b1da9ff7238c452007aed86907bcb7"
C_CODE="b2db0655432506e21541a235ed03cf831bcf6d09"
C_DOCS="107ae72ee023dc5b4a65e96c3ec40ee20990a697"
D_CODE="8dee96e3b79e32b664ee157891de45657c68f028"
E_CODE="b25b5f1912811cf0123dca3456c85f5a3e74394a"
E_DOCS="4ff8c418958727373b7b6ce8e853833748221fef"
F_CODE="67f726b0161eea748285402f222bc279b55ef1b4"
F_DOCS="84f5439c44485ff92e5dc705ac26cae8acd769f8"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "error: worktree is not clean; commit or stash local changes first" >&2
  git status --short >&2
  exit 2
fi

case "${INTEGRATION_BRANCH}" in
  main|master|dev|"${BASE_BRANCH}")
    echo "error: refusing to use protected/source branch '${INTEGRATION_BRANCH}'" >&2
    exit 2
    ;;
esac

echo "==> Fetching all remote agent refs"
git fetch origin --prune

if ! git show-ref --verify --quiet "refs/remotes/origin/${BASE_BRANCH}"; then
  echo "error: origin/${BASE_BRANCH} does not exist" >&2
  exit 2
fi

for ref in \
  origin/agent-a-runtime-model-truth \
  origin/agent-b-verified-semantics \
  origin/agent-c-rest-calibration \
  origin/agent-d-reachability \
  origin/agent-e-full-robot-zoo-validation \
  origin/agent-f-repro-ci-red-team; do
  if ! git show-ref --verify --quiet "refs/remotes/${ref}"; then
    echo "error: required remote branch '${ref}' is missing" >&2
    exit 2
  fi
done

for sha in \
  "${A_CODE}" "${A_DOCS}" "${B_CODE}" "${B_DOCS}" \
  "${C_CODE}" "${C_DOCS}" "${D_CODE}" \
  "${E_CODE}" "${E_DOCS}" "${F_CODE}" "${F_DOCS}"; do
  if ! git cat-file -e "${sha}^{commit}" 2>/dev/null; then
    echo "error: pinned commit ${sha} is unavailable after fetch" >&2
    exit 2
  fi
done

if git show-ref --verify --quiet "refs/heads/${INTEGRATION_BRANCH}"; then
  if [[ "${FORCE_RESET}" != "1" ]]; then
    echo "error: local branch '${INTEGRATION_BRANCH}' already exists" >&2
    echo "       rerun with FORCE_RESET=1 to recreate it" >&2
    exit 2
  fi
  current="$(git branch --show-current)"
  if [[ "${current}" == "${INTEGRATION_BRANCH}" ]]; then
    git switch --detach >/dev/null
  fi
  git branch -D "${INTEGRATION_BRANCH}"
fi

if git show-ref --verify --quiet "refs/remotes/origin/${INTEGRATION_BRANCH}"; then
  echo "warning: origin/${INTEGRATION_BRANCH} already exists; this script creates a fresh local branch from the Step-2 base" >&2
fi

echo "==> Creating ${INTEGRATION_BRANCH} from origin/${BASE_BRANCH}"
git switch --create "${INTEGRATION_BRANCH}" "origin/${BASE_BRANCH}"

cherry_pick_one() {
  local label="$1"
  local sha="$2"
  echo "==> ${label}: cherry-pick ${sha}"
  if ! git cherry-pick "${sha}"; then
    cat >&2 <<EOF

Cherry-pick conflict while integrating ${label} (${sha}).
Resolve conflicts, then run:
  git add <resolved-files>
  git cherry-pick --continue
After that, rerun this script only after deleting/recreating the integration
branch, or continue the remaining pinned commits manually using this file.
EOF
    exit 3
  fi
}

# Dependency order from goal.md: A -> B -> C -> D -> E -> F.
cherry_pick_one "Agent A runtime model truth" "${A_CODE}"
cherry_pick_one "Agent A handoff" "${A_DOCS}"
cherry_pick_one "Agent B verified semantics/sites" "${B_CODE}"
cherry_pick_one "Agent B handoff" "${B_DOCS}"
cherry_pick_one "Agent C rest calibration/targets" "${C_CODE}"
cherry_pick_one "Agent C handoff" "${C_DOCS}"
cherry_pick_one "Agent D canonical projection" "${D_CODE}"

# Agent E's later artifact commit contains a no-fetch scaffold with 45/46
# sources unavailable and dirty-worktree metadata. Integrate code + handoff,
# deliberately skip commit 9549b9e.
cherry_pick_one "Agent E manifest-driven validation code" "${E_CODE}"
cherry_pick_one "Agent E handoff provenance update" "${E_DOCS}"

# Agent F contributes the audit/CI implementation. Its checked-in audit output
# describes the pre-integration prototype, so remove those stale generated
# results after importing the audit code.
cherry_pick_one "Agent F acceptance audit" "${F_CODE}"
cherry_pick_one "Agent F handoff update" "${F_DOCS}"

STALE_RESULTS=(
  artifacts/retargeting_v3_step2/test_results/acceptance_audit.json
  artifacts/retargeting_v3_step2/test_results/acceptance_audit.junit.xml
  artifacts/retargeting_v3_step2/test_results/pytest_acceptance_gates.junit.xml
)
for path in "${STALE_RESULTS[@]}"; do
  if [[ -e "${path}" ]]; then
    git rm -f "${path}"
  fi
done

# Agent D did not push its required handoff. Add an explicit reconstruction so
# the missing evidence is visible rather than silently treated as complete.
mkdir -p docs/retargeting_v3/subagents
cat > docs/retargeting_v3/subagents/step2_agent_d_handoff.md <<'EOF'
# Step 2 Agent D Handoff — Integration Reconstruction

Status: **incomplete evidence / integration-generated record**

The `agent-d-reachability` branch did not contain the handoff required by
`goal.md`. This record is reconstructed from commit
`8dee96e3b79e32b664ee157891de45657c68f028` and is not Agent D self-attestation.

Implemented files:

- `soma_retargeter/robotics/v3/canonical_projection.py`
- `soma_retargeter/robotics/v3/chain_projection.py`
- `soma_retargeter/robotics/v3/numerical_jacobian.py`
- `tests/v3/test_chain_projection_agent_d.py`
- `tests/v3/test_numerical_jacobian_agent_d.py`
- `tests/v3/test_reachability_agent_d.py`

The commit adds canonical-motion projection helpers, range-normalized neutral
and continuity priors, rank-zero unreachable status, solver diagnostics, and
focused tests. Missing evidence: no Agent D command log, aggregate test result,
or complete-model projection artifact was pushed.
EOF

# Make CI run on this integration branch if/when it is pushed.
python - <<PY
from pathlib import Path
p = Path('.github/workflows/retargeting_v3_step2.yml')
if p.exists():
    text = p.read_text()
    if '      - ${INTEGRATION_BRANCH}\n' not in text:
        marker = '      - retargeting-v3-step2-kinematic-core\n'
        if marker in text:
            text = text.replace(marker, marker + '      - ${INTEGRATION_BRANCH}\n', 1)
    p.write_text(text)
PY

git add docs/retargeting_v3/subagents/step2_agent_d_handoff.md .github/workflows/retargeting_v3_step2.yml
if ! git diff --cached --quiet; then
  git commit -m "chore: finalize Step 2 agent integration scaffold"
fi

echo
echo "==> Integration branch created"
echo "    branch: ${INTEGRATION_BRANCH}"
echo "    head:   $(git rev-parse HEAD)"
echo

echo "==> Static sanity checks"
python -m compileall -q soma_retargeter/robotics/v3 soma_retargeter/tools scripts || {
  echo "compileall failed" >&2
  exit 4
}

TEST_STATUS=0
if [[ "${RUN_TESTS}" == "1" ]]; then
  echo "==> Running combined tests (failures are expected to expose remaining integration work)"
  set +e
  PYTHONPATH=. python -m pytest -q \
    tests/v3 \
    tests/test_kinematic_profile_v3_adapter.py \
    tests/test_kinematic_profile_v3_math.py \
    tests/test_kinematic_profile_v3_calibration_targets.py
  TEST_STATUS=$?
  set -e
  if [[ ${TEST_STATUS} -ne 0 ]]; then
    echo "warning: combined tests failed with exit code ${TEST_STATUS}" >&2
  fi
else
  echo "==> RUN_TESTS=0; tests skipped"
fi

echo
echo "Next commands:"
echo "  git status --short"
echo "  git log --oneline --decorate --graph -20"
echo "  python scripts/audit_retargeting_v3_step2.py --artifact-dir artifacts/retargeting_v3_step2 --source-root . --output-json /tmp/step2_audit.json --junit-xml /tmp/step2_audit.xml"
echo "  git push -u origin ${INTEGRATION_BRANCH}"

echo
if [[ ${TEST_STATUS} -eq 0 ]]; then
  echo "Merge completed and selected tests passed. This is not yet a Step-2 PASS until full Robot Zoo artifacts and Agent F red-team pass."
else
  echo "Merge completed, but selected tests exposed remaining failures. Do not push as completed or start Step 3 yet."
fi

exit ${TEST_STATUS}
