#!/usr/bin/env bash
set -Eeuo pipefail

# Explicit scope decision: keep all 46 sources resolved, vendor/use 44 entries,
# and defer exactly two snapshot-generation failures. This wrapper does not
# permit arbitrary partial success: it verifies the exact totals before push.

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
  echo "error: run inside the repository" >&2
  exit 2
fi

BASE_BRANCH="${BASE_BRANCH:-retargeting-v3-step2-assets-clean-sync}"
OUTPUT_BRANCH="${OUTPUT_BRANCH:-retargeting-v3-step2-assets-vendored}"
ROBOT_ZOO_CACHE="${ROBOT_ZOO_CACHE:-${REPO_ROOT}/../soma_robot_zoo_cache}"
WORKTREE_ROOT="${WORKTREE_ROOT:-${REPO_ROOT}/../.soma-retargeter-worktrees}"
WORKTREE="${WORKTREE_ROOT}/${OUTPUT_BRANCH}"

cd "${REPO_ROOT}"
git fetch origin --prune

if [[ "$(git branch --show-current)" != "${BASE_BRANCH}" ]]; then
  git checkout "${BASE_BRANCH}"
  git pull --ff-only
fi

# A previous failed strict run may have left this local branch behind.
if git show-ref --verify --quiet "refs/heads/${OUTPUT_BRANCH}"; then
  if git worktree list --porcelain | grep -Fq "branch refs/heads/${OUTPUT_BRANCH}"; then
    git worktree remove --force "${WORKTREE}" >/dev/null 2>&1 || true
  fi
  git branch -D "${OUTPUT_BRANCH}"
fi

ALLOW_PARTIAL_ASSETS=1 \
PUSH_ASSET_BRANCH=0 \
KEEP_WORKTREE=1 \
ROBOT_ZOO_CACHE="${ROBOT_ZOO_CACHE}" \
WORKTREE_ROOT="${WORKTREE_ROOT}" \
OUTPUT_BRANCH="${OUTPUT_BRANCH}" \
  bash scripts/fetch_and_vendor_robot_zoo_assets.sh

LOCK_FILE="${WORKTREE}/assets/robot_zoo/robot_zoo_lock.json"
if [[ ! -f "${LOCK_FILE}" ]]; then
  echo "error: lock file was not generated" >&2
  exit 3
fi

python - "${LOCK_FILE}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
entries = data.get("entries", {})
totals = data.get("totals", {})

deferred = sorted(
    robot_id
    for robot_id, row in entries.items()
    if row.get("snapshot_status") == "snapshot_failed"
)

expected = {
    "entries": 46,
    "source_available": 46,
    "vendored": 38,
    "fetch_only": 5,
    "local_existing": 1,
    "snapshot_failed": 2,
}
errors = [f"{key}: expected {value}, got {totals.get(key)!r}" for key, value in expected.items() if totals.get(key) != value]
if len(deferred) != 2:
    errors.append(f"expected exactly 2 deferred snapshots, got {deferred}")

blockers = data.get("blockers", [])
for blocker in blockers:
    if not any(str(blocker).startswith(robot_id + ":") for robot_id in deferred):
        errors.append(f"unexpected blocker outside deferred set: {blocker}")

if errors:
    raise SystemExit("\n".join(errors))

data["scope_decision"] = {
    "decision": "accept_exactly_two_deferred_snapshots",
    "deferred_snapshot_ids": deferred,
    "deferred_snapshot_count": 2,
    "in_scope_source_count": 44,
    "in_scope_breakdown": {
        "vendored_snapshots": 38,
        "fetch_only_cached": 5,
        "project_local_existing": 1,
    },
    "blocking_count_for_current_scope": 0,
    "note": "The two listed snapshot failures are explicitly outside future project scope; their source and failure reasons remain recorded for auditability.",
}
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
print("accepted deferred snapshots:", ", ".join(deferred))
PY

cd "${WORKTREE}"
git add assets/robot_zoo/robot_zoo_lock.json
git commit -m "assets: record two deferred Robot Zoo snapshots"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "error: asset worktree is dirty before push" >&2
  git status --short >&2
  exit 4
fi

git push -u origin "HEAD:${OUTPUT_BRANCH}"
ASSET_HEAD="$(git rev-parse HEAD)"

echo
echo "Asset scope prepared and pushed."
echo "  branch: ${OUTPUT_BRANCH}"
echo "  head:   ${ASSET_HEAD}"
echo "  scope:  44 sources in use, exactly 2 snapshots deferred"
echo
echo "Next:"
echo "  git fetch origin"
echo "  git checkout ${OUTPUT_BRANCH}"
