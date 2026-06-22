#!/usr/bin/env bash
set -Eeuo pipefail

# Fetch all manifest-declared public robot descriptions into an external cache,
# build license-aware kinematic snapshots in a clean worktree, commit them, and
# optionally push the clean asset branch.
#
# Typical use:
#   bash scripts/fetch_and_vendor_robot_zoo_assets.sh
#
# Environment overrides:
#   BASE_BRANCH=retargeting-v3-step2-assets-clean-sync
#   OUTPUT_BRANCH=retargeting-v3-step2-assets-vendored
#   ROBOT_ZOO_CACHE=/path/outside/repository
#   INSTALL_MISSING_DEPS=1
#   ALLOW_PARTIAL_ASSETS=0
#   PUSH_ASSET_BRANCH=1
#   KEEP_WORKTREE=0

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
  echo "error: run this script inside the soma_retargeter_autoconfig repository" >&2
  exit 2
fi

BASE_BRANCH="${BASE_BRANCH:-retargeting-v3-step2-assets-clean-sync}"
OUTPUT_BRANCH="${OUTPUT_BRANCH:-retargeting-v3-step2-assets-vendored}"
ROBOT_ZOO_CACHE="${ROBOT_ZOO_CACHE:-${REPO_ROOT}/../soma_robot_zoo_cache}"
INSTALL_MISSING_DEPS="${INSTALL_MISSING_DEPS:-1}"
ALLOW_PARTIAL_ASSETS="${ALLOW_PARTIAL_ASSETS:-0}"
PUSH_ASSET_BRANCH="${PUSH_ASSET_BRANCH:-1}"
KEEP_WORKTREE="${KEEP_WORKTREE:-0}"
WORKTREE_ROOT="${WORKTREE_ROOT:-${REPO_ROOT}/../.soma-retargeter-worktrees}"
WORKTREE="${WORKTREE_ROOT}/${OUTPUT_BRANCH}"
MANIFEST="assets/robot_zoo/robot_zoo_manifest.json"
INVENTORY="assets/robot_zoo/source_inventory.json"
LOCK_FILE="assets/robot_zoo/robot_zoo_lock.json"
SNAPSHOT_ROOT="assets/robot_zoo/snapshots"
MENAGERIE_REF="feadf76d42f8a2162426f7d226a3b539556b3bf5"
MENAGERIE_URL="https://github.com/google-deepmind/mujoco_menagerie.git"

case "${OUTPUT_BRANCH}" in
  main|master|dev|"${BASE_BRANCH}")
    echo "error: refusing protected/source output branch '${OUTPUT_BRANCH}'" >&2
    exit 2
    ;;
esac

mkdir -p "${ROBOT_ZOO_CACHE}" "${WORKTREE_ROOT}"
ROBOT_ZOO_CACHE="$(cd "${ROBOT_ZOO_CACHE}" && pwd)"
WORKTREE_ROOT="$(cd "${WORKTREE_ROOT}" && pwd)"
WORKTREE="${WORKTREE_ROOT}/${OUTPUT_BRANCH}"

if [[ "${ROBOT_ZOO_CACHE}" == "${REPO_ROOT}"* ]]; then
  echo "error: ROBOT_ZOO_CACHE must be outside the Git checkout" >&2
  exit 2
fi

cleanup() {
  if [[ "${KEEP_WORKTREE}" != "1" && -d "${WORKTREE}" ]]; then
    git -C "${REPO_ROOT}" worktree remove --force "${WORKTREE}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

cd "${REPO_ROOT}"
echo "==> Fetching remote refs"
git fetch origin --prune

if ! git show-ref --verify --quiet "refs/remotes/origin/${BASE_BRANCH}"; then
  echo "error: origin/${BASE_BRANCH} does not exist" >&2
  exit 2
fi

if git show-ref --verify --quiet "refs/remotes/origin/${OUTPUT_BRANCH}"; then
  echo "error: origin/${OUTPUT_BRANCH} already exists; choose another OUTPUT_BRANCH or delete/review it first" >&2
  exit 2
fi

if [[ -e "${WORKTREE}" ]]; then
  git worktree remove --force "${WORKTREE}" >/dev/null 2>&1 || rm -rf "${WORKTREE}"
fi
if git show-ref --verify --quiet "refs/heads/${OUTPUT_BRANCH}"; then
  git branch -D "${OUTPUT_BRANCH}"
fi

echo "==> Creating clean worktree ${WORKTREE}"
git worktree add -b "${OUTPUT_BRANCH}" "${WORKTREE}" "origin/${BASE_BRANCH}"
cd "${WORKTREE}"
export PYTHONPATH="${WORKTREE}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "error: freshly-created worktree is unexpectedly dirty" >&2
  git status --short >&2
  exit 2
fi

if [[ "${INSTALL_MISSING_DEPS}" == "1" ]]; then
  echo "==> Installing pinned asset tooling"
  python -m pip install --disable-pip-version-check "robot_descriptions==2.0.0" "pycollada>=0.8" >/dev/null
fi

python - <<'PY'
import importlib.metadata as md
import mujoco

required = {
    "robot-descriptions": "2.0.0",
    "mujoco": "3.8.1",
}
for package, expected in required.items():
    actual = md.version(package)
    if actual != expected:
        raise SystemExit(f"{package} version mismatch: expected {expected}, got {actual}")
print("asset toolchain versions verified")
PY

export ROBOT_ZOO_CACHE
export ROBOT_DESCRIPTIONS_CACHE="${ROBOT_ZOO_CACHE}/robot_descriptions"
mkdir -p "${ROBOT_DESCRIPTIONS_CACHE}"

mapfile -t DESCRIPTIONS < <(
  python - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("assets/robot_zoo/robot_zoo_manifest.json").read_text())
for name in sorted({entry.get("description_name") for entry in payload["models"] if entry.get("description_name")}):
    print(name)
PY
)

echo "==> Pulling ${#DESCRIPTIONS[@]} robot_descriptions modules"
PULL_LOG="${ROBOT_ZOO_CACHE}/robot_descriptions_pull.log"
: > "${PULL_LOG}"
FAILED_PULLS=()
for description in "${DESCRIPTIONS[@]}"; do
  echo "    - ${description}"
  success=0
  for attempt in 1 2 3; do
    if python -m robot_descriptions pull "${description}" >>"${PULL_LOG}" 2>&1; then
      success=1
      break
    fi
    # Import is an official fallback and triggers the same pinned description fetch.
    if python - "${description}" >>"${PULL_LOG}" 2>&1 <<'PY'
import importlib
import sys
name = sys.argv[1]
module = importlib.import_module(f"robot_descriptions.{name}")
paths = [getattr(module, attr, None) for attr in ("URDF_PATH", "MJCF_PATH", "XACRO_PATH", "PACKAGE_PATH")]
if not any(paths):
    raise SystemExit(f"{name} exposes no model/package path")
PY
    then
      success=1
      break
    fi
    sleep "$((attempt * 2))"
  done
  if [[ "${success}" != "1" ]]; then
    FAILED_PULLS+=("${description}")
  fi
done

MENAGERIE_DIR="${ROBOT_ZOO_CACHE}/mujoco_menagerie"
echo "==> Fetching pinned MuJoCo Menagerie ${MENAGERIE_REF}"
if [[ ! -d "${MENAGERIE_DIR}/.git" ]]; then
  rm -rf "${MENAGERIE_DIR}"
  git init -q "${MENAGERIE_DIR}"
  git -C "${MENAGERIE_DIR}" remote add origin "${MENAGERIE_URL}"
fi
git -C "${MENAGERIE_DIR}" fetch --depth=1 origin "${MENAGERIE_REF}"
git -C "${MENAGERIE_DIR}" checkout --detach --force FETCH_HEAD
test "$(git -C "${MENAGERIE_DIR}" rev-parse HEAD)" = "${MENAGERIE_REF}"

if (( ${#FAILED_PULLS[@]} > 0 )); then
  printf 'warning: failed robot_descriptions pulls:\n' >&2
  printf '  %s\n' "${FAILED_PULLS[@]}" >&2
  if [[ "${ALLOW_PARTIAL_ASSETS}" != "1" ]]; then
    echo "error: rerun after fixing network/source failures, or set ALLOW_PARTIAL_ASSETS=1 for a diagnostic branch" >&2
    exit 3
  fi
fi

echo "==> Resolving all manifest entries into a sanitized inventory"
python -m soma_retargeter.tools.sync_robot_zoo_v3 \
  --manifest "${MANIFEST}" \
  --cache-root "${ROBOT_ZOO_CACHE}" \
  --allow-fetch \
  --output "${INVENTORY}" \
  >/tmp/soma_robot_zoo_inventory_stdout.json

echo "==> Building mesh-free, license-aware kinematic snapshots"
SNAPSHOT_ARGS=(
  --manifest "${MANIFEST}"
  --cache-root "${ROBOT_ZOO_CACHE}"
  --output-root "${SNAPSHOT_ROOT}"
  --lock-output "${LOCK_FILE}"
  --clean
)
if [[ "${ALLOW_PARTIAL_ASSETS}" == "1" ]]; then
  SNAPSHOT_ARGS+=(--allow-partial)
fi
python scripts/build_robot_zoo_snapshots.py "${SNAPSHOT_ARGS[@]}"

echo "==> Validating generated repository content"
python - <<'PY'
import json
from pathlib import Path
import re

root = Path("assets/robot_zoo")
lock = json.loads((root / "robot_zoo_lock.json").read_text())
manifest = json.loads((root / "robot_zoo_manifest.json").read_text())
if len(lock["entries"]) != len(manifest["models"]):
    raise SystemExit("lock entry count does not match manifest")
if not lock.get("cache_is_external"):
    raise SystemExit("asset cache was not external to repository")
for path in (root / "snapshots").rglob("*"):
    if not path.is_file():
        continue
    if path.stat().st_size > 2_000_000:
        raise SystemExit(f"oversized snapshot file: {path}")
    if path.suffix.lower() in {".stl", ".dae", ".obj", ".ply", ".glb", ".gltf", ".fbx", ".3ds"}:
        raise SystemExit(f"mesh committed by mistake: {path}")
    text = path.read_text(errors="ignore")
    if re.search(r"/(?:home|mnt|Users|private/var)/", text):
        raise SystemExit(f"absolute local path leaked: {path}")
    if "cxxx_190" in text.lower():
        raise SystemExit(f"private-asset denylist match: {path}")
print(json.dumps(lock["totals"], sort_keys=True))
PY

python -m pytest -q \
  tests/v3/test_robot_zoo_source_resolution_local_cache.py \
  tests/v3/test_full_robot_zoo_validation.py

# Only generated, policy-approved files are staged. The external caches and pull
# logs can never enter Git.
git add \
  "${INVENTORY}" \
  "${LOCK_FILE}" \
  "${SNAPSHOT_ROOT}"

if git diff --cached --quiet; then
  echo "error: no generated asset changes were staged" >&2
  exit 4
fi

if git diff --cached --name-only | grep -E '\.(stl|dae|obj|ply|glb|gltf|fbx|3ds)$' >/dev/null; then
  echo "error: a visual mesh is staged" >&2
  exit 4
fi

if git diff --cached --numstat | awk '$1 != "-" && $1 + $2 > 100000 { exit 1 }'; then
  :
else
  echo "error: suspiciously large generated text diff" >&2
  exit 4
fi

git commit -m "assets: vendor pinned Robot Zoo kinematic snapshots"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "error: worktree is dirty after asset commit" >&2
  git status --short >&2
  exit 5
fi

ASSET_HEAD="$(git rev-parse HEAD)"
echo "==> Clean asset commit: ${ASSET_HEAD}"

if [[ "${PUSH_ASSET_BRANCH}" == "1" ]]; then
  git push -u origin "HEAD:${OUTPUT_BRANCH}"
  echo "==> Pushed origin/${OUTPUT_BRANCH}"
else
  echo "==> PUSH_ASSET_BRANCH=0; branch was not pushed"
fi

cat <<EOF

Completed clean Robot Zoo asset preparation.

Base branch:   ${BASE_BRANCH}
Asset branch:  ${OUTPUT_BRANCH}
Asset commit:  ${ASSET_HEAD}
External cache:${ROBOT_ZOO_CACHE}
Pull log:      ${PULL_LOG}

Next:
  git fetch origin
  git checkout ${OUTPUT_BRANCH}
EOF
