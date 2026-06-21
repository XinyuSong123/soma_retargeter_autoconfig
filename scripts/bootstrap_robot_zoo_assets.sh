#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

CACHE_ROOT="${ROBOT_DESCRIPTIONS_CACHE:-${HOME}/.cache/robot_descriptions}"
OUTPUT="${RETARGETING_V3_ASSET_SYNC_OUTPUT:-artifacts/retargeting_v3_assets/source_sync.json}"
EXTRA_ARGS=()
if [[ "${1:-}" == "--include-fetch-only" ]]; then
  EXTRA_ARGS+=(--include-fetch-only)
fi

python - <<'PY'
import importlib
import importlib.metadata
import sys

required = {
    "mujoco": "3.8.1",
    "robot_descriptions": "2.0.0",
    "pycollada": "0.9.3",
}
errors = []
for distribution, expected in required.items():
    try:
        actual = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        errors.append(f"missing {distribution}=={expected}")
        continue
    if actual != expected:
        errors.append(f"{distribution}: expected {expected}, got {actual}")

for module in ("mujoco", "robot_descriptions", "collada"):
    try:
        importlib.import_module(module)
    except Exception as exc:
        errors.append(f"cannot import {module}: {type(exc).__name__}: {exc}")

if errors:
    print("Robot Zoo environment is incomplete:", file=sys.stderr)
    for error in errors:
        print(f"- {error}", file=sys.stderr)
    print("Install with: python -m pip install -e '.[robot-zoo]'", file=sys.stderr)
    raise SystemExit(2)
PY

export ROBOT_DESCRIPTIONS_CACHE="${CACHE_ROOT}"
mkdir -p "${ROBOT_DESCRIPTIONS_CACHE}"

python -m soma_retargeter.tools.sync_robot_zoo_v3 \
  --manifest assets/robot_zoo/robot_zoo_manifest.json \
  --source-lock assets/robot_zoo/source_lock.json \
  --cache-root "${ROBOT_DESCRIPTIONS_CACHE}" \
  --output "${OUTPUT}" \
  "${EXTRA_ARGS[@]}"

python -m soma_retargeter.tools.sync_robot_zoo_v3 \
  --manifest assets/robot_zoo/robot_zoo_manifest.json \
  --source-lock assets/robot_zoo/source_lock.json \
  --cache-root "${ROBOT_DESCRIPTIONS_CACHE}" \
  --output "${OUTPUT%.json}.verify.json" \
  --verify-only \
  "${EXTRA_ARGS[@]}"

echo "Robot Zoo required-source bootstrap completed."
echo "Cache: ${ROBOT_DESCRIPTIONS_CACHE}"
echo "Inventory: ${OUTPUT}"
