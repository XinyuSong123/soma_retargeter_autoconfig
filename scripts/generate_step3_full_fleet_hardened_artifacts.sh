#!/usr/bin/env bash
set -euo pipefail

artifact_root="${1:-${STEP3_1_ARTIFACT_ROOT:-artifacts/retargeting_v3_step3_runtime_quality}}"
if [[ $# -gt 0 ]]; then
  shift
fi

PYTHONPATH=. python -m soma_retargeter.tools.run_v3_full_fleet_runtime_quality \
  --artifact-root "${artifact_root}" \
  --step2-profile-root artifacts/retargeting_v3_step2_capability \
  --step3-shadow-root artifacts/retargeting_v3_step3_runtime_shadow \
  --lock assets/robot_zoo/robot_zoo_lock.json \
  --manifest assets/robot_zoo/robot_zoo_manifest.json \
  --clip-root assets/motions \
  --required-core-clips \
    assets/motions/bvh/Neutral_walk_forward_002__A057.bvh \
    assets/motions/bvh/wave_R_001__A428.bvh \
    assets/motions/bvh/body_stretch_1_004__A069.bvh \
    assets/motions/bvh/item_pick_up_standing_R_001__A410.bvh \
  --short-max-frames 120 \
  --mid-max-frames 300 \
  --deterministic-rerun \
  "$@"

PYTHONPATH=. python scripts/audit_retargeting_v3_step3_full_fleet.py \
  --artifact-dir "${artifact_root}" \
  --source-root .
