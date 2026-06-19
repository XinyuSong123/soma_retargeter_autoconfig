# Retargeting v2 Implementation Report

## Current Commits

- `bb91825 Use mesh bounds for v2 virtual sites`
- `9496fa9 Validate compiled profile cache fingerprints`
- `bcbdfad Validate v2 left right chain symmetry`
- `75ab6a8 Strengthen compiled profile health validation`
- `7d52b6a Record root ground provenance in v2 profiles`
- `c48ef2e Add optional runtime collision barriers`
- `1c1ebc1 Add compile-time self collision metadata`
- `9de10ae Add v2 benchmark artifact reporting`
- `3e1b6c9 Add contact lock release state machine`
- `05a78a8 Add optional ground height barrier`
- `2201779 Guard priority zero joint limit residuals`
- `d53e5bb Apply v2 priority bands to runtime weights`
- `0fa2955 Add normalized temporal IK regularizers`
- `c9b437a Add v2 pole-vector IK tasks`
- `272fadd Add range-normalized joint limit barrier`
- `c363e3e Add v2 direction IK objectives`
- `96bbea3 Use segment-local scaler for v2 runtime profiles`
- `2156b71 Add segment-local v2 target builder`

## Implemented Design Items

- Dataclass profile model: `soma_retargeter/robotics/retarget_profile.py`
- MJCF morphology extraction: `soma_retargeter/robotics/morphology.py`
- Reachability projection helpers: `soma_retargeter/robotics/reachability.py`
- v2 task compiler: `soma_retargeter/robotics/task_compiler.py`
- Autoconfig CLI: `soma_retargeter/tools/autoconfigure_robot.py`
- Runtime registry integration: `soma_retargeter/robot_registry_parser.py`
- Segment-local target builder: `soma_retargeter/robotics/human_to_robot_scaler.py`
- MJCF explicit sites plus primitive and STL mesh bounds for distal virtual sites and collision proxies: `soma_retargeter/robotics/morphology.py` and `soma_retargeter/robotics/task_compiler.py`
- Direction, pole-vector, temporal, joint-limit, contact, and ground-barrier objectives: `soma_retargeter/pipelines/ik_objectives.py`
- v2 runtime wiring and priority guard: `soma_retargeter/pipelines/newton_pipeline.py`
- Compile-level benchmark artifact CLI: `soma_retargeter/tools/benchmark_retargeting.py`
- Compile-level self-collision proxy and high-risk pair metadata: `soma_retargeter/robotics/morphology.py` and `soma_retargeter/robotics/task_compiler.py`
- Root-motion scale and ground-height provenance metadata: `soma_retargeter/robotics/task_compiler.py`
- Compiled profile health-gate validation, including left/right chain length mismatch thresholds: `soma_retargeter/robot_registry_parser.py`
- Compiled profile cache fingerprint invalidation for robot morphology, source config, and compiler version: `soma_retargeter/robot_registry_parser.py`

## Tests

Latest full test command:

```bash
conda run -n soma-retargeter-v2 pytest -q
```

Latest result before this report update: `65 passed`. The benchmark CLI tests are in `tests/test_benchmark_retargeting.py`.

## Benchmark Summary

Generated command:

```bash
python -m soma_retargeter.tools.benchmark_retargeting --robots roboparty_rpo unitree_g1 --motions assets/motions/bvh --compare legacy v2 --output artifacts/retargeting_v2 --force
```

Current artifact scope is compile-level validation. Runtime motion metrics such as foot slide, penetration, RMSE, and velocity percentiles are present in the schema but marked `not_run`.

RoboParty RPO currently compiles with schema v2, `xyzw` quaternion order, confidence `1.0`, enabled sparse tasks, mesh-derived hand/foot virtual sites, mesh-derived self-collision metadata, morphology site-count metadata, and matching cache fingerprints.

Unitree G1 currently produces a v2 artifact, but its chain summary shows zero reachable ranks and minimal chain lengths; this is a known limitation of the current registered model/config path and needs deeper registry/morphology work before runtime acceptance.

## Remaining Work

- Implement true motion-level benchmark loops for legacy and v2 runtime.
- Add E3 v2 and OLI registry assets when available.
- Add runtime benchmark coverage for optional collision barriers on real motions.
- Strengthen semantic auto-detection beyond explicit maps.
- Add quantitative torso leakage gates to motion-level benchmarks.
- Produce before/after motion metrics for acceptance thresholds.
