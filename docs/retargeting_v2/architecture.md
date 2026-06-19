# Retargeting v2 Architecture

Retargeting v2 adds a morphology-aware compile step before Newton IK. The compile step reads robot registry metadata, MJCF morphology, and a minimal semantic `ik_map`, then writes a deterministic schema-v2 compiled profile.

## Pipeline

1. `robot_registry_parser.ensure_compiled_retarget_profile()` resolves the registered robot and generated profile path.
2. `robotics.morphology.analyze_mjcf_morphology()` extracts body tree, joint axes, limits, rest positions, explicit MJCF sites, primitive geom bounds, STL mesh bounds, and model warnings.
3. `robotics.task_compiler.compile_retarget_profile()` builds semantic sites from explicit maps plus conservative body-name inference, explicit-site/child-anchor/mesh/geom distal virtual sites, chain profiles, reachability bases, normalized task specs, contact defaults, solver priority bands, confidence, and warnings.
4. Runtime config loading attaches `compiled_retarget_profile`, priority diagnostics, direction tasks, and pole-vector tasks.
5. `pipelines.newton_pipeline.NewtonPipeline` uses the compiled profile to instantiate only enabled sparse objectives.

## Runtime Objectives

- Position tasks remain analytic IK objectives for enabled end-effectors and root targets.
- Root motion metadata records semantic hips/feet leg-length scale, nominal pelvis height, ground height, and provenance.
- Torso rotation targets are projected to the compiled reachable basis before they enter IK.
- Middle-limb direction tasks track parent-to-child unit vectors.
- Pole-vector tasks track bend-plane normals and fall back per environment on degenerate source geometry.
- Contact toe/heel targets use per-environment targets, per-environment weights, hysteresis, minimum stance duration, and release blending.
- Optional ground barriers penalize sole-anchor penetration with per-environment stance/swing weights.
- Joint-limit and temporal objectives are range and sample-rate normalized.
- Compiled collision metadata records mesh/geom-derived sphere proxies, high-risk non-adjacent pairs, and margins. Runtime sphere-pair barriers are optional and enabled with `collision_weight`.

## Generated Artifacts

`soma_retargeter.tools.benchmark_retargeting` currently generates reproducible compile-level benchmark artifacts:

- `benchmark_summary.json`
- `benchmark_frames.csv`
- `environment.json`
- `commands.txt`
- `per_robot/<robot>.json`
- `failures/<robot>.json` on failure

Motion-level runtime metrics are intentionally marked `not_run` until the runtime benchmark loop is implemented.
