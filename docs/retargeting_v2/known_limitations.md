# Retargeting v2 Known Limitations

- Runtime motion benchmark metrics are not implemented yet. Current artifacts are compile-level profile checks with runtime metrics marked `not_run`.
- Self-collision runtime barriers are optional and disabled by default through `collision_weight=0.0`; motion-level evidence is still needed before making them a default feasibility term.
- E3 v2 and OLI are not registered in the current workspace, so benchmark artifacts cover only `roboparty_rpo` and `unitree_g1`.
- Unitree G1 v2 profile currently reports zero chain ranks and placeholder chain lengths, indicating incomplete registry/morphology coverage for that asset path.
- Semantic auto-detection is still shallow; explicit `ik_map` remains the reliable path.
- Virtual hand sites are conservative and do not yet use mesh/geom distal bounds.
- Root/ground metadata is compile-time provenance only; runtime root-height stabilization still needs motion-level acceptance metrics.
- The priority residual guard currently protects joint-limit margin residuals, not all high-priority feasibility residuals.
