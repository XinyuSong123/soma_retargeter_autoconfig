# Retargeting v2 Known Limitations

- Runtime benchmark artifacts now include separated bounded legacy/v2 smoke rollouts and profile-task residual summaries when BVH motions and semantic site trajectories are available, but full-motion comparison and solver-native residual extraction beyond those summaries are not implemented yet.
- Profile validation now covers numeric health gates and left/right chain length mismatch thresholds, but true symmetry tying is still compile-time reporting rather than a separate constrained optimizer.
- Self-collision runtime barriers are optional and disabled by default through `collision_weight=0.0`; motion-level evidence is still needed before making them a default feasibility term.
- E3 v2 and OLI are not registered in the current workspace, so benchmark artifacts cover only `roboparty_rpo` and `unitree_g1`.
- Unitree G1 v2 profile currently reports zero chain ranks and placeholder chain lengths, indicating incomplete registry/morphology coverage for that asset path.
- Semantic auto-detection now fills missing entries with conservative body-name matching, but topology and rest-pose spatial inference are still shallow; explicit `ik_map` remains the most reliable path.
- Virtual hand and foot sites now prefer explicit MJCF sites, distal child anchors, and then primitive geom or STL mesh distal bounds when available, but non-STL mesh formats and rotated/fromto geom bounds are still limited.
- Root/ground metadata uses compile-time semantic hips and virtual foot site rest positions only; runtime root-height stabilization still needs motion-level acceptance metrics.
- The priority residual guard currently protects joint-limit margin residuals, not all high-priority feasibility residuals.
