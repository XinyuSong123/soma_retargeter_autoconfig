# Numerical Agent D Handoff: Target And Projection

Scope: parent-frame rotation transfer, canonical independence, and residual normalization.

Findings incorporated:

- Canonical capability motions were order-dependent because prior solutions carried across motions.
- Endpoint residual normalization used sampled joint-limit endpoint displacement rather than chain length.
- Rotation transfer needed the parent-frame delta convention from `goal.md`.

Integrated result:

- Canonical motion projection defaults to `use_continuity_prior=False` and `continuity_prior_weight=0.0`.
- Continuity is opt-in and separated from capability benchmark behavior.
- Endpoint position projection normalizes by neutral kinematic chain length and records `normalization_reference="neutral_chain_length"`.
- Target builder uses parent-frame rotation delta `R(t) R0^T` and applies it before the robot neutral relative rotation.
- Extreme joint-limit stress records residual evidence without forcing reachability.
