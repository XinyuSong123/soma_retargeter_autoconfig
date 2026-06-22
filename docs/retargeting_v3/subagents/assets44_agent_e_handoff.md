# Assets44 Agent E Handoff

Area: full validation and failure analysis.

Result: PASS.

Findings:

- Full validation is slow, not hung. Expensive positive humanoids include `ergocub_urdf`, JVR C variants, Toddlerbot, and large NASA/fetch-only humanoids.
- Single-robot probes confirm source/load/semantic closure and expose new real algorithm failures for several newly mapped humanoids.
- Failure diagnostics must be copied from final `failures/*.json` reports after full artifact generation.

Final evidence:

- `summary.json` records source/load/semantic failures at zero.
- `failures/` reports contain motion/task/metric/actual/threshold/rank details for every algorithm failure.
- `deterministic_rerun.json` records 33 matched comparisons.
