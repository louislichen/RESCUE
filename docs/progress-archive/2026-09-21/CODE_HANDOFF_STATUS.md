# RESCUE Code Handoff Status

Status: **CODE_HANDOFF_PENDING**

The current archive branch contains validated progress documentation and the repository's pre-existing source tree, but it does **not yet contain the complete latest local implementation produced during the 2026-09-20/21 correctness-closure and mechanism-diagnostic work**.

Missing handoff code includes, at minimum, the finalized Harness V3 implementation and the locally executed experiment/diagnostic runners used for:

- correctness closure and PRE_EXPERIMENT_CORRECTNESS_GATE;
- four-arm valid rerun;
- decision-relevance diagnostics;
- local Student controllability / same-budget Oracle diagnostics;
- R1 value-norm pilot;
- Teacher decision-attribution Phase C.

Do not treat this branch as a complete executable handoff until this marker is replaced with **CODE_HANDOFF_COMPLETE** and the source manifest has been committed.

The legacy repository code under `exp/` remains available, but it is not a substitute for the latest validated local harness/runner code.
