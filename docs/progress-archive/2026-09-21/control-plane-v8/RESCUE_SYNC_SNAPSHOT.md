# Historical V8 Control-Plane Snapshot

This is an exact archival copy of the web-supplied V8 sync snapshot content as of PROJECT_STATE_VERSION 2026-09-21.9.

PROJECT_STATE_VERSION: 2026-09-21.9

CURRENT_PHASE:
- HARNESS_FROZEN / PRE_EXPERIMENT_VALIDATION
- FOUR_ARM_DEVELOPMENT_REMEASUREMENT: COMPLETE (CASE 3)
- DECISION_RELEVANCE_DIAGNOSTIC: COMPLETE (CASE INFO-B)
- GUIDANCE_REPRESENTATION_MECHANISM_SPRINT: COMPLETE (CONTROLLABILITY-A)
- R1 VALUE-NORM PILOT: COMPLETE -> R1-FAIL
- NEXT METHOD CANDIDATE: NONE / NOT_YET_AUTHORIZED

HARNESS_STATUS:
- FROZEN_V3
- PRE_EXPERIMENT_CORRECTNESS_GATE: PASS

VALID DEVELOPMENT RESULTS:
1. Four-arm rerun: 128 rows, CASE 3; B=C=A token-wise; C-D accepted-token difference 0 on all rows.
2. Decision relevance: 496 positions, S=T 429 / S!=T 67; Teacher delta margin -0.0135, delta logp -0.0157, cosine ~0, 0/67 flips.
3. Mechanism sprint: same-budget Oracle delta margin +1.73, median closure 1.011, flips 42/67; Teacher efficiency median -0.0065; no conditional signal.
4. R1 value-norm pilot: R1-FAIL; R1 mean delta margin identical to R0, larger perturbation but 0/67 flips.

INVALIDATED:
- old Adapter QA / old Null QA / old Adapter K4 / old Null K4
- old full-window paired quality conclusions
- timing-truncation conclusion
- old sequence-loss conclusion
- derived prior-quality conclusion

NEXT_TASK: NOT_YET_AUTHORIZED
NEXT_METHOD_CANDIDATE: NONE

