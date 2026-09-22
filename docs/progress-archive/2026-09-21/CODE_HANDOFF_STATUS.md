# RESCUE Code Handoff Status

Status: **MAIN_SOURCE_EXPANDED / FULL_DIRECT_SOURCE_PENDING**

The branch contains a substantial directly-browsable handoff source tree under:

`handoff/2026-09-22/source/`

However, a strict audit against the verified handoff ZIP shows that **not every VALID_CURRENT /
VALID_DIAGNOSTIC source file has yet been expanded directly into GitHub**. The complete ZIP remains
the byte-level authority until the remaining runnable dependencies are copied.

## Direct-source coverage at this audit

Verified handoff package:
- `RESCUE_CODE_HANDOFF_20260922.zip`
- SHA256 `ffa8da6207a65a784c22e4e2225fe5330dedcdd24c2328918d6941486b81fe37`
- 434 package members
- 74 VALID_CURRENT/VALID_DIAGNOSTIC Python files
- 10 VALID_CURRENT/VALID_DIAGNOSTIC shell scripts

GitHub directly-browsable source tree currently includes the main research paths:
- Harness V3 gate / provenance entry points / integrity tests
- Qwen Adapter backend and frozen source-lock adapter/common/steering
- per-prefix sequence supervision
- fixed-K4 protocol
- four-arm QA/K4 preparation/runners/orchestration
- decision-relevance prepare/common/measurement/orchestration
- Student local Oracle / controllability diagnostics
- R1 pilot
- Teacher decision-attribution main runner
- mechanism-sprint probes/orchestration
- V3 resource / identity / integrity / roll-up tests
- frozen configs and test entry points

## Important remaining direct-source gaps

At least these runnable dependencies are still present only in the verified handoff package and
must be expanded before declaring FULL_DIRECT_SOURCE_COMPLETE:

- `harness/rescue_harness/training_identity.py`
- `harness/rescue_harness/gpu_smokes.py`
- `harness/rescue_harness/regression_scenarios.py`
- `harness/rescue_harness/test_harness.py`
- `runtime/prefix_fix/real_k4_runner.py`
- `runtime/source/sequence_loss_runner.py`
- `runners/four_arm/four_arm_aggregate.py`
- `runners/decision_relevance/dr_aggregate.py`
- `runners/teacher_decision/td_prepare.py`
- `runners/teacher_decision/td_student_alignment.py`
- `runners/teacher_decision/td_aggregate.py`
- additional mechanism-sprint aggregation/grounding/failure-structure utilities and figure/report helpers

Therefore this branch is already useful for code review and understanding, but the directly expanded
tree should **not yet be described as a completely standalone replacement for the verified ZIP**.

Historical invalidated results remain invalidated. R1 remains retired. HSPR remains PROPOSED / NOT_RUN.
