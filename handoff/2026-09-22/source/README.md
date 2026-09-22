# RESCUE Source Handoff — 2026-09-22

This directory is the **directly browsable source handoff** for the validated RESCUE work
performed through 2026-09-21/22.

It exists because a compressed-only handoff is not sufficient for a successor researcher:
the core implementation must be searchable and reviewable directly in GitHub.

## Directly expanded code

### harness/
- `PRE_EXPERIMENT_CORRECTNESS_GATE.py`
- `make_run_provenance.py`
- `targeted_regressions_002.py`
- `rescue_harness/harness.py`

These implement the Harness V3 correctness/integrity/provenance gate used before the valid
development experiments.

### runtime/
- `prefix_fix/prefix_seqloss.py`
- `prefix_fix/sequence_prefix_runner.py`
- `source/qwen_adapter_backend.py`
- `source/fixed_k4_protocol.py`
- `source/entropy_adapter.py`
- `source_lock/adapter.py`
- `source_lock/common.py`
- `source_lock/steering.py`

These cover the corrected Qwen2 Adapter path, per-prefix sequence supervision, the fixed-K=4
protocol, and the frozen method definitions.

### runners/four_arm/
- four-arm preparation/common machinery
- QA runner
- K4 runner
- orchestration script

Scientific status: **VALID DEVELOPMENT / CASE 3**.

### runners/decision_relevance/
- preparation
- common decision metrics
- paired A/C/D/T measurement stream
- orchestration

Scientific status: **VALID DEVELOPMENT / CASE INFO-B**.

### diagnostics/mechanism_sprint/
- Student local controllability / same-budget Oracle
- connectivity probe
- mechanism config preparation
- decision-feature extraction
- Track A/B/C/D orchestration scripts

Scientific status: **VALID DEVELOPMENT / CONTROLLABILITY-A**.

### runners/r1_pilot/
R1 value-norm candidate evaluation.

Scientific status: **R1-FAIL / RETIRED**.

### runners/teacher_decision/
Teacher decision-attribution Phase C implementation.

Scientific status:
- TD-A: YES
- TD-B: LOSSY
- TD-C: NOT EXPLOITABLE under the frozen transport

### tests/
- fixed-K4/backend tests
- V3 GPU preflight
- identity preflight
- result-integrity smoke
- V3 roll-up
- V3 orchestration

### configs/
Frozen identities/configs needed to understand the validated runs.

## Complete handoff package

The original complete handoff package remains the authoritative byte-level archive for files
that are not expanded here:

`RESCUE_CODE_HANDOFF_20260922.zip`

SHA256:

`ffa8da6207a65a784c22e4e2225fe5330dedcdd24c2328918d6941486b81fe37`

Package facts:
- 434 files total
- internal SHA256SUMS: 433/433 OK
- quick secret scan: 0 hits

Some very large auxiliary analysis/test modules and result snapshots are intentionally not copied
line-for-line into the browsable source directory when the connector cannot safely transfer the
whole text without truncation. **Do not treat a truncated copy as canonical.** Use the complete
handoff package / manifest for those files.

## External artifacts not committed

The repository intentionally does not duplicate:
- Qwen model weights
- large prior tensors
- large result tarballs
- large intermediate JSONL/tensor outputs
- credentials/secrets

Their identities and provenance are documented under
`docs/progress-archive/2026-09-21/EVIDENCE_INDEX.md`.

## Validity boundary

Historical pre-correctness Adapter results remain INVALIDATED.
R1 is RETIRED.
HSPR / sign-preserving head transport is PROPOSED / NOT_RUN.
No research execution is authorized by this handoff.
