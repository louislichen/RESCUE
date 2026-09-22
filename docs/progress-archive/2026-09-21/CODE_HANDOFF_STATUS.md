# RESCUE Code Handoff Status

Status: **CODE_HANDOFF_COMPLETE / MAIN_SOURCE_EXPANDED**

The RESCUE work represented by this branch is paused and handed off to the next researcher.

## Directly browsable source

The principal source tree is now expanded under:

`handoff/2026-09-22/source/`

It includes directly browsable implementation for:

- Harness V3 pre-experiment correctness gate, identity/provenance entry points and integrity tests;
- Qwen Adapter backend and frozen Adapter/source-lock steering definitions;
- per-prefix sequence-supervision implementation and runner;
- fixed-K=4 protocol and tests;
- valid four-arm preparation / QA / K4 / orchestration;
- decision-relevance preparation / common metrics / paired measurement / orchestration;
- Student local controllability and same-budget Oracle diagnostic;
- R1 value-norm pilot;
- Teacher decision-attribution Phase C;
- mechanism-sprint probes, feature extraction and orchestration;
- V3 GPU/resource/identity/result-integrity/roll-up test entry points;
- frozen config identities and handoff test documentation.

See:

`handoff/2026-09-22/source/README.md`

for the handoff map and scientific status of each code family.

## Complete package

The verified original package remains the byte-level authority for auxiliary files that are not
expanded directly in GitHub:

- `RESCUE_CODE_HANDOFF_20260922.zip`
- SHA256 `ffa8da6207a65a784c22e4e2225fe5330dedcdd24c2328918d6941486b81fe37`
- 434 files total
- package-internal SHA256 verification: 433/433 OK
- quick secret scan: 0 hits

Very large auxiliary analysis/test modules, result snapshots, figures and external tensor/model
artifacts are not duplicated blindly into the browsable source tree. A connector-truncated copy
must never be treated as canonical; the full package / manifest is authoritative for those files.

## Repository comparison

Relative to `main` at `fc46ae2ab26cd30fe3fdc92684fb08d51f76a5a9`, the handoff branch contains
dozens of handoff commits and directly visible source/config/test additions.

## Research validity

- Harness V3: accepted / PRE gate PASS
- four-arm rerun: VALID DEVELOPMENT / CASE 3
- decision relevance: VALID DEVELOPMENT / INFO-B
- mechanism sprint: VALID DEVELOPMENT / CONTROLLABILITY-A
- R1: RETIRED / R1-FAIL
- Teacher decision attribution: TD-A YES / TD-B LOSSY / TD-C NOT EXPLOITABLE under frozen transport
- HSPR / sign-preserving transport: PROPOSED / NOT_RUN
- historical pre-correctness Adapter results: INVALIDATED

No new RESCUE research execution is authorized by this handoff.
