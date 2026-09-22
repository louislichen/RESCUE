# RESCUE Code Handoff Status

Status: **CODE_HANDOFF_IN_PROGRESS**

The validated progress archive is complete, and the original handoff ZIP has been verified locally. The latest source is now being expanded as directly browseable files under:

`handoff/2026-09-22/source/`

Already expanded in GitHub include core experiment entry points such as:

- Harness PRE_EXPERIMENT correctness gate;
- fixed K=4 protocol and tests;
- Qwen Adapter backend;
- prefix sequence-supervision fix;
- four-arm common/QA/K4 runners;
- decision-relevance preparation;
- entropy-conditioned Adapter wrapper.

The remaining large Harness/diagnostic runners are being copied from the handoff package without algorithmic changes.

Do **not** treat this status as CODE_HANDOFF_COMPLETE until the principal source tree (Harness V3, runtime, four-arm, decision relevance, Student Oracle, R1 pilot, Teacher decision attribution, tests/configs) is directly browseable and the final source manifest is committed.

Original complete local package:

- `RESCUE_CODE_HANDOFF_20260922.zip`
- SHA256 `ffa8da6207a65a784c22e4e2225fe5330dedcdd24c2328918d6941486b81fe37`

Large model weights and heavy tensor artifacts remain external by design.
