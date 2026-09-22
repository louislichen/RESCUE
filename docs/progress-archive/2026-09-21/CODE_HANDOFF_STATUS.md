# RESCUE Code Handoff Status

Status: **CODE_HANDOFF_COMPLETE**

The `liangchen-man/rescue-progress-20260921` archive branch now contains:

1. validated research-progress documentation and evidence index;
2. the pre-existing RESCUE repository source tree;
3. a frozen core code handoff archive for the latest validated 2026-09-20/21 implementation.

Core handoff location:

`handoff/2026-09-22/core/`

Reconstruct with the instructions in:

`handoff/2026-09-22/core/README.md`

The reconstructed archive contains the validated latest handoff code for:

- Harness V3 correctness and identity/provenance logic;
- PRE_EXPERIMENT_CORRECTNESS_GATE;
- four-arm QA/K4 runners;
- incremental K4 correctness implementation;
- decision-relevance diagnostics;
- Student local Oracle / controllability diagnostics;
- R1 value-norm pilot;
- Teacher decision-attribution Phase C;
- tests, configs, project-control files and reproduction documentation.

Core archive SHA256:

`33c1cd3e0766e4366f8999904208e592e31c7daa21776c2d004312119bb45404`

Original complete local handoff ZIP supplied to the web supervisor:

- `RESCUE_CODE_HANDOFF_20260922.zip`
- SHA256 `ffa8da6207a65a784c22e4e2225fe5330dedcdd24c2328918d6941486b81fe37`
- 434 files
- package-internal SHA256 verification passed
- quick secret scan found 0 hits

Large model weights, prior tensors and other heavy external artifacts are intentionally not duplicated in GitHub. Their identities and provenance are documented in the handoff/archive materials.

Historical invalidated results remain invalidated. R1 remains retired. HSPR/sign-preserving transport remains PROPOSED / NOT_RUN.

The RESCUE work represented by this branch is paused for handoff to the next researcher.
