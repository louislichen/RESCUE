# RESCUE Core Code Handoff Archive — 2026-09-22

This directory contains the latest validated RESCUE handoff source archive produced from `RESCUE_CODE_HANDOFF_20260922.zip`.

## What is included

The reconstructed tarball contains the current handoff code needed by a successor researcher:

- frozen Harness V3 and patched runtime sources;
- pre-experiment correctness gate and identity/provenance logic;
- four-arm QA/K4 runners;
- incremental K4 verifier and correctness tests;
- decision-relevance diagnostics;
- Student local Oracle / controllability diagnostics;
- R1 value-norm pilot code;
- Teacher decision-attribution Phase C code;
- project-control files;
- relevant configs, tests and reproduction documentation.

Large result snapshots, figures, prior tensors and model weights are intentionally excluded from the core tarball. Their identities and provenance are documented in the handoff documentation and the project progress archive.

## Reconstruct

```bash
cat RESCUE_CODE_CORE_HANDOFF_20260922.tar.gz.part00 \
    RESCUE_CODE_CORE_HANDOFF_20260922.tar.gz.part01 \
  > RESCUE_CODE_CORE_HANDOFF_20260922.tar.gz

sha256sum RESCUE_CODE_CORE_HANDOFF_20260922.tar.gz
# expected:
# 33c1cd3e0766e4366f8999904208e592e31c7daa21776c2d004312119bb45404

tar -xzf RESCUE_CODE_CORE_HANDOFF_20260922.tar.gz
```

Part identities:

- part00: SHA256 `89943e1d8ccbf28f2b3aa2a7f9d63c77c8d516bba50bcd57827218f2411282dd`, 250000 bytes
- part01: SHA256 `f5d11632defe756dfcd352854fe35bb235a83627a705de14680b74ff896091f0`, 240685 bytes

Full core archive:

- SHA256 `33c1cd3e0766e4366f8999904208e592e31c7daa21776c2d004312119bb45404`
- size: 490685 bytes
- packaged files: 247

Original complete local handoff ZIP received by the web supervisor:

- `RESCUE_CODE_HANDOFF_20260922.zip`
- SHA256 `ffa8da6207a65a784c22e4e2225fe5330dedcdd24c2328918d6941486b81fe37`
- size: 2907881 bytes
- 434 files total
- package-internal `SHA256SUMS.txt`: 433/433 OK
- quick secret scan: 0 hits

The full handoff ZIP also contains curated result snapshots, figures and the small adapter checkpoint. The GitHub core archive focuses on executable source and handoff metadata; large/external artifacts remain referenced by identity rather than duplicated.

## Validity

Before quoting results, read:

- `../../../../docs/progress-archive/2026-09-21/VALIDATED_PROGRESS_SUMMARY.md`
- `../../../../docs/progress-archive/2026-09-21/POST_V8_LATEST_PROGRESS.md`
- `../../../../docs/progress-archive/2026-09-21/EVIDENCE_INDEX.md`

Historical invalidated results remain invalidated. R1 is retired. HSPR/sign-preserving transport is only proposed/not run.
