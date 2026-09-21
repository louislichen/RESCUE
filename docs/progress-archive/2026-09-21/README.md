# RESCUE Progress Archive — 2026-09-21

This branch archives the current RESCUE research progress under the `liangchen-man` namespace without modifying `main`.

## Scope

- Historical canonical control-plane snapshot available to the web session: **V8 / PROJECT_STATE_VERSION 2026-09-21.9**.
- Post-V8 locally returned progress is archived separately in `POST_V8_LATEST_PROGRESS.md`.
- The latest local canonical V10 bundle was **not available to this web session**, so this branch does **not** fabricate or overwrite canonical V10 control files.
- Historical invalidated Adapter results remain invalidated.
- No new experiment is authorized by this archive.

## Key research state

1. Harness V3 was independently accepted and passed the real-GPU pre-experiment correctness gate.
2. A valid four-arm rerun produced CASE 3: Base, Prefill-Teacher and Full-window-Teacher had identical QA/K4 development behavior; Teacher did not outperform Null.
3. Decision-relevance diagnostics showed current Teacher attention guidance was nearly orthogonal to Target correction at Student–Target disagreement positions.
4. Same-budget Student local Oracle demonstrated substantial controllability: 42/67 disagreement decisions could be flipped, proving the injection site has headroom.
5. R1 = attention × ||V|| was preregistered and failed; it is retired.
6. Post-V8 TD Phase C found strong Teacher decision-aware signal at the Teacher side, but severe loss in cross-head transport/mixing before Student exploitation.
7. Project execution is paused after archival; future HSPR/sign-preserving operator work is only a proposal, not an executed result.

See:
- `control-plane-v8/` for the exact V8 snapshot.
- `POST_V8_LATEST_PROGRESS.md` for the newest locally returned results after V8.
- `PAUSE_MARKER.md` for the current execution stop state.
