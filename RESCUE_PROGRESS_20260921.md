# RESCUE Progress Snapshot — 2026-09-21

This branch archives the current RESCUE progress under the `liangchen-man` namespace.

Start here:

- [Progress archive index](docs/progress-archive/2026-09-21/README.md)
- [Validated progress summary](docs/progress-archive/2026-09-21/VALIDATED_PROGRESS_SUMMARY.md)
- [Post-V8 latest progress](docs/progress-archive/2026-09-21/POST_V8_LATEST_PROGRESS.md)
- [Evidence index](docs/progress-archive/2026-09-21/EVIDENCE_INDEX.md)
- [Pause marker](docs/progress-archive/2026-09-21/PAUSE_MARKER.md)
- [Historical V8 snapshot](docs/progress-archive/2026-09-21/control-plane-v8/RESCUE_SYNC_SNAPSHOT.md)

## Current research takeaway

The corrected and independently reviewed harness established a trustworthy development baseline. The validated four-arm rerun found no QA/K4 benefit from the current Teacher-attention guidance. Mechanism diagnostics then showed:

- the Student attention injection site has substantial local controllability;
- raw Teacher attention and the R1 value-norm variant do not recover useful decision direction;
- the Teacher itself contains strong decision-aware signal;
- current cross-head softmax transport strongly cancels signed decision information before Student exploitation.

No new sign-preserving transport method has been executed. Further experimental work is paused by user decision.

## Validity

Historical invalidated results remain invalidated. See the archive for exact claim boundaries and provenance.
