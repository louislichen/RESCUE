# RESCUE Code Handoff Status

Status: **FULL_DIRECT_SOURCE_COMPLETE**

84 / 84 VALID_CURRENT / VALID_DIAGNOSTIC executable source files are directly present in GitHub.

## What this means

Every executable source file that the verified handoff package classifies as `VALID_CURRENT` or
`VALID_DIAGNOSTIC` is now directly browsable under `handoff/2026-09-22/source/`, and each one is a
**byte-for-byte copy** of its counterpart inside the verified handoff ZIP. Nothing in that tree was
rewritten, shortened, reformatted, or reproduced from chat text.

The original complete handoff ZIP remains the **byte-level authority** for the whole package,
including the non-executable material (docs, configs, results, project-control, figures).

## Source of truth

`RESCUE_CODE_HANDOFF_20260922.zip`

- SHA256 `ffa8da6207a65a784c22e4e2225fe5330dedcdd24c2328918d6941486b81fe37`
- 434 package members
- internal `SHA256SUMS.txt` verified **433/433 OK**
- 74 `VALID_CURRENT`/`VALID_DIAGNOSTIC` Python files + 10 shell scripts = 84 executable sources

## How completeness was established

The expected set is enumerated mechanically from `CODE_MANIFEST.json` inside the ZIP, filtered to
`status in (VALID_CURRENT, VALID_DIAGNOSTIC)` and extension in `.py`/`.sh`. For each file the
**working-tree SHA256** and the **git index blob** were compared against the ZIP bytes; the index was
written with `core.autocrlf=false` so byte content is preserved exactly rather than normalised.

```text
expected            = 84
present             = 84
sha256_match        = 84
missing             = 0
mismatch            = 0
```

### Fidelity restoration — two commits

A strict audit against the ZIP showed the previously expanded tree was **not** byte-faithful even
where paths existed. The pre-task state of the 84 files was:

```text
byte-identical to the ZIP : 34
present but NOT faithful  : 33   (reformatted, with statements removed or replaced by a stub)
absent                    : 17
```

- The 33 non-faithful copies had been reformatted and, in places, had code removed. Examples:
  `harness/rescue_harness/harness.py` was 43 247 bytes in the ZIP but 10 143 bytes on the branch,
  ending with the explicit stub `# ... ReSCUE harness continues ... exactly as in the handoff
  package.`; `diagnostics/mechanism_sprint/td_features.py` had lost its periodic progress print;
  `runners/four_arm/run_four_arm.sh` was 7 467 -> 2 989 bytes.
- **Commit 1** added only the 17 files that were absent, preserving the original task boundary.
- **Commit 2** overwrote all 33 non-faithful files with the corresponding ZIP bytes.
- No web-side variant was retained, summarised, or merged. The ZIP bytes are the only source of
  truth.

## Evidence artefacts

- `handoff/2026-09-22/DIRECT_SOURCE_REMAINDER_MANIFEST.json` — the 17 files added in commit 1,
  each `MATCH` (source-ZIP sha256 vs copied sha256)
- `handoff/2026-09-22/DIRECT_SOURCE_COMPLETENESS_AUDIT.json` — expected/present/missing
- `handoff/2026-09-22/DIRECT_SOURCE_FIDELITY_AUDIT.json` — per-file ZIP sha256 vs GitHub content,
  plus the pre-task state of every file
- `handoff/2026-09-22/DIRECT_SOURCE_FIDELITY_RESTORATION.json` — the 33 files restored in commit 2
  and why

## Directly expanded tree now covers

- Harness V3 gate, provenance builder, applied-diff tool, targeted regressions, integrity tests
- `rescue_harness/` in full: harness, training identity, regression scenarios, GPU smokes, CPU suite
- patched source-lock worker (`patched_sources/q3_teacher_prior_worker.py`)
- Qwen adapter backend, entropy adapter, per-prefix sequence supervision, fixed-K4 protocol
- incremental K4 runner, divergence forensics, target numerics probe
- four-arm QA/K4/mechanism/aggregate runners and orchestration
- decision-relevance prepare/common/stream/aggregate runners and orchestration
- Student local Oracle controllability, mechanism-sprint probes and summarisation
- R1 pilot and aggregation
- Teacher decision-attribution prepare/attribution/student-alignment/aggregate
- V3 gate scripts, identity/resource/integrity/roll-up checks, K4 and backend tests
- handoff utilities (`tools/refresh_hashes.py`, `runners/make_run_tree.sh`)

## Notes for a successor

- `runtime/source_lock/steering.py` is the **frozen COMPAT-001/002 baseline** (its sha256 is part of
  the checkpoint identity contract). It is **not** the runtime. The validated runtime is the
  `STEERING_SOURCE` string inside `harness/rescue_harness/harness.py`. Do not install `steering.py`
  as the hook.
- `runtime/source_lock/run_adapter_protocol.py` legitimately contains two CRLF bytes in the frozen
  original; they were committed verbatim. Anyone re-checking out this repository with
  `core.autocrlf=true` may get local line-ending conversion — verify against the ZIP with
  `core.autocrlf=false` if byte equality matters.
- `REPRODUCTION.md`, `VALIDITY_AND_CLAIMS.md`, `HANDOFF_STATE.md` and `EXTERNAL_ARTIFACTS.md` inside
  the ZIP are the authoritative reading order.

## Unchanged scientific boundaries

Historical invalidated results remain invalidated. R1 remains retired. HSPR remains PROPOSED / NOT_RUN.
No new research was executed in producing this status.
