# RESCUE Pilot Scope and Reproducibility Notes

## Scope

This branch publishes the validated implementation artifacts for local
document/span KV recomputation and write-back under context reuse. It is a
research prototype, not a production runtime or performance release.

The reported study is a small, single-model, single-dataset pilot on frozen
HotpotQA windows. R2/R6 use 43 evaluated examples; R7/R8 use 44 evaluated
examples after the documented exclusion. These populations must remain
separate.

## What is included

- The R2 final span-repair implementation at
  `exp/kv_reuse_attn/answer_difference/length_sweep_attention_decode_eval_maxL_fixed_v2.py`.
- R6 support code and deterministic exclusion tests under
  `RESCUE_R6_R11_SPRINT/code/`.
- Accepted R7 span-repair, selector, attention-asset, gate-statistics, and
  CPU-test scripts under `RESCUE_R6_R11_SPRINT/code/`.

The source commits and file-level provenance are recorded in the release
receipt, not inferred from filenames. Existing upstream files are retained as
upstream history; no raw evidence package is added by this publication.

## Evidence and claim boundary

- R7/R8 metrics are descriptive and reproducible under frozen metric semantics.
- Strict actual-cost matching is `NOT_VERIFIED`; nominal budgets are not equal
  actual computation.
- Teacher advantage is `NOT_ESTABLISHED`.
- Teacher–Student attention divergence is not supported by the current evidence.
- R9 V2 is `CLOSED_INVALID`, not a scientific failure; full-660 generalization
  remains unresolved.
- R11 is feasibility-only and does not establish backend performance.

Accordingly, this branch does not claim that selective repair is effective,
efficient, faster, causal, superior, or generalizable.

## Verification

Only CPU-only syntax, pure-logic/unit, help, diff, provenance, secret/path,
and file-size checks are appropriate for this release. Do not load model
weights, datasets, CUDA, caches, or raw decode outputs as part of the release
verification.

## License

The upstream repository license and notices remain unchanged.
