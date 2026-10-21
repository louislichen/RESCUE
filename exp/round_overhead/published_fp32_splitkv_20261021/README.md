# FP32 split-KV: verified execution update (2026-10-21)

This publication contains the validated execution code, fixed protocols, CPU
tests and a compact copy of the completed experiment report/figures. The new M1
selector-maintenance prototype is still under development and is not included.

## Result, not a >10% success claim

The fair primary contrast gives **both** frozen RESCUE and the Qwen2 retrieval
Top32/fixed-k4 Graph port the same new Target attention implementation. It is
not a comparison against ordinary SD or an unchanged slower execution baseline.

|Actual input Tokens|RESCUE TPS|Retrieval TPS|Relative TPS gain|Paired 95% CI|
|---:|---:|---:|---:|---|
|2048|59.7310|56.2862|+6.1202%|[+0.5260%,+12.3649%]|
|4096|54.0572|52.1780|+3.6015%|[−2.7339%,+11.4754%]|
|6144|51.7561|50.7529|+1.9767%|[−3.0809%,+8.0014%]|
|8192|47.0486|44.9038|+4.7765%|[+0.3885%,+10.0033%]|
|12288|38.6362|38.0439|+1.5570%|[−2.4437%,+6.0667%]|
|16384|31.5908|32.9003|−3.9801%|[−7.8618%,−0.3007%]|

The execution change accelerates RESCUE itself by 8.01%–32.11%, but retrieval
also improves by 6.56%–31.83%. The six-length relative >10% objective remains
**not achieved**. All 792 evaluated outputs exactly match the frozen original
Target canonical Tokens; 384 original/candidate work and acceptance trajectories
are unchanged. Intermediate Target floating-point values need not be bitwise
identical. Target decode is FP32 with global TF32 disabled; Draft/prefix BF16.

These are 16 **reused development documents**, two repeats, one nonconcurrent
private owner block per arm/context, cold doc271 plus 32 warm requests. All real
late Graph captures and request-time JIT remain charged. Paired whole-document
bootstrap: 10,000 draws, seed 20261031, pointwise 95% intervals, not simultaneous
and not covering owner/clock drift. No fresh-confirmation, fully-prewarmed
production, general stochastic-distribution or original-paper/SOTA claim.

## Code and protocols

- [Default-off FP32 attention candidate](../fp32_splitkv_candidate_v1.py)
- [Scoped Target adapter](../fp32_splitkv_target_adapter_v1.py)
- [Model Token gate](../fp32_splitkv_model_gate_v1.py)
- [Four-arm timing runner](../fp32_splitkv_timing_v1.py)
- [Fixed timing preregistration](../fp32_splitkv_timing_v1_preregistration.json)
- [Portable report/plot analysis](../fp32_splitkv_timing_report_v1.py)
- [Independent saved-data audit](../fp32_splitkv_timing_audit_v1.py)
- [Read-only collector](../collect_fp32_splitkv_timing_v1.py)

The operator V1/V2 gates remain failed historical checks; V3 changes only its
explicit analytic-reference contract and is not retroactively applied to them.
The candidate, gates and launchers do not silently execute a model on import.
Launchers deliberately require explicit opt-in and fixed isolated paths. Review
their environment/assets/pins before any use; do not restart an existing root.

Before publication, the complete FP32-related CPU regression selection passed
114/114 tests (zero skips), and 198 selected Python files plus 18 shell launchers
passed syntax checks. Archive-specific tests used the existing local evidence;
they do not imply that excluded raw archives are distributed here.

## Report and figures

[Full report](analysis_v2/REPORT.md) ·
[Relative TPS gain](analysis_v2/relative_tps_gain.png) ·
[Acceptance length](analysis_v2/acceptance_length.png) ·
[Actual whole-generate TPS](analysis_v2/warm_tps.png)

The nine PNG/SVG pairs separately show acceptance length/rate, TPS, relative
gain, both execution gains, finite-owner costs, Graph captures, actual Target
call metadata and round/work descriptors. These were actually rendered and
visually inspected, not inferred from source code alone.

Original full local evidence seal SHA256:
`9b727ca38bea86343dd6d5004342da2969ff3b6f1b4d605016445fdf967a47bd`.
Timing terminal SHA256:
`e56ffb5f51c7d399cdce4b930c6a09a9f0265fffbc5cc93540f296efb3d64b1f`.
Summary SHA256:
`4776d5b5f199a879a1ccaa4eea373e3789baf10692eb864b2615e21dc2a4ca35`.

The compact publication intentionally excludes weights, credentials, compiled
caches, natural-document/canonical generation dumps and the full 83 MB raw
archive. Full-report links to raw/proof paths therefore require the original
local archive; this publication alone is not a self-contained raw-data rerun.
The saved report CLI reproduced the summary byte-for-byte using that archive
without Torch/runtime imports; independent audit passed 25,891 checks and a
second report comparison passed 1,293 checks.
