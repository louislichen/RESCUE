# Frozen source-content ablations

Exploratory checks added after the primary shuffled-content result. No model, strength, checkpoint, candidate positions, or test-set tuning changed. Both mapper and gate receive either aligned teacher V, per-layer mean teacher V over the selected remote positions, or zero teacher V. Teacher-attention position selection remains available to all controls. Mean retains document-level teacher content; zero eliminates teacher V content input but is out of the trained source distribution. These controls do not replace a separately trained native-only adapter comparison.

| Source | Output KL | Next-token agreement % | Actual acceptance % | Decode TPS | Mean gate |
|---|---:|---:|---:|---:|---:|
| aligned | 1.0037 | 70.83 | 65.26 | 26.62 | 0.241 |
| mean | 0.9859 | 70.00 | 65.44 | 26.55 | 0.295 |
| zero | 1.0650 | 70.00 | 64.35 | 26.45 | 0.216 |

| Comparison | ΔKL [95% CI] | ΔAcceptance pp [95% CI] |
|---|---:|---:|
| aligned_vs_mean | +0.0177 [-0.0111, +0.0475] | -0.18 [-1.32, +0.89] |
| aligned_vs_zero | -0.0613 [-0.0902, -0.0342] | +0.91 [+0.16, +2.00] |

Paired document bootstrap: 5,000 resamples, 95% intervals, no multiplicity correction. Aligned acceptance/proposal/block counts and generated outputs exactly reproduce the primary run. Output agreement: 90/90. TPS comparisons here use this supplementary session only.

[Primary pilot](report.md).
