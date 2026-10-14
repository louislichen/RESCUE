# Selective V injection: first pilot

Qwen2-7B → 1.5B, frozen models, FP32 teacher / BF16 student, H100 eager attention. 24 train / 12 validation / 30 new test documents; 600–2048 input tokens. 512-position working cache, 128 remote V corrections in four final student layers. Teacher signals come only from a completed prefix; no fresh teacher forward per draft token.

## Validation choices

Checkpoint and strength use validation output KL + 0.25 teacher-argmax CE. Tables retain forced nonzero diagnostic strengths; the zero-allowed choices are reported below.

| Method | Forced strength | Zero-allowed strength | Chosen epoch |
|---|---:|---:|---:|
| reconstruction | 0.05 | 0 | 2 |
| performance_fixed | 0.5 | 0.5 | 8 |
| performance_gated | 0.5 | 0.5 | 8 |

## Held-out output and actual decoding

KL and next-token agreement are document-equal means on teacher-forced states. Actual acceptance is total accepted draft tokens / total proposed tokens. TPS is total output tokens / total timed seconds. All SD routes receive the same completed-prefix attention acquisition; mapper and gate costs are included.

| Method | Output KL ↓ | Next-token agreement % | Actual acceptance % | Accepted/block | Decode TPS | Total TPS |
|---|---:|---:|---:|---:|---:|---:|
| native_full | 0.9969 | 68.33 | 64.35 | 2.492 | 36.12 | 18.11 |
| teacher_token | 1.0497 | 68.33 | 63.51 | 2.457 | 29.74 | 16.35 |
| reconstruction | 1.0533 | 68.33 | 63.29 | 2.452 | 27.80 | 15.74 |
| performance_fixed | 0.9924 | 70.83 | 64.82 | 2.485 | 28.10 | 15.84 |
| performance_gated | 1.0037 | 70.83 | 65.26 | 2.495 | 27.20 | 15.57 |
| gated_shuffled | 1.0057 | 70.00 | 65.39 | 2.500 | 26.88 | 15.41 |
| gated_constant | 1.0185 | 70.00 | 64.65 | 2.485 | 28.04 | 15.80 |
| teacher_greedy | — | — | — | — | 46.86 | 21.24 |

## Paired changes versus native teacher-token selection

Document bootstrap, 5,000 paired resamples, 95% intervals; no multiplicity correction.

| Method | ΔKL [95% CI] | ΔAcceptance pp [95% CI] | Decode TPS ratio [95% CI] |
|---|---:|---:|---:|
| native_full | -0.0528 [-0.1342, +0.0110] | +0.84 [-1.11, +3.19] | 1.215 [1.194, 1.244] |
| teacher_token | +0.0000 [+0.0000, +0.0000] | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| reconstruction | +0.0035 [-0.0024, +0.0098] | -0.21 [-0.90, +0.24] | 0.935 [0.933, 0.938] |
| performance_fixed | -0.0573 [-0.1045, -0.0164] | +1.32 [+0.20, +2.90] | 0.945 [0.936, 0.959] |
| performance_gated | -0.0461 [-0.0945, -0.0056] | +1.75 [+0.53, +3.52] | 0.915 [0.906, 0.928] |
| gated_shuffled | -0.0441 [-0.0864, -0.0097] | +1.88 [+0.59, +3.70] | 0.904 [0.895, 0.918] |
| gated_constant | -0.0313 [-0.0619, -0.0052] | +1.15 [+0.24, +2.50] | 0.943 [0.936, 0.955] |

## Gate and content controls

gated_constant uses the same performance-gated mapper with a constant validation mean gate. gated_shuffled retains unshuffled teacher content as gate input and unchanged attention locations, shuffling only mapped content directions. Later query states and realized gates can change through upstream content corrections; gates are not replayed. This tests position-content alignment, not complete removal of all teacher content from the gate.

| Comparison | ΔKL [95% CI] | ΔAcceptance pp [95% CI] |
|---|---:|---:|
| performance_gated − performance_fixed | +0.0113 [-0.0040, +0.0306] | +0.44 [+0.00, +1.05] |
| performance_gated − gated_constant | -0.0148 [-0.0336, +0.0016] | +0.61 [+0.00, +1.65] |
| performance_gated − gated_shuffled | -0.0020 [-0.0119, +0.0086] | -0.13 [-0.53, +0.18] |
| performance_fixed − native_full | -0.0045 [-0.0810, +0.0872] | +0.47 [-1.86, +2.73] |
| performance_gated − native_full | +0.0068 [-0.0745, +0.1004] | +0.91 [-1.53, +3.25] |

## Integrity and limits

- Complete unique records: 826 offline, 240 runtime. SD output agreement with the separately generated FP32 teacher: 210/210.
- Zero gate exactly reproduces the native selection output; backing K/V are unchanged by the read transformation; only selected remote positions are corrected. Gate derivative finite-difference check passed. Frozen student weights receive no gradients during performance training.
- Output supervision uses the shared tokenizer vocabulary, not full unshared padded logits. Greedy acceptance is evaluated, not directly differentiated. Test documents are new but training and validation are reused from stage two.
- This is V-only content injection, not K+V injection. K is not mapped and Q/K projection weights remain frozen; later query activations can respond to earlier V corrections. Four late layers, one source pairing, one rank, one seed, and about 2K context do not establish general long-context gains or validate layer pairing as optimal.
- Full teacher and student backing caches and full student prefill remain. This is not a storage compression or prefill-bypass result; timings are for eager linear validation, not optimized SpecExtend.

[Method and reproduction](../README.md). Raw results: offline_test.jsonl, runtime_test.jsonl.
