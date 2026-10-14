# Attention-weighted coverage selector: cost prototype

This directory contains a **synthetic selector microbenchmark only**. It does
not implement an evaluated end-to-end decoding method and has no acceptance
or quality results. The previous read-preservation experiments remain separate.

Run:

```bash
/home/wanghl/miniconda3/envs/rescue/bin/python exp/attention_coverage/benchmark.py
```

H100, FP32, TF32 disabled, synthetic normalized V dimensions 512/256,
1,024 candidates, 65 compulsory anchors, 447 additional selected positions.
Both model spaces are scored independently, with equal coverage weights.
Three repetitions after warmup; all iterative choices stay on GPU.

| History | Greedy one at a time | Batch 16 | Batch 32 |
|---|---:|---:|---:|
| 2,048 | 52.3 ms | 4.1 ms | 2.2 ms |
| 16,384 | 298.2 ms | 20.5 ms | 10.9 ms |

Values sum median cosine-matrix preparation and median selection time.
Raw repetitions and environment: [summary.json](out_benchmark/summary.json).
The two similarity matrices occupy 16/128 MiB respectively; this excludes KV
caches, intermediate tensors and model weights. Attention acquisition, model
execution and cache application are excluded from all timings.

Batch selection is an approximation: candidates within the same batch are
scored against the same prior selection, so their mutual redundancy is not
recomputed. Its quality is untested. These timings do not establish SD speedup.

With history N, candidate count K, optional budget B and total feature
dimension D, preparation costs O(NKD), exact greedy selection O(BNK), and
similarity storage O(NK) per model. Batch size q reduces the number of gain
recomputations to ceil(B/q), while changing the selected set. In these runs,
repeated gain computations dominate cosine-matrix construction.
