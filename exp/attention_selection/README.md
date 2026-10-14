# Native-KV attention selection experiments

Implementation of the stage-two [E1–E9 checklist](../specextend_bridgekv/design/attention_selection_diagnostics.md). The teacher and student stay frozen. Historical student K/V are native, retain their original positions, and are gathered only for reading. There is no mapper or teacher-V injection.

## Protocol

- Qwen2-7B-Instruct → Qwen2-1.5B-Instruct, one H100, BF16 eager attention.
- Fresh HotpotQA training examples: 24 train, 12 validation, 30 test. Exclude document titles from the earlier 300-example experiments and keep all new splits title-disjoint.
- Two working budgets: 256 and 512 tokens per student layer, including the sink and 64 recent positions. Query states 0, 2, 4, 6 on teacher-generated trajectories; actual decoding generates up to 32 tokens with draft k=4.
- First aggregate heads within each layer. Global teacher/student rules use the final layer. Train remote-history layer correspondence on training documents only.
- Union/core quotas, attention-bias layer and strength, and selector checkpoints are chosen on validation data. The selector is trained on measured output-KL changes under equal-budget single-token swaps, then evaluated on joint eight-token replacements.
- Current-query, common-full-history attention is an offline diagnostic. Runtime only reads completed teacher prefill/verification signals and actually observed student positions. Complete attention acquisition and caching costs are timed.
- The complete draft backing cache is retained. These experiments measure working-cache retrieval and cannot establish removal of all historical KV storage.

## Files

- `prepare_data.py`: deterministic fresh, disjoint data and manifest.
- `attention.py`: native-KV gather and single-layer attention-score intervention, with absolute RoPE positions preserved.
- `selection.py`: recent/random/teacher/student/chunk/token/union/difference rules and utility predictor.
- `run.py`: collection, layer matching, swap labels, selector fitting, validation and held-out offline experiments.
- `runtime.py`: real greedy speculative decoding, accepted-token/cache rollback, committed-prefix retrieval and stage timing.
- `verify.py`: native equality, sparse/dense FP32 equivalence, exact-budget, suffix and rollback integration audits.
- `supplement.py`: equal-budget teacher/student-exclusive peak ablations and utility correlations.
- `numerical_audit.py`: batch-shape and causal-visibility diagnostics; separate FP32-teacher runtime recheck.
- `analyze.py`: completeness audit, document-paired bootstrap, report and PNG/PDF scientific figures.
- `data_fresh/`: fixed samples, token IDs, evidence-document positions and data provenance.
- `out_stage2/`: raw JSON/JSONL, selected configurations, checks, report and figures. Local tensor checkpoints are ignored by Git.

## Run

```bash
/home/wanghl/miniconda3/envs/rescue/bin/python exp/attention_selection/prepare_data.py --output exp/attention_selection/data_fresh
/home/wanghl/miniconda3/envs/rescue/bin/python exp/attention_selection/run.py --phase smoke
/home/wanghl/miniconda3/envs/rescue/bin/python exp/attention_selection/run.py --phase all
/home/wanghl/miniconda3/envs/rescue/bin/python exp/attention_selection/supplement.py
/home/wanghl/miniconda3/envs/rescue/bin/python exp/attention_selection/numerical_audit.py --fp32-runtime
/home/wanghl/miniconda3/envs/idn/bin/python exp/attention_selection/analyze.py
```

Run GPU commands sequentially to avoid contaminating timing. Each phase can resume from existing complete records. Use a new `--output` and `--cache` for a changed model/data/protocol; experiment tensors default to `/mnt/8T/cl/rescue_experiment_cache/attention_selection_stage2`. The FP32 runtime uses the original frozen validation choices, regenerates its own teacher reference and remains separate from BF16 results.

All E1–E9 experiment groups and supplemental audits have completed. Main findings are in [`out_stage2/conclusions.md`](out_stage2/conclusions.md); full protocol, tables and limitations are in [`out_stage2/report.md`](out_stage2/report.md), with record checks in [`out_stage2/completeness.json`](out_stage2/completeness.json).
