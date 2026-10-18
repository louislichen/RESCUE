# Scheduling audit

Read-only CPU accounting for the saved Qwen horizon, V-injection pilot, and
frozen Vicuna CSGB monitor. No models, CUDA, training, or archive extraction.

From the repository root:

```bash
python exp/scheduling_audit/analyze.py
```

The script regenerates [audit.json](audit.json), verifies the Qwen final source
inventory, and checks that CSGB document/stream/repeat aggregation reproduces
the archived primary TPS point estimate. It preserves different experiments'
measurement scopes. Subtracting recorded time is fixed-trace arithmetic, not
an optimized decoding benchmark or a new confidence interval.

Architecture, implementation findings, priorities, and limits are in
[the scheduling review](../../docs/SCHEDULING_AUDIT.md).
