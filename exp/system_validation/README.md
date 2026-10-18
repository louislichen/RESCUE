# Full-system implementation validation

Execution code lives in [rescue](../../rescue/README.md); design and ownership
are described in [SYSTEM_DESIGN.md](../../docs/SYSTEM_DESIGN.md).
The completed comparison and its evidence limits are in [REPORT.md](REPORT.md).
The separate native-SD and Qwen2 SpecExtend-retrieval comparison is in
[BASELINE_COMPARISON.md](BASELINE_COMPARISON.md). SpecExtend there refers to a
retrieval-method port, not its upstream full tree/Flash/hybrid engine.
The subsequent [critical-path optimization](SYSTEM_OPTIMIZATION.md) includes
separate stage diagnostics, training-only calibration and a new 128-request
matched matrix. Pure execution improves speed with unchanged trajectories;
the complete method still trails the equally optimized baselines.
The next [execution reuse and injection-route comparison](EXECUTION_REUSE.md)
tests deferred probes, immutable gate-affine memoization, packed/incremental
working KV and training-selected static injection budgets. It retains the
same mapper/risk weights and includes explicit no-injection controls.

- `out_smoke_v1_20261017`: first two training documents, one timing repeat,
  32-token maximum, one calibration document. This is an earlier code snapshot.
- `out_validation_v1_20261017`: eight existing validation documents, two
  timing repeats, 64-token maximum, two separate training calibration documents.

Each run keeps its own frozen sources, protocol, configurations, references,
cost samples and raw request records. These are existing research splits and
single-model-pair system measurements, not a new independent paper confirmation
set. The TCP mode is an actual separate CPU length service on loopback; it does
not distribute the Target and Draft models to separate machines.

The new coverage selector is now integrated into the model execution path.
The system comparison changes selection, refresh cadence and costs together;
it cannot isolate coverage quality or validate it using older read-preservation
results. Original synthetic coverage timings keep their original scope.
