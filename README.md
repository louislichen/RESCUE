# RESCUE
Official implementation of RESCUE

## Paper writing and session handoff

The current research architecture, evidence boundaries, and source reports are
summarized in [docs/PAPER_HANDOFF.md](docs/PAPER_HANDOFF.md).
For a new VSCode session, use [docs/NEW_SESSION_PROMPT.md](docs/NEW_SESSION_PROMPT.md).

## Current experimental implementation

The integrated three-module execution system is in [rescue](rescue/README.md),
with [system design](docs/SYSTEM_DESIGN.md) and
[implementation validation](exp/system_validation/README.md).
The [matched baseline comparison](exp/system_validation/BASELINE_COMPARISON.md)
currently remains slower than native SD and Qwen2 SpecExtend-retrieval ports;
the latter are not an upstream full-system SpecExtend reproduction.
The [critical-path audit and optimization](exp/system_validation/SYSTEM_OPTIMIZATION.md)
records a further execution speedup with unchanged draft trajectories and
the remaining method-level overhead and length-policy limitations.
The subsequent [execution reuse and routing report](exp/system_validation/EXECUTION_REUSE.md)
records deferred attention probes, gate-affine reuse, batched working-KV reads
and train-only injection-route selection with no-injection controls.
The [long-context optimization](exp/long_system/README.md) adds efficient
attention, bounded hierarchical selection, adaptive refresh, joint-rollout
training and injection-conditioned length. Its matched SpecExtend comparison
has a separate protocol and [report](exp/long_system/REPORT.md); training loss
or functional checks alone do not establish a speed or SOTA result.
The subsequent [module-objective study](exp/module_objectives/REPORT.md)
separates draft fidelity, reference costs, wasted proposals, verification work
and schedule deadlines. It is a fresh-development pilot; the earlier held-out
matrix remains paused, and none of its local proxy gains establishes full-method success.
The following [module-metric optimization](exp/module_opt/REPORT.md) implements
query-conditioned V directions, lightweight post-injection risk and causal
verified feedback. Direction-only injection has a small draft-quality signal;
the ratio length policy reduces waste but needs more verification. New selection
anchors and read-conditioned direction did not improve the pilot. No stable
net acceleration is established; 180 confirmation outputs agree with Target.
The next [internal module-architecture study](exp/module_arch/REPORT.md) adds
compact native-residual V mapping, prefix-aware replay with recollection, and
depth-two draft-length decisions. The study is in progress; its frozen plans
and partial records are not final performance evidence.

Historical SpecExtend/BridgeKV runners and reports remain under
[`exp/specextend_bridgekv`](exp/specextend_bridgekv/README.md).
