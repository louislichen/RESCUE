# RESCUE session entry

For paper writing, research discussion, or continuation of the attention-guided
speculative decoding work, first read [docs/PAPER_HANDOFF.md](docs/PAPER_HANDOFF.md).
It records the user's latest architecture, implemented experiments, evidence
limits, and paths to the source reports. A ready-to-use writing prompt is in
[docs/NEW_SESSION_PROMPT.md](docs/NEW_SESSION_PROMPT.md).

Keep these distinctions explicit:

- Module numbers describe the paper: 1 = context selection, 2 = selective
  content injection, 3 = draft-length control. Experiment groups A/B/C instead
  mean selection / length / their joint decision.
- The original exp/attention_coverage benchmark is synthetic. Coverage is now
  integrated in rescue with a limited existing-validation-set system run;
  that comparison also changes refresh cadence and costs, so it does not
  isolate coverage quality or establish the paper hypothesis.
- Read-preservation selection is the completed older experiment, with a
  different objective. Its results cannot validate the new coverage selector.
- Performance-supervised V injection has a positive acceptance signal against
  its matched selection baseline. Precise token-content correspondence and the
  independent value of learned gates remain unproved.
- Current length/joint policies have not shown net acceleration; do not write
  the complete paper hypothesis as an established empirical conclusion.
- The newer archived CSGB 4K pilot controls tree node budgets at fixed maximum
  depth 10, with Native V and no mapper. It increases accepted draft tokens per
  round against fixed8 but lowers whole-request TPS on eight training-monitor
  documents. It does not validate dynamic maximum depth, mapper integration,
  attention-feature attribution, or held-out generalization. See the latest
  sync section in docs/PAPER_HANDOFF.md.
- The integrated Qwen system has positive speed signals against the prior
  pipeline and exact saved Target outputs on 80 requests, but remains slower
  than Target AR. Its TCP service offloads only length decisions; Target and
  Draft model execution has not been split across edge/cloud machines.
- A subsequent matched comparison remains slower than native full-context SD
  and Qwen2 ports of SpecExtend retrieval (512/1024-token bounds). Those ports
  do not reproduce upstream SpecExtend's Vicuna tree/Flash/hybrid engine.
  See exp/system_validation/BASELINE_COMPARISON.md; the prior +28.93% is only
  against the older intervened pipeline, not either of these baselines.
- A later critical-path run improves decode/whole-request TPS by 4.94%/3.38%
  versus legacy execution with all 16 trajectories unchanged. Common GQA/cache
  optimizations also apply to native/retrieval controls; RESCUE still trails
  both. Fixed-k4 and two-parameter post-injection hazard calibration do not
  establish additional dynamic-length speedup. See
  exp/system_validation/SYSTEM_OPTIMIZATION.md.
- The subsequent reuse run enables deferred probes, committed-position static
  gate-affine memoization and packed working KV. It improves paired whole-request
  TPS by 0.99% with 16/16 risk trajectories unchanged. Train-selected one-layer,
  64-token injection improves another fixed-k4 comparison by 2.91%, but still
  trails native SD by 6.15% and has no established gain over disabling injection.
  Zero-allowed training selects no injection. This is execution reuse and static
  route calibration, not joint-rollout mapper/length retraining or a complete
  method success. See exp/system_validation/EXECUTION_REUSE.md.
- The newer GovReport long-context implementation adds efficient attention,
  hierarchical bounded-support coverage, adaptive refresh, last-layer rollout
  mapper replay and injection-conditioned length training. Development has
  114+84 exact requests; held-out testing is paused at the user's request with
  71/320 exact saved outputs. Development-selected nonzero injection still
  trails the faster Qwen SpecExtend retrieval control; zero-allowed selection
  disables injection. The complete tree-flow Qwen port is not the original
  Vicuna/FP16 reproduction. No final held-out speed conclusion or SOTA result
  exists. Do not resume experiments without a new user instruction; see
  exp/long_system/REPORT.md and out_comparison_v4_20261017/PAUSED.json there.
- The user's subsequent goal-oriented request authorizes a separate module
  study, not resuming that paused test. The completed fresh-development pilot
  has 132 exact SD outputs, 1680 common-prefix quality rows and 45 CPU checks.
  M1 reduces working references and shows local quality signals, but the
  rolling system remains slower than full native reads. M2 improves KL without
  greedy correction; the new margin-preservation training does not fix it.
  M3 reduces some waste but does not yield net acceleration. Early publication
  improves async TPS by only 0.18% and still trails local serial scheduling.
  Target-aware V-oracle corrections on training-monitor queries are diagnostic,
  not learned performance or a global capacity bound. See
  exp/module_objectives/REPORT.md and docs/MODULE_OBJECTIVES.md.
- The subsequent module-metric optimization is complete on three new development
  confirmation documents, 8K/16K inputs, 128-token output limits. Selection
  anchors failed train-side rolling quality and were not promoted. Query-conditioned
  V directions improve common-prefix matches from 105/144 without injection to
  109/144 and reduce waste against the old mapper, but have no stable net speedup.
  Adding actual Student Q/K read conditioning did not improve the pilot. Lightweight
  attention plus causal feedback with ratio scoring reduces waste from fixed4's
  930 to 876 while increasing verification rounds from 500 to 514; it remains
  slower. Surplus scoring over-drafts. The two stages reuse the same three
  confirmation documents, not independent replications; all 180 confirmation
  outputs match Target and 52 CPU checks pass. Old defaults remain, no new
  mapper-plus-length joint success is established, and the 71/320 test remains
  paused. See exp/module_opt/REPORT.md and its frozen sources, not earlier pilots
  with different cohorts/output limits, for these numbers.
- The next internal M2/M3 architecture study is in progress at exp/module_arch.
  It adds compact native-residual V mapping, all-final-layer prefix-aware replay,
  recollection, and a causal depth-two decision between draft lengths 2 and 4.
  Primary compact training retained zero correction, so its speed cannot validate
  content transfer. A predefined supplementary study tests open-gate initialization
  and full-span balanced fit sampling; a separate train-fast length endpoint is
  also retained. New confirmation is 604/613/614, not the paused test. Consult
  exp/module_arch/REPORT.md and each stage's COMPLETE/audit status before citing
  results. Pending runs and training loss are not established performance gains.

Use the original per-experiment protocols and reports for numbers. Keep
different datasets, budgets, numerical precisions, and output limits separate.
The latest explicit user request determines whether to write, implement, run
experiments, or publish changes; a handoff alone is not a new experiment request.
