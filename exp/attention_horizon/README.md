# Attention-guided selection and speculation horizon

Three preregistered mechanism groups: A) native-KV read-preserving joint
selection, B) target-referenced attention risk and adaptive draft length,
C) joint workset/horizon choice versus sequential composition.

Completed results: [conclusions](out/conclusions.md), [full report](out/report.md),
[summary](out/summary.json), [integrity checks](out/completeness.json),
[figure PNG](out/mechanisms.png), [figure PDF](out/mechanisms.pdf).

## Protocol

- Frozen Qwen2-7B → 1.5B, FP32 teacher / BF16 student, H100, eager attention,
  TF32 disabled. Teacher always verifies against its complete canonical cache.
- Greedy exact speculation: accept the consecutive student tokens equal to
  teacher argmax, then use the teacher correction/bonus. These are not measured
  stochastic `min(1,p/q)` acceptance probabilities.
- 48/16/30 entirely new HotpotQA train/validation/test documents, title-disjoint
  across splits and from all previous experiments. Input length 600–2048.
- Budget 512, including sink and latest 64 positions. Maximum output 64 tokens.
  Draft lengths 1,2,4,8,12. New draft suffixes can temporarily grow the workset.
- All planning uses only a completed committed-prefix query. Teacher attention
  is reused from prefill/verification. The student obtains a full-key shadow
  attention row using the actual current last-layer Q and native backing K;
  this reads the full key cache and its cost is charged. It does not rerun the
  student model or modify its actual working attention. It is not the same as
  a full-context student forward through all layers.
- Read preservation uses last-layer **per-query-head** attention and the correct
  GQA value-head grouping, in each model's own space. Errors are normalized
  per head; raw teacher/student values are never compared directly.
- Worksets start from teacher Top-k / previous memberships and use a bounded
  number of vectorized swaps. No inference-time gradient fitting or full DP.
- Training logs maximal branches and valid first-rejection prefix labels.
  Accepted EOS is a separate competing terminal event; all-continuation tails
  are right-censored. Future attention/checks are labels only, never features.
- A small boundary predictor chooses a cap from current/past features. An
  additional attention-only online control revises the cap after each draft
  token, using only student observations already available. Teacher references
  remain anchored to the last committed query; their age is an explicit input.
  This trajectory signal is not current-query teacher/student disagreement.
- A selection-only and B length controls have no V injection. C also reruns the
  best validation-chosen complete configurations with the frozen previous V
  mapper/gate, without retraining or test-set tuning.
- Report stage timing, acceptance length, proposed tokens, verification count,
  output equality, decode and prefill-inclusive TPS; document-paired bootstrap.

## Groups and decisions

A: teacher/student Top-k, 75/25 peak union, teacher-only read preservation,
joint normalized read preservation (student weight 0.25 or 1; validation may
choose zero). Compare equal budgets with fixed k=4; test both read errors and
actual output/acceptance, not proxy alone.

B: fixed k=4, best static k chosen on validation, structural-only risk,
confidence-only risk, attention-only risk, attention+confidence risk. Predict
conditional continuation/EOS/rejection hazards with a tiny MLP, use measured
cycle costs, and select maximum expected committed tokens per second.
Predictor checkpoints/hyperparameters/cost model and static k are validation
chosen; evaluation is on new test documents only.
Additional practical controls disable unnecessary student probes/captures for
fixed, historical-hazard and confidence-only policies. The historical policy
uses training hazards and an EMA updated after verification, with its update
rate selected from 0/0.1/0.3/0.6 on validation. Confidence-only uses exactly the
same trained predictor but constructs only its actual structural/logit inputs.

Predictor: normalized selected features → Linear(32) → SiLU → Linear(32) →
SiLU → Linear(36), reshaped into 12 positions × 3 event logits. Masked
at-risk cross entropy retains all observed continuation/terminal hazards and
censors unseen tails. Documents have equal total base weight, divided across
their blocks and causal frames; likelihood is normalized over the resulting
weighted at-risk positions (not separately by the observed event length).
AdamW lr 0.002, weight decay 0.01, gradient clip 1, 120 epochs; select epoch
and temperature in {0.5,0.75,1,1.5,2} on validation. Seeds 42/43/44 are all
saved; choose one on validation. Teacher/student model weights remain frozen.

C: teacher/best-joint workset × fixed/adaptive horizon, sequential selection
then conditional horizon, and simultaneous choice over a bounded pool of
teacher/read-teacher/read-joint worksets and k. Joint planning pays all candidate
construction/scoring costs; the sequential route only pays its one-workset cost.
The joint design does not claim a global optimum over arbitrary subsets.

Long-context/multi-seed extension is conditional on usable mechanism signals;
if a proxy or joint policy fails, report the failure and retain the strongest
baseline rather than declaring the requested hypothesis established.

## Run

```bash
/home/wanghl/miniconda3/envs/rescue/bin/python exp/attention_horizon/prepare_data.py
/home/wanghl/miniconda3/envs/rescue/bin/python -u exp/attention_horizon/run.py --phase all
/home/wanghl/miniconda3/envs/rescue/bin/python exp/attention_horizon/report.py
```

Individual resumable phases: `verify`, `offline`, `collect`, `fit`, `validate`,
`test`. Collection profiles additional teacher block sizes on training only;
its timing is diagnostic and is never reported as deployment throughput.
Runtime methods run in a document-specific shuffled order. The frozen previous
mapper is an integration check; it is not retrained on the new training split.
Models, tensor checkpoints and run logs are local ignored artifacts. Source,
protocol, split manifests, per-document JSONL and reports are reviewable.
