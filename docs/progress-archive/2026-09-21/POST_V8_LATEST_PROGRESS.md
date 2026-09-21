# RESCUE Post-V8 Latest Progress Archive

> Status: archival summary of local execution returns received after V8.  
> This file does **not** replace the canonical project-control files because the corresponding V10 control bundle was not available to this web session.

## 1. R1 value-norm pilot

Task: `RESCUE-R1-VALUE-NORM-DECISION-ALIGNMENT-PILOT-001`

Verdict: **R1-FAIL**

- Population: 67 Student–Target disagreement positions.
- R0 mean ΔM: -0.015392.
- R1 mean ΔM: -0.015392.
- Null mean ΔM: -0.020522.
- R1 − R0 paired ΔM: +0.000000 with CI spanning zero.
- R1 perturbation magnitude was larger than R0 at every position, but Target-directed flips remained 0/67.
- Oracle efficiency remained near zero / slightly negative.
- Preregistered falsification criterion was satisfied; R1 was retired.
- No QA/K4 rerun, retraining, alpha sweep, layer sweep, or alternative norm sweep was performed.

Interpretation: simple value-norm reweighting increases perturbation magnitude but does not recover decision-relevant direction.

## 2. Guidance representation mechanism sprint

Task: `RESCUE-GUIDANCE-REPRESENTATION-MECHANISM-SPRINT-001`

Key verdict: **CONTROLLABILITY-A**

On the 67 disagreement positions:

- Same-budget local Student Oracle actual ΔM: **+1.73** [1.54, 1.91].
- Median decision-deficit closure: **1.011**.
- Target-directed flips: **42/67 = 62.7%**.
- Teacher efficiency: median approximately **-0.65%**.
- Teacher and Null were not meaningfully distinguishable.
- 20 predetermined strata produced **NO_CONDITIONAL_SIGNAL**.

Interpretation: the Student attention injection site has substantial local correction capacity; the bottleneck is the guidance direction / representation, not simple scale or injection-site controllability.

## 3. Teacher decision attribution Phase C

Task family: `RESCUE-DECISION-AWARE-TEACHER-SIGNAL-SPRINT-001`

Local completion status reported: **67/67 positions completed**.

### Three-part result

**TD-A — Teacher signal present: YES**

- Teacher-side decision-aware value `V_TD_teacher`: **2.770** [2.472, 3.074].
- Actual Teacher-side ΔM: **+2.823** [2.523, 3.121].
- Positive direction: **67/67**.
- Range of actual ΔM: approximately 1.00 to 6.59.

**TD-B — Projection / transport: LOSSY**

- Token-space projection for the current tokenizer pair was effectively identity:
  - coverage 1.0
  - deviation 0.0
- Frozen cross-head softmax mixing removed approximately **82.7%** of signed L1 mass (median 86.2%, range 58.4–95.2%).
- Sign agreement with the dominant Teacher head was only approximately 0.530.

**TD-C — Student exploitation: NOT EXPLOITABLE under current frozen transport**

- Student-side efficiency: mean approximately **+0.0078**, median approximately **-0.0019**.
- No positions with efficiency > 0.5.
- Median Student-side ΔM: 0.0.
- Target-directed flips: **2/67**, versus local Student Oracle **42/67**.

Paired development diagnostics:
- TD − R0 efficiency difference: +0.01374 [0.00324, 0.02553].
- TD − R1: positive.
- TD − Null: positive.
- The differences distinguish TD from prior frozen representations, but absolute recovery remains only about 0.8% of Oracle headroom.

### Bottleneck localization

Current evidence supports:

1. Teacher contains strong decision-aware signal.
2. Token mapping itself is not the dominant loss for the current tokenizer pair.
3. Current cross-head mixing/transport strongly cancels signed decision signal.
4. Student attention injection remains locally capable, but current transport recovers almost none of the available Oracle headroom.

This does **not** yet prove a new deployable operator.

## 4. Execution validity notes

- Teacher attribution capture matched the frozen observer on checked prefixes.
- Attribution score tensors were deterministic across independent runs.
- Frozen evidence recomputation hooks matched stored Student gradients/budgets/values.
- 7B long-context memory probe reached about 68.8% of 11 GB per GPU and did not hit the preregistered resource stop.
- Frozen harness/source-lock remained unchanged.
- A custom fp32 reconstruction path showed small Teacher-logit deviations on two named positions; these were disclosed and did not alter the TD flip counts.

## 5. Latest research interpretation

The evidence chain now supports the following development interpretation:

```text
Current raw Teacher attention: not decision-relevant
R1 attention × ||V||: falsified
Student injection site: locally controllable
Teacher model: contains strong decision-aware signal
Token projection: not the dominant bottleneck on this tokenizer pair
Frozen cross-head softmax mixing: strongly lossy for signed TD signal
```

The next operator idea discussed after this result was a sign-preserving / head-specialized transport operator, but **no such method has been executed in this archive**.

## 6. Claim boundaries

Allowed:
- development mechanism diagnostics
- local controllability evidence
- R1 falsification
- Teacher decision-aware signal existence
- transport-loss diagnostics

Not allowed:
- confirmation
- final paper effect size
- speedup / deployment acceleration
- adaptive-K or Repair benefit
- claiming a new sign-preserving operator works
- reviving invalidated pre-correctness Adapter results
