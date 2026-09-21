# RESCUE Validated Progress Summary — through 2026-09-21

## 1. Research objective

RESCUE studies whether Teacher-side information can be transferred to a smaller Student/draft model to improve answer quality and fixed-K speculative draft quality, with emphasis on attention/guidance mechanisms, selective repair, and later adaptive K.

This archive covers the current Attention Adapter / Teacher→Student guidance mainline.

---

## 2. Correctness recovery and harness freeze

A critical-path audit invalidated earlier Adapter/Null conclusions because the old steering path did not correctly honor the Transformers 4.57.1 cache interface and several experimental-integrity invariants were missing.

Major issues identified and closed included:

- `past_key_value` vs `past_key_values` cache compatibility;
- empty-hook transparency;
- selected-layer KV-cache lifecycle;
- multi-token task-gradient connectivity;
- Student/Teacher/checkpoint/training-config identity;
- prior resume identity;
- stale completion markers and atomic writes;
- attempted/valid/excluded denominator accounting;
- bf16/float16 numerical fail-closed behavior;
- K=4 incremental correctness.

Final frozen harness:

- **HARNESS_FREEZE_V3**
- manifest SHA256: `0e569ea84d36ae91d7ad73dd7299b60c9dcab1ff0abb14e270f447449ff43788`
- independent verdict: **ACCEPT_HARNESS_FREEZE_V3**
- final independent review artifact SHA256:
  `330615041a4a56b86710f59d52d1cbcc0fb23ae2ba29e45d5a9ff5686fc3724a`
- real-GPU PRE_EXPERIMENT_CORRECTNESS_GATE: **10/10 PASS**

Earlier contaminated Adapter/Null QA, K4, sequence-loss and full-window conclusions remain **INVALIDATED** and must not be revived.

---

## 3. Valid four-arm development rerun

Task:
`RESCUE-MINIMAL-FOUR-ARM-VALID-RERUN-001`

Frozen development split: 128 rows.

Arms:

- A: Base
- B: Prefill-only Teacher
- C: Full-window Teacher
- D: Full-window Null

### QA

| Arm | EM | F1 |
|---|---:|---:|
| A | 0.46875 | 0.594971 |
| B | 0.46875 | 0.594971 |
| C | 0.46875 | 0.594971 |
| D | 0.46875 | 0.594971 |

B and C were token-identical to A on 128/128 samples.

### Fixed K=4

| Arm | Proposed | Accepted | Acceptance ratio |
|---|---:|---:|---:|
| A | 575 | 420 | 0.730435 |
| B | 575 | 420 | 0.730435 |
| C | 575 | 420 | 0.730435 |
| D | 577 | 420 | 0.727903 |

All arms satisfied 128/128 Target-only correctness under the frozen incremental verifier.

Primary case: **CASE 3**.

Interpretation: current raw Teacher attention guidance changes internal attention/logits but provides no measurable QA or K4 draft benefit under the frozen configuration.

---

## 4. Decision relevance

Task:
`RESCUE-TEACHER-PRIOR-DECISION-RELEVANCE-DIAGNOSTIC-001`

496 valid decision positions:
- S=T: 429
- S≠T: 67

At the 67 positions where the Student and Target disagree:

- Teacher Δ target margin: **-0.0135**
- Teacher Δ log p(Target token): **-0.0157**
- cosine with Target correction direction: approximately **+0.0003**
- Target-directed flips: **0/67**

Teacher did not clearly outperform Null.

Interpretation: zero flips are primarily an information/direction problem, not merely insufficient scale.

---

## 5. Local controllability upper bound

Task:
`RESCUE-GUIDANCE-REPRESENTATION-MECHANISM-SPRINT-001`

Using a same-budget local Student Oracle at the exact frozen injection site:

- actual Oracle ΔM: **+1.73** [1.54, 1.91]
- median deficit closure: **1.011**
- Target-directed flips: **42/67 = 62.7%**
- Teacher efficiency: median about **-0.65%**
- Teacher vs Null: not meaningfully distinguishable
- 20 predetermined strata: **NO_CONDITIONAL_SIGNAL**

Interpretation: the Student injection site has substantial local correction capacity; current Teacher guidance direction does not exploit it.

---

## 6. R1 value-norm candidate

Candidate:
`TEACHER_ATTN_VALUE_NORM`

Task:
`RESCUE-R1-VALUE-NORM-DECISION-ALIGNMENT-PILOT-001`

Single variable:
`attention_j → attention_j × ||V_j||`

Result: **R1-FAIL**

- R1 mean ΔM = R0 mean ΔM = **-0.015392**
- R1 perturbation magnitude was larger than R0 at 100% of positions
- Target-directed flips: **0/67**
- Oracle efficiency: still slightly negative
- preregistered falsification criterion satisfied

Interpretation: simple value-norm reweighting changes magnitude/semantic grounding but does not recover decision-relevant direction.

---

## 7. Teacher decision attribution

Task family:
`RESCUE-DECISION-AWARE-TEACHER-SIGNAL-SPRINT-001`

Latest local result:

### TD-A — Teacher-side signal exists

- `V_TD_teacher`: **2.770** [2.472, 3.074]
- actual Teacher-side ΔM: **+2.823** [2.523, 3.121]
- positive on **67/67** disagreement positions

### TD-B — cross-head transport is lossy

- token-space projection for the current tokenizer pair: effectively identity
- frozen softmax head mixing removes approximately **82.7%** of signed L1 mass
- median cancellation approximately 86.2%

### TD-C — current Student exploitation is tiny

- Student-side efficiency about **+0.0078** on average, median about **-0.0019**
- Target-directed flips: **2/67**
- local Student Oracle reference: **42/67**

Interpretation: the Teacher contains a strong decision-aware signal, but the current transport/mixing pipeline discards most of it before the Student can exploit it.

This is a development mechanism finding, not a deployable method result.

---

## 8. Current mechanistic picture

```text
Teacher has decision-aware information                    YES
Raw Teacher attention is decision-relevant               NO
Simple attention×||V|| reweighting works                  NO
Student attention injection site is locally controllable YES
Token projection is dominant loss on current pair        NO
Frozen cross-head convex mixing is strongly lossy        YES, diagnostic evidence
A new sign-preserving operator is already validated      NO
```

---

## 9. Current pause

Further experimental work is paused by user decision.

The next operator idea discussed was a sign-preserving / head-specialized transport diagnostic, but it has **not been executed** and must not be described as a completed method.

No additional Adapter training, adaptive-K, Repair or cache-system experiment is authorized by this archive.
