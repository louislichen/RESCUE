# Attention Transfer Experiment

## Goal

This experiment tests whether a large teacher model can directly guide a smaller
student model's attention focus by transferring attention distributions over
token positions.

The first implementation intentionally does not align hidden states, Q/K/V
vectors, or attention heads. Teacher attention is treated as a distribution over
text spans, then projected onto the student's token positions.

## Models

- Teacher: `/mnt/8T/cl/models/Qwen2-7B-Instruct`
- Student: `/mnt/8T/cl/models/Qwen2-1.5B-Instruct`

Both models use their own tokenizers. Teacher token positions are mapped to
student token positions through character-span overlap.

## Prompt

For each HotpotQA sample, documents are formatted as:

```text
[Doc 1]
{title_1}
{sentences_1}

[Doc 2]
{title_2}
{sentences_2}

Question: {question}
Answer:
```

The default `doc_mode=all` keeps all context documents, including distractors.

If HotpotQA is not locally available, `--dataset_name synthetic` runs a tiny
built-in sample only for pipeline sanity checks. It is not a substitute for the
real HotpotQA result.

For online streaming without a local copy, use:

```text
--dataset_name hotpotqa/hotpot_qa --dataset_config distractor
--dataset_split validation --streaming true
```

Internally this uses the Hub parquet file through an `hf://datasets/...`
streaming URL because direct repo-id loading failed in the current environment.

## Attention Mapping

For a selected teacher layer, the script extracts the last query row:

```text
A_teacher[layer] shape: [heads, q_len, k_len]
p_teacher_token = mean_heads(A_teacher[layer, :, -1, :])
```

Then it projects `p_teacher_token` to student token positions:

```text
p_student[j] = sum_i p_teacher[i] * overlap_chars(teacher_i, student_j) / len(teacher_i)
```

The projected distribution is normalized and used as the teacher attention
target in the student token coordinate system.

## Intervention

The default intervention is `--steering_mode prob_mix`. For selected student
layers and only the last query row:

```text
alpha = lambda / (1 + lambda)
student_attention_probs[:, :, -1, :] =
  (1 - alpha) * student_attention_probs[:, :, -1, :] + alpha * p_student
```

No top-k filtering is used in the first experiment. All previous/current tokens
visible to the causal attention row receive a teacher-derived weight.

The older pre-softmax prior form is still available as `--steering_mode
log_prior`:

```text
student_attention_scores[:, :, -1, :] += lambda * log(p_student + eps)
```

## Logged Attention Metrics

For each steered decode step, the script can record how close the student
attention row is to the projected teacher prior before and after intervention:

```text
attn_before[layer].jsd / pearson
attn_after[layer].jsd / pearson
```

This verifies whether the intervention is actually pulling student attention
toward the teacher distribution, not only changing the generated answer.

## First Sweep

Recommended small sanity run:

```text
max_samples = 1 or 2
max_L = 1024
lambda = 1.0
layer_scope = last_layer
max_new_tokens = 8
```

After the chain is verified, use:

```text
lambda in {0.5, 1.0, 2.0, 4.0}
layer_scope in {last_layer, last_4_layers}
```

## Speculative Acceptance

`run_hotpot_spec_acceptance.py` evaluates whether attention transfer improves
the draft model used by speculative decoding. It compares:

```text
baseline: Qwen2-1.5B normal draft tokens
steered:  Qwen2-1.5B draft tokens with attention transfer
verifier: Qwen2-7B greedy verifier
```

The token-level acceptance metrics are:

```text
accept_rate = accepted_tokens / proposed_tokens
full_accept_rate = fully accepted draft blocks / draft blocks
mean_accept_len = mean accepted prefix length per draft block
first_token_match_rate = verifier match rate for the first draft token
teacher_call_reduction_proxy = generated_tokens / verifier_calls
```

The default prior mode for this system experiment is `prompt_once`: collect the
teacher prompt attention once, then reuse it during answer generation. This is a
more realistic serving proxy than recomputing teacher attention before every
draft token. `--prior_mode stepwise` remains available as an oracle-style check.

## Student Quality

`run_hotpot_quality_eval.py` uses the same frozen sample file and compares pure
student greedy answers:

```text
baseline: Qwen2-1.5B normal greedy
steered:  Qwen2-1.5B greedy with attention transfer
```

This is separate from speculative decoding because speculative output is
teacher-verified and corrected, so it should not be used as the pure small-model
quality metric.

## 300-Sample HotpotQA Results

Frozen sample file:

```text
exp/attn_transfer/data/hotpot_stream_300_maxL2048.jsonl
```

Common settings:

```text
max_samples = 300
max_new_tokens = 32
draft_k = 4 for speculative acceptance
steering_mode = prob_mix
prior_mode = prompt_once
dtype = bfloat16
device = cuda:0
```

### Speculative Acceptance

Baseline, normal Qwen2-1.5B draft without attention steering:

```text
accept_rate = 0.6046
full_accept_rate = 0.4831
mean_accept_len = 2.3331
first_token_match_rate = 0.7278
teacher_call_reduction_proxy = 3.2887
```

| Config | Accept rate | Full accept | Mean accept len | First-token match | Call reduction proxy | Clean EM | Clean F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lambda=0.5, last_layer | 0.6080 | 0.4836 | 2.3469 | 0.7331 | 3.3028 | 0.1467 | 0.3060 |
| lambda=0.5, last_4_layers | 0.6086 | 0.4841 | 2.3508 | 0.7358 | 3.3079 | 0.1467 | 0.3057 |
| lambda=1.0, last_layer | 0.6095 | 0.4844 | 2.3541 | 0.7337 | 3.3097 | 0.1467 | 0.3060 |
| lambda=1.0, last_4_layers | 0.6085 | 0.4828 | 2.3545 | 0.7416 | 3.3128 | 0.1467 | 0.3057 |
| lambda=2.0, last_layer | 0.6075 | 0.4835 | 2.3473 | 0.7334 | 3.3027 | 0.1467 | 0.3060 |
| lambda=2.0, last_4_layers | 0.6024 | 0.4750 | 2.3332 | 0.7381 | 3.2925 | 0.1467 | 0.3058 |

The best acceptance configuration in this sweep is `lambda=1.0, last_layer`.
Multi-layer steering often raises first-token match, but it does not reliably
improve full-block acceptance.

### Pure Student Quality

Baseline, normal Qwen2-1.5B greedy without attention steering:

```text
clean_em = 0.3167
clean_f1 = 0.4872
raw_em = 0.1300
raw_f1 = 0.2871
```

| Config | Clean EM | Clean F1 | Raw EM | Raw F1 |
| --- | ---: | ---: | ---: | ---: |
| lambda=0.5, last_layer | 0.3133 | 0.4885 | 0.1300 | 0.2902 |
| lambda=0.5, last_4_layers | 0.3400 | 0.5038 | 0.1533 | 0.3040 |
| lambda=1.0, last_layer | 0.3100 | 0.4842 | 0.1367 | 0.2928 |
| lambda=1.0, last_4_layers | 0.3333 | 0.5036 | 0.1667 | 0.3163 |
| lambda=2.0, last_layer | 0.3133 | 0.4899 | 0.1367 | 0.2976 |
| lambda=2.0, last_4_layers | 0.3500 | 0.5043 | 0.1633 | 0.3049 |

Quality gains mainly come from steering the last four layers. The best clean
EM/F1 configuration is `lambda=2.0, last_4_layers`, while the best raw EM/F1 is
`lambda=1.0, last_4_layers`.

## Deeper / Stronger Steering Addendum

The implementation now accepts generic `last_N_layers` scopes, for example
`last_8_layers` and `last_12_layers`. A post-processing edge case was also
fixed: if strong steering produces an empty answer, `extract_final_answer`
returns an empty string and the sample receives zero score instead of crashing
the run.

### Quality Probes

| Samples | Config | Baseline clean EM/F1 | Steered clean EM/F1 | Baseline raw EM/F1 | Steered raw EM/F1 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 20 | lambda=4.0, last_12_layers | 0.4000 / 0.5065 | 0.3500 / 0.5144 | 0.1000 / 0.2364 | 0.0500 / 0.1779 |
| 50 | lambda=4.0, last_4_layers | 0.3200 / 0.4788 | 0.3400 / 0.4631 | 0.1000 / 0.2565 | 0.1200 / 0.2364 |
| 50 | lambda=2.0, last_8_layers | 0.3200 / 0.4788 | 0.3400 / 0.4874 | 0.1000 / 0.2565 | 0.0800 / 0.1839 |
| 50 | lambda=4.0, last_8_layers | 0.3200 / 0.4788 | 0.3600 / 0.5101 | 0.1000 / 0.2565 | 0.0400 / 0.1489 |
| 50 | lambda=2.0, last_12_layers | 0.3200 / 0.4788 | 0.3800 / 0.5352 | 0.1000 / 0.2565 | 0.0600 / 0.2038 |

### 300-Sample Quality Confirmation

| Config | Clean EM | Clean F1 | Raw EM | Raw F1 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.3167 | 0.4872 | 0.1300 | 0.2871 |
| lambda=2.0, last_12_layers | 0.3667 | 0.5216 | 0.1033 | 0.2295 |
| lambda=4.0, last_8_layers | 0.3433 | 0.4987 | 0.0600 | 0.1800 |

The best clean QA setting is now `lambda=2.0, last_12_layers`. Increasing to
`lambda=4.0` does not improve clean quality further and substantially hurts raw
answer quality.

### Speculative Acceptance Probes

| Samples | Config | Mode | Accept rate | Full accept | Mean accept len | First-token match | Call reduction proxy | Clean EM/F1 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | lambda=2.0, last_12_layers | baseline | 0.6336 | 0.5368 | 2.4252 | 0.7582 | 3.3672 | 0.2200 / 0.3717 |
| 50 | lambda=2.0, last_12_layers | steered | 0.5606 | 0.4301 | 2.1608 | 0.7056 | 3.1174 | 0.2200 / 0.3716 |
| 50 | lambda=4.0, last_8_layers | baseline | 0.6336 | 0.5368 | 2.4252 | 0.7582 | 3.3672 | 0.2200 / 0.3717 |
| 50 | lambda=4.0, last_8_layers | steered | 0.5870 | 0.4628 | 2.2643 | 0.7521 | 3.2164 | 0.2200 / 0.3547 |

Deeper/stronger steering should not be expanded to 300-sample speculative
acceptance under the current mechanism: it improves the student's cleaned QA
score but shifts draft tokens away from the verifier's greedy trajectory.
