# Attention Adapter Transfer

## Goal

Train a small adapter that converts projected teacher attention heads into
student-compatible attention priors, then injects them into selected Qwen2-1.5B
attention layers.

This experiment keeps teacher and student tokenizers separate. Teacher attention
is projected into the student token coordinate system through character-offset
overlap, following `exp/attn_transfer`.

## Adapter

For each selected decode state:

```text
teacher_projected: [teacher_layers, teacher_heads, student_tokens]
student_rows:      [student_layers, student_heads, student_tokens]
```

The adapter learns a teacher layer/head mixture for every selected student
layer/head, plus a per-head injection gate:

```text
prior, alpha = adapter(teacher_projected, student_rows)
A_new = (1 - alpha) * A_student + alpha * prior
```

Only the adapter is trained. Both Qwen2-7B and Qwen2-1.5B are frozen.

## First Pilot

The first run trains on a small number of prompt states and first answer tokens.
It is intended to validate the GPU path and whether the adapter can affect
student logits through attention steering before scaling to more samples.

```bash
conda run -n rescue python exp/attn_adapter_transfer/train_adapter.py \
  --samples_path exp/attn_transfer/data/hotpot_stream_300_maxL2048.jsonl \
  --output_dir exp/attn_adapter_transfer/out_train_pilot \
  --max_train_samples 4 \
  --answer_tokens_per_sample 1 \
  --max_prompt_tokens 1536 \
  --teacher_layer_scope last_4_layers \
  --student_layer_scope last_4_layers
```

Then evaluate on held-out samples:

```bash
conda run -n rescue python exp/attn_adapter_transfer/run_quality_eval_adapter.py \
  --samples_path exp/attn_transfer/data/hotpot_stream_300_maxL2048.jsonl \
  --checkpoint exp/attn_adapter_transfer/out_train_pilot/adapter.pt \
  --output_dir exp/attn_adapter_transfer/out_quality_pilot \
  --skip_samples 4 \
  --max_samples 4 \
  --max_new_tokens 16 \
  --max_prompt_tokens 1536
```

## Initial GPU Runs

Environment check outside the sandbox:

```text
torch = 2.7.1+cu128
cuda available = true
gpu = NVIDIA H100
```

### Sanity pilot

```text
output_dir = exp/attn_adapter_transfer/out_train_pilot
train samples = 4
effective steps = 3
max_alpha = 0.35
mean loss = 4.1308
```

Held-out quality on 4 samples:

```text
baseline clean EM/F1 = 0.5000 / 0.7986
adapter  clean EM/F1 = 0.5000 / 0.7986
adapter alpha mean   = 0.0417
```

### 32-sample pilot

```text
output_dir = exp/attn_adapter_transfer/out_train_pilot32
train samples = 32
effective steps = 32
max_alpha = 0.70
mean loss = 1.9278
```

Held-out quality on 8 samples:

```text
baseline clean EM/F1 = 0.3750 / 0.5333
adapter  clean EM/F1 = 0.3750 / 0.5333
adapter alpha mean   = 0.0834
text changes         = 0 / 8
```

### Strong 32-sample pilot

```text
output_dir = exp/attn_adapter_transfer/out_train_strong32
train samples = 32
answer tokens per sample = 4
effective steps = 97
max_alpha = 0.80
init_gate_logit = -0.5
mean loss = 0.7008
```

Held-out quality on 8 samples:

```text
baseline clean EM/F1 = 0.3750 / 0.5333
adapter  clean EM/F1 = 0.3750 / 0.5333
adapter alpha mean   = 0.3018
text changes         = 1 / 8
```

Current conclusion: the adapter training and GPU injection path are working, but
the first objective mostly learns a conservative prior. Stronger gates can change
generation, but the observed change is not yet a quality improvement. The next
implementation step should add cached teacher projections and a direct
teacher-trajectory/speculative-acceptance objective, rather than only first-token
gold-answer CE.

## Fitting Diagnostics

The diagnostic script `diagnose_adapter_fit.py` compares baseline student and
adapter-steered student on the same token-level target:

```text
baseline CE
adapter CE
baseline argmax target hit
adapter argmax target hit
adapter changed-argmax rate
target probability delta
```

### Gold-target strong adapter

Checkpoint:

```text
exp/attn_adapter_transfer/out_train_strong32/adapter.pt
```

Train-set diagnostic on the 32 training samples:

```text
tokens = 97
baseline CE = 0.6795
adapter CE = 0.6939
CE delta = +0.0144
baseline argmax acc = 0.8041
adapter argmax acc = 0.7835
argmax acc delta = -0.0206
changed argmax rate = 0.0309
target prob delta = -0.0032
alpha mean = 0.3018
```

Conclusion: the gold-target adapter did not fit the training target. It slightly
worsened CE and argmax accuracy even on the samples used for training.

### Teacher-greedy target adapter

Training target changed from gold answer tokens to Qwen2-7B greedy answer text
tokenized by the student tokenizer:

```text
target_mode = teacher_greedy
output_dir = exp/attn_adapter_transfer/out_train_teacher32
train samples = 32
effective token steps = 128
mean loss = 0.8810
```

Train-set diagnostic:

```text
tokens = 128
baseline CE = 0.8584
adapter CE = 0.8703
CE delta = +0.0119
baseline argmax acc = 0.7656
adapter argmax acc = 0.7969
argmax acc delta = +0.0312
changed argmax rate = 0.0312
target prob delta = +0.0008
alpha mean = 0.3017
```

Held-out diagnostic on 8 samples after the training block:

```text
tokens = 32
baseline CE = 1.2097
adapter CE = 1.2107
CE delta = +0.0011
baseline argmax acc = 0.7500
adapter argmax acc = 0.7500
argmax acc delta = 0.0000
changed argmax rate = 0.0000
target prob delta = +0.0053
alpha mean = 0.3017
```

Conclusion: teacher-greedy training is a more appropriate acceptance-oriented
target and gives a small train-set argmax improvement, but it still does not
produce a strong fit. The next useful target should directly optimize either
student QA loss over full answer tokens or speculative acceptance/block matching,
and teacher projections should be cached before scaling.

## Formal QA-Target Experiment

Goal: directly optimize small-model QA behavior, not attention similarity.

Training configuration:

```text
output_dir = exp/attn_adapter_transfer/out_train_goldqa128_e3
target_mode = gold
train samples = first 128 frozen HotpotQA samples
epochs = 3
answer tokens per sample = up to 16
effective token steps = 1527
teacher layers = last_4_layers
student layers = last_4_layers
max_alpha = 1.0
init_gate_logit = 0.0
regularization weights = 0.0
mean training loss = 0.6161
```

Train-set fitting diagnostic on the same 128 samples:

```text
tokens = 509
baseline CE = 0.6106
adapter CE = 0.6076
CE delta = -0.0029
baseline argmax acc = 0.8389
adapter argmax acc = 0.8291
argmax acc delta = -0.0098
changed argmax rate = 0.0334
target prob delta = +0.0017
alpha mean = 0.4974
```

Interpretation: the adapter slightly lowers train-token CE and target
probability, but it does not improve train-token argmax accuracy. This is not a
strong overfit; the attention-only adapter has limited control over the final
student logits.

Held-out QA evaluation:

```text
eval samples = 50
sample range = after the first 128 training samples
max_new_tokens = 32
baseline clean EM/F1 = 0.4000 / 0.5385
adapter  clean EM/F1 = 0.4200 / 0.5676
baseline raw EM/F1 = 0.1600 / 0.3272
adapter  raw EM/F1 = 0.1800 / 0.3645
alpha mean = 0.4974
```

Paired statistics:

```text
text changed = 23 / 50
clean EM: +0.0200 mean delta, 1 better, 0 worse, 49 same
clean F1: +0.0291 mean delta, 4 better, 2 worse, 44 same
raw EM: +0.0200 mean delta, 1 better, 0 worse, 49 same
raw F1: +0.0374 mean delta, 9 better, 5 worse, 36 same
```

Scientific takeaway: using QA gold-token CE as the direct objective gives a
positive held-out QA signal on 50 samples, but the effect is still small and the
train-set fitting diagnostic is weak. The result is promising enough to justify
a larger confirmatory run, but not yet strong enough to claim that this adapter
reliably improves small-model QA.
