# Per-state teacher-attention controllability diagnostic

## Question

Can a frozen Qwen2-1.5B student be made to select the frozen Qwen2-7B teacher's
next token by changing only its last-query attention? This experiment fits a
separate set of parameters to each rejected state. It measures controllability
within this intervention family, not generalization or speculative acceptance.

## State selection

- Use the existing frozen 300-sample HotpotQA validation file, at most 2048
  prompt tokens and 32 generated tokens.
- Run ordinary greedy speculative decoding with `draft_k=4` and a teacher bonus
  token after a fully accepted block.
- Keep the first rejected draft token from each question, at most one state per
  question. Both models receive exactly the same prefix token IDs.
- Use the teacher verifier's argmax as the optimization target, not the gold QA
  answer. The selected student's baseline argmax must differ from that target.
- Check identical tokenizer vocabularies and per-prompt tokenization. Their
  model output dimensions contain different padding, so teacher targets must
  also lie inside the student's output vocabulary.
- Record token IDs and the generation position in `states.jsonl` so the selected
  states can be reproduced and inspected for selection bias.
- Exclude and log states whose teacher verifier or student draft argmax changes
  when reconstructing the same prefix in BF16. Select replacement questions
  until there are 64 reproducible mismatches. This conditions the diagnostic on
  numerically stable states; the exclusions are reported, not silently removed.

## Intervention and optimization

Freeze both models. Collect the last query row from all teacher heads in its
last four layers: 112 distributions over the shared token positions. For each
of the student's 48 heads in its last four layers, learn:

1. A softmax-weighted mixture of those 112 teacher distributions.
2. A sigmoid gate between zero and one for mixing that prior with the student's
   current attention row.

No temperature transformation, auxiliary attention loss, residual adapter,
model fine-tuning, or logit modification is used. Optimize next-token CE on the
teacher target with Adam, learning rate 0.1, at most 100 updates per restart and
two restarts. Initialize gates to 0.2 / 0.8; mixture logits are uniform / slightly
perturbed. Use identical initializations for the real and shuffled conditions.

Stop a condition at its first argmax match. Save the matching checkpoint; if
none matches, retain the state with minimum CE, including the unmodified
baseline as a fallback. Thus this is a success-search diagnostic, not a
comparison of converged probability optima or equal actual optimizer steps.

## Paired control

The shuffled condition applies one common random permutation to document token
positions in every teacher head. Keep the first token, question, and generated
suffix fixed. This preserves each head's probability values and total document
attention mass while breaking its correspondence to document content. Both
conditions have the same parameterization and maximum optimization budget.

This control tests dependence on correct token-position information. It does
not by itself establish that a specific teacher head's semantic function has
been transferred.

## Efficient, checked forward path

Because only the last query row in the final four layers is changed, earlier
token KV states and the last token's lower 24 layers remain unaffected. Cache
the prefix and the input to student layer 24 once, then run the final four
layers plus the original norm and LM head during optimization.

Recreate the mutable `DynamicCache` from immutable tuple caches for each
iteration. Never append the current token repeatedly to a shared cache.

On the first eight states, check:

- Zero gates reproduce the original cached logits.
- Nonzero gates produce the same logits and mixture/gate gradients as a full
  cached model forward with the same attention intervention.
- Both optimized parameter groups receive finite, nonzero gradients.
- Also report whether full-sequence and cached unmodified forwards agree on
  argmax; their BF16 numerical differences are recorded separately.

## Commands and artifacts

```bash
/home/wanghl/miniconda3/envs/rescue/bin/python -u \
  exp/attn_adapter_transfer/run_attention_oracle.py --max_states 8

# Resume the same selected states/results and extend to 64, without rerunning
# the first eight optimizations.
/home/wanghl/miniconda3/envs/rescue/bin/python -u \
  exp/attn_adapter_transfer/run_attention_oracle.py --max_states 64
```

Default output: `exp/attn_adapter_transfer/out_oracle_last4/`.

- `config.json`: immutable optimization configuration and source-data checksum.
- `selection.json`, `states.jsonl`: sampling accounting and fixed prefixes.
- `discarded_states.jsonl`: reconstruction argmax changes and excluded IDs.
- `oracle_results.jsonl`: per-state baseline and paired optimization results.
- `optimization_trace.jsonl`: initial/every-10-update/final diagnostics.
- `state_weights/`: per-condition best/matching weights for each state.
- `summary.json`: repair counts, paired outcomes, probabilities/ranks, timings,
  and numerical/gradient validation results.
- `verification.json`: artifact integrity and six saved-checkpoint replays
  covering a successful state, a failed state, and a later-generation state.

Primary outcome: the fraction of originally mismatching states that can be made
to match the teacher. These parameters are fitted using the exact target at
each tested state. The outcome is neither a held-out acceptance rate nor a
guarantee that a general-purpose adapter can learn the same behavior.

If optimization fails, the result concerns this teacher-mixture/gate family and
budget. It does not prove that every possible attention intervention fails.

## Results

Completed on the NVIDIA H100 in the `rescue` environment.

| Condition | Repaired states | Repair fraction | Mean target probability | Mean target rank |
| --- | ---: | ---: | ---: | ---: |
| Original student | 0 / 64 | 0% | 0.1017 | 15.6094 |
| Optimized real teacher attention | 61 / 64 | 95.3125% | 0.3685 | 1.0313 |
| Optimized shuffled-document teacher attention | 61 / 64 | 95.3125% | 0.3583 | 1.0313 |

Paired results: 61 repaired by both, three repaired by neither, zero repaired
only by either condition. Neither condition matched any target at initialization;
all successes required optimization. Total Adam updates were 1800 / 2063; success
stopping means these are not equal actual optimization budgets.

The three failures were candidate states 12, 23, and 68: teacher token ` The`
versus student token ` `. One has target rank 1 by the strict-greater-logit rank
definition, but ties lose under PyTorch argmax's first-index tie rule. Repair
counts use exact argmax equality, not rank alone.

Selection examined 73 questions: three produced no rejection, 70 produced
candidates, and six candidates were excluded because teacher or student argmax
changed on BF16 prefix reconstruction. The remaining 64 are the reported states.
Of these, 48 are at the first generated token and 16 at later positions. This
selection emphasizes early disagreements and does not assess full draft blocks.

All eight numerical validation states had zero max error for zero-gate logits,
nonzero-gate fast versus full cached logits, and mixture/gate gradients. Original
full-sequence versus cached BF16 forwards agreed on argmax in six of those eight
states; the full-sequence logits are therefore an additional numerical diagnostic,
not the optimization reference. This experiment consistently uses cached
reproducible states. Fixed the existing attention steering helper so an exactly
zero gate preserves the original BF16 attention row without renormalizing it.

Reloaded both conditions' saved checkpoints for candidate states 0, 12, and 40.
All six replays reproduced the recorded argmax, CE, and target probability
(absolute tolerance 1e-5), including the later-prefix and failed cases.

Interpretation: this teacher-mixture/gate family can control the student's
next-token selection on most individually optimized states. Correct document
token positions provide no extra repaired states in this run. Because the
shuffle preserves teacher attention on the first token, question, and generated
suffix, this result does not establish that all teacher information is
unnecessary. A teacher-free control and held-out evaluation of shared/dynamic
parameters are needed to distinguish useful teacher guidance from generic
attention perturbation and per-state fitting.
