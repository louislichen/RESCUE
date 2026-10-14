# Selective teacher-V content injection

This pilot follows the attention-selection stage by testing content enrichment.
Qwen2-7B and Qwen2-1.5B remain frozen; only a mapper and optional query-dependent
gate are trained. The teacher verifier never consumes mapped states.

## Fixed protocol

- Reuse 24 training and 12 validation documents from stage two. Acquire 30 new
  test documents, excluding every previous question and document title.
- FP32 teacher, BF16 student, eager attention, TF32 disabled, one H100.
- Contexts 600–2048 tokens. Read 512 student-cache positions selected by the
  teacher's last-layer head-mean attention; preserve sink and latest 64 tokens.
- Inject only the top 128 remote positions among that working set, in student
  layers 25–28 (human numbering). Teacher sources follow the training-only
  stage-two attention pairing: layers 25, 27, 27, 28.
- Teacher V and attention come from a completed prefix. Training deliberately
  uses the previous completed query's attention, matching runtime availability.
  No fresh teacher query is run for each draft token.
- K is returned unchanged by identity. V corrections exist only in working
  reads, after the native backing cache has been updated. Rejected draft tails
  are cropped from both actual caches after verification. Generated student
  states can depend on earlier injected content; they are not claimed to equal
  the frozen student's uninterrupted native trajectory.

## Mapper, gate, and objectives

Each injected layer has a teacher-value mapper:
`LayerNorm(512) → Linear(512,64) → SiLU → Linear(64,256)`.
Flattening all GQA value heads avoids assuming one-to-one attention-head
alignment. The mapped vector is rescaled by each native value's RMS. The
correction is bounded componentwise:

`delta = 2 * RMS(Vs) * tanh((mapped_value - Vs) / (2 * RMS(Vs)))`

`V_read = Vs + gate * delta`.

The learned gate sees the mapper's teacher-content latent, projected native V,
the current student's normalized deep-layer query state, log teacher attention,
relative position, and mapped/native cosine. It outputs one sigmoid gate per
token and layer. The gate can attenuate corrections but has no exact hard-zero
threshold; deployment also considers a globally zero intervention on validation.

Three objectives/policies are compared:

1. **Reconstruction**: native-value normalized MSE, followed by constant blending.
2. **Performance, fixed**: train the mapper through frozen student output using
   `KL(p_teacher || q_student) + 0.25 * CE(teacher_argmax)` and a 0.01 normalized
   applied-residual penalty; train at strength 0.25.
3. **Performance, gated**: initialize from reconstruction and jointly train the
   mapper and selective gate with the same output objective and penalty.

This is a differentiable output-quality surrogate, not direct optimization of
the discontinuous greedy acceptance rate. Actual acceptance is measured by SD.
Checkpoint and strength choices use validation only; test is examined afterward.
Both a zero-allowed validation choice and the best forced-positive diagnostic
are recorded, so a rejected intervention is not silently presented as deployment.

Controls use native full KV, native teacher-token selection, reconstruction,
performance with constant strength, learned gates, the same learned mapper with
a validation-matched constant mean gate, and shuffled teacher mapped content.
The shuffled control leaves location scores and the gate's teacher-content
inputs unshuffled, and shuffles only mapped content directions within the
injected remote positions. Earlier corrections can change later query states,
so realized downstream gates can differ; they are not replayed from the normal
intervention trajectory.

Runtime generates at most 32 tokens with draft k=4, plus the usual verified bonus
or correction. Full prefill, attention acquisition, selection, source gathering,
mapper, gates, drafting, verifier, and cache updates are charged. Full backing
caches remain allocated; this pilot does not claim storage savings or removal
of student prefill.

## Run

```bash
/home/wanghl/miniconda3/envs/rescue/bin/python exp/selective_kv/prepare_data.py
/home/wanghl/miniconda3/envs/rescue/bin/python exp/selective_kv/run.py --phase all
/home/wanghl/miniconda3/envs/idn/bin/python exp/selective_kv/analyze.py
```

Frozen post-pilot source ablations (after the primary content shuffle retained
gains) have a separate protocol and report:

```bash
/home/wanghl/miniconda3/envs/rescue/bin/python exp/selective_kv/source_controls.py
/home/wanghl/miniconda3/envs/idn/bin/python exp/selective_kv/analyze_source_controls.py
```

GPU phases run sequentially. Local tensor caches are under
`/mnt/8T/cl/rescue_experiment_cache/selective_kv_pilot`; model checkpoints and
process logs are ignored by Git. Raw JSON results and the final report live in
`out_pilot/`. Resume requires the same data, parameters, and source code.

The first pilot and frozen supplementary source controls have completed. See
[Chinese conclusions](out_pilot/conclusions.md), [primary report](out_pilot/report.md),
and [source-content controls](out_pilot/source_controls_report.md).
