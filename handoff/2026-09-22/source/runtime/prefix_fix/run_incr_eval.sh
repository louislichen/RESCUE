#!/usr/bin/env bash
# Incremental-K4 evaluation + evidence assembly (handoff copy).
# Produced prefix_fix_001/k4_incr/k4_correctness.json (verify_mode=incremental).
set -u

# ---------------------------------------------------------------------------
# PORTABILITY NOTE (handoff 2026-09-22)
# The frozen original hardcoded absolute paths of the execution host.  Every path
# below is now overridable from the environment.  Command lines, ordering, flags,
# exit-code handling and fail-closed gates are byte-for-byte the frozen behaviour.
# Original literal values (execution host, for provenance only):
#   R   = /home/ztl/rescue_runs/parallel_closure_003/k4_seqloss_dev_closure_002
#   H   = $R/harness_freeze_001
#   A   = /home/ztl/rescue_runs/parallel_closure_003/k4_resume_sequence_loss_001/attempts/20260915_resume_001
#   SL  = /home/ztl/rescue_runs/parallel_closure_003/q3_executable_completion_004/postmeeting_001/source_lock
#   LED = /home/ztl/rescue_runs/parallel_closure_003/q3_executable_completion_004/postmeeting_001/data_authority_and_role_ledgers
#   PY  = /home/ztl/anaconda3/envs/rescue-r0/bin/python
#   TM  = /home/ztl/jwt/models/locked/Qwen2-7B__453ed1575b739b5b03ce3758b23befdb0967f40e
#   SM  = /home/ztl/jwt/models/locked/Qwen2-1.5B__8a16abf2848eda07cc5253dec660bf1ce007ad7a
#   DS  = /home/ztl/jwt/datasets/hotpotqa/hotpot_qa/distractor/validation
#   CK  = $R/prefix_fix_001/run_001/checkpoint_080.pt
# ---------------------------------------------------------------------------
R=${RESCUE_ROOT:?set RESCUE_ROOT (the run root that holds harness_freeze_001, four_arm_run_001, ...)}
H=${HARNESS_ROOT:-$R/harness_freeze_001}
A=${ATTEMPT_ROOT:?set ATTEMPT_ROOT (the 20260915_resume_001 attempt directory)}
SL=${SOURCE_LOCK:?set SOURCE_LOCK (the postmeeting_001/source_lock directory)}
LED=${LEDGER_ROOT:?set LEDGER_ROOT (the postmeeting_001/data_authority_and_role_ledgers directory)}
PY=${PYTHON_BIN:-python}
TM=${TEACHER_MODEL:?set TEACHER_MODEL (locked Qwen2-7B snapshot path)}
SM=${STUDENT_MODEL:?set STUDENT_MODEL (locked Qwen2-1.5B snapshot path)}
DS=${DATASET_PATH:?set DATASET_PATH (hotpot_qa distractor validation directory)}
CK=${CHECKPOINT:?set CHECKPOINT (checkpoint_080.pt)}
PRIOR=${PRIOR_CACHE:?set PRIOR_CACHE (validation_teacher_priors.pt)}
MANIFEST_SHA=${MANIFEST_SHA:-0e569ea84d36ae91d7ad73dd7299b60c9dcab1ff0abb14e270f447449ff43788}

set -uo pipefail
D=${PREFIX_FIX_DIR:-$R/prefix_fix_001}
cd $D
CUDA_VISIBLE_DEVICES=0,1,2,3 $PY -u real_k4_runner.py --mode eval \
  --root "${K4_OUT:-$D/k4_incr}" --source-dir "$ATTEMPT_ROOT/source" --source-lock "$SL" \
  --teacher-model "$TM" \
  --student-model "$SM" \
  --dataset-path "$DS" \
  --validation-ledger "$LED" \
  --validation-prior-cache "$PRIOR" --checkpoint $CK \
  --device cuda:0 --target-max-memory 8GiB --target-dtype bfloat16 --verify-mode incremental \
  --shard-name incr --shard-lo 0 --shard-hi 128 >> $R/incr_eval.log 2>&1
echo "incr eval exit=$?" >> $R/incr_eval.log
$PY ${ASSEMBLE_EVIDENCE:-$R/assemble_evidence.py} --root $R --out $R/evidence_002 >> $R/incr_eval.log 2>&1
echo "assemble done" >> $R/incr_eval.log
