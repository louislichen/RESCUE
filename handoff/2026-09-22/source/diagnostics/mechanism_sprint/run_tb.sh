#!/usr/bin/env bash
# RESCUE-GUIDANCE-REPRESENTATION-MECHANISM-SPRINT-001 - Track B orchestrator (handoff copy).
# Teacher representation R1 vs R0. Diagnostic only; no training.
set -u

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

G=${RUN_DIR:-$R/mech_run_001}; O=${OUTPUT_ROOT:-$G/out}
cd $G || exit 1
echo "=== TRACK B start $(date -Is) ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 $PY -u tb_teacher_representation.py --harness-root "$H"   --config "${MECH_CONFIG:-$G/out/mech_experiment_config.json}" --k4-dir "${K4_DIR:-$R/prefix_fix_001}"   --source-dir "${RUN_SOURCE_DIR:-$G/run_source_union}" --source-lock "$SL" --teacher-model "$TM"   --student-model "$SM" --dataset-path "$DS" --validation-ledger "$LED/validation.csv"   --validation-prior-cache "$PRIOR" --checkpoint "$CK" --device cuda:0   --target-max-memory 8GiB --target-dtype bfloat16 --shard-name b0 --attribution-probe   --out "$O" > "$O/log_tb.txt" 2>&1
echo "=== TRACK B done $(date -Is) exit=$? ==="
