#!/usr/bin/env bash
# RESCUE-GUIDANCE-REPRESENTATION-MECHANISM-SPRINT-001 - Track A orchestrator (handoff copy).
# Student local gradient + same-budget feasible Oracle. Diagnostic only; no training.
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
MANIFEST_SHA=${MANIFEST_SHA:-0e569ea84d36ae91d7ad73dd7299b60c9dcab1ff0abb14e270f447449ff43788}

G=${RUN_DIR:-$R/mech_run_001}; O=${OUTPUT_ROOT:-$G/out}
cd $G || exit 1
echo "=== TRACK A start $(date -Is) ==="
PIDS=()
for i in 0 1 2 3; do
  lo=$((i*496/4)); hi=$(((i+1)*496/4))
  CUDA_VISIBLE_DEVICES=$i $PY -u ta_controllability.py --harness-root "$H"     --config "${MECH_CONFIG:-$G/out/mech_experiment_config.json}"     --dr-rows "${DR_ROWS:-$R/dr_run_001/out/decision_relevance_rows.jsonl}"     --source-dir "${RUN_SOURCE_DIR:-$G/run_source_union}" --source-lock "$SL" --teacher-model "$TM"     --student-model "$SM" --dataset-path "$DS" --validation-ledger "$LED/validation.csv"     --validation-prior-cache "$PRIOR" --checkpoint "$CK" --device cuda:0     --groups "S!=T,S=T" --shard-lo "$lo" --shard-hi "$hi" --shard-name "a$i" --out "$O"     > "$O/log_ta_a$i.txt" 2>&1 &
  PIDS+=($!)
done
FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done
echo "=== TRACK A done $(date -Is) fail=$FAIL ==="
