#!/usr/bin/env bash
# RESCUE-GUIDANCE-REPRESENTATION-MECHANISM-SPRINT-001 - Track C + D orchestrator (handoff copy).
# Semantic grounding + pre-registered failure strata. Diagnostic only; no training.
set -u

R=${RESCUE_ROOT:?set RESCUE_ROOT (the run root that holds harness_freeze_001, four_arm_run_001, ...)}
H=${HARNESS_ROOT:-$R/harness_freeze_001}
SL=${SOURCE_LOCK:?set SOURCE_LOCK (the postmeeting_001/source_lock directory)}
LED=${LEDGER_ROOT:?set LEDGER_ROOT (the postmeeting_001/data_authority_and_role_ledgers directory)}
PY=${PYTHON_BIN:-python}
TM=${TEACHER_MODEL:?set TEACHER_MODEL (locked Qwen2-7B snapshot path)}
SM=${STUDENT_MODEL:?set STUDENT_MODEL (locked Qwen2-1.5B snapshot path)}
DS=${DATASET_PATH:?set DATASET_PATH (hotpot_qa distractor validation directory)}
CK=${CHECKPOINT:?set CHECKPOINT (checkpoint_080.pt)}
PRIOR=${PRIOR_CACHE:?set PRIOR_CACHE (validation_teacher_priors.pt)}

G=${RUN_DIR:-$R/mech_run_001}; O=${OUTPUT_ROOT:-$G/out}
DR=${DR_ROWS:-$R/dr_run_001/out/decision_relevance_rows.jsonl}
cd $G || exit 1
PIDS=()
for i in 0 1 2 3; do
  lo=$((i*496/4)); hi=$(((i+1)*496/4))
  CUDA_VISIBLE_DEVICES=$i $PY -u td_features.py --harness-root "$H"     --config "${MECH_CONFIG:-$G/out/mech_experiment_config.json}" --dr-rows "$DR"     --source-dir "${RUN_SOURCE_DIR:-$G/run_source_union}" --source-lock "$SL" --teacher-model "$TM"     --student-model "$SM" --dataset-path "$DS" --validation-ledger "$LED/validation.csv"     --checkpoint "$CK" --device cuda:0 --shard-lo "$lo" --shard-hi "$hi"     --shard-name "d$i" --out "$O" > "$O/log_td_d$i.txt" 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
CUDA_VISIBLE_DEVICES=0,1,2,3 $PY -u tc_semantic_grounding.py --harness-root "$H"   --config "${MECH_CONFIG:-$G/out/mech_experiment_config.json}" --k4-dir "$R/prefix_fix_001" --dr-rows "$DR"   --source-dir "${RUN_SOURCE_DIR:-$G/run_source_union}" --source-lock "$SL" --teacher-model "$TM"   --student-model "$SM" --dataset-path "$DS" --validation-ledger "$LED/validation.csv"   --validation-prior-cache "$PRIOR" --checkpoint "$CK" --device cuda:0   --target-max-memory 8GiB --target-dtype bfloat16 --shard-name c0 --out "$O" > "$O/log_tc.txt" 2>&1
