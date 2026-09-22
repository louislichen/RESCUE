#!/usr/bin/env bash
# RESCUE-MINIMAL-FOUR-ARM-VALID-RERUN-001 - handoff orchestrator.
set -u
R=${RESCUE_ROOT:?set RESCUE_ROOT}; H=${HARNESS_ROOT:-$R/harness_freeze_001}
A=${ATTEMPT_ROOT:?set ATTEMPT_ROOT}; SL=${SOURCE_LOCK:?set SOURCE_LOCK}; LED=${LEDGER_ROOT:?set LEDGER_ROOT}
PY=${PYTHON_BIN:-python}; TM=${TEACHER_MODEL:?set TEACHER_MODEL}; SM=${STUDENT_MODEL:?set STUDENT_MODEL}
DS=${DATASET_PATH:?set DATASET_PATH}; CK=${CHECKPOINT:?set CHECKPOINT}; PRIOR=${PRIOR_CACHE:?set PRIOR_CACHE}
MANIFEST_SHA=${MANIFEST_SHA:-0e569ea84d36ae91d7ad73dd7299b60c9dcab1ff0abb14e270f447449ff43788}
G=${RUN_DIR:-$R/four_arm_run_001}; O=${OUTPUT_ROOT:-$G/out}; N=${PLANNED_N:-128}; MAXNEW=${MAX_NEW_TOKENS:-16}
mkdir -p "$O"; cd "$G" || exit 1
U=$G/run_source_union; rm -rf "$U"; mkdir -p "$U"
for f in "$A"/source/*.py "$R"/prefix_fix_001/*.py; do ln -sf "$(readlink -f "$f")" "$U/$(basename "$f")"; done
SRC=$U
CUDA_VISIBLE_DEVICES="" $PY -u four_arm_prepare.py --harness-root "$H" --source-lock "$SL"   --checkpoint "$CK" --teacher-model "$TM" --student-model "$SM" --dataset-path "$DS"   --validation-ledger "$LED/validation.csv" --validation-prior-cache "$PRIOR"   --manifest-sha256 "$MANIFEST_SHA" --planned-n $N --max-new-tokens $MAXNEW --out "$O" || exit 2
CFG=$O/four_arm_experiment_config.json
CUDA_VISIBLE_DEVICES="" $PY -u "$R/pre_gate_v3_001/v3_identity_preflight.py"   --harness-root "$H" --source-lock "$SL" --run-source-dir "$A/source,$R/prefix_fix_001"   --checkpoint "$CK" --teacher-model "$TM" --student-model "$SM"   --train-ledger "$LED/train.csv" --validation-ledger "$LED/validation.csv"   --provenance-envelope "$H/results/run_provenance_080.json"   --direct-training-config "$H/results/training_config_080.json"   --recorded-contract "$H/results/checkpoint_identity_contract.json" --python "$PY" --out "$O/identity" || exit 3
PIDS=()
for i in 0 1 2 3; do lo=$((i*N/4)); hi=$(((i+1)*N/4)); CUDA_VISIBLE_DEVICES=$i $PY -u four_arm_qa.py  --harness-root "$H" --config "$CFG" --source-dir "$SRC" --source-lock "$SL"  --teacher-model "$TM" --student-model "$SM" --dataset-path "$DS" --validation-ledger "$LED/validation.csv"  --validation-prior-cache "$PRIOR" --checkpoint "$CK" --device cuda:0 --max-new-tokens $MAXNEW  --shard-lo "$lo" --shard-hi "$hi" --shard-name "q$i" --out "$O" & PIDS+=($!); done
for p in "${PIDS[@]}"; do wait "$p"; done
for i in 0 1 2 3; do lo=$((i*N/4)); hi=$(((i+1)*N/4)); CUDA_VISIBLE_DEVICES=0,1,2,3 $PY -u four_arm_k4.py  --harness-root "$H" --config "$CFG" --k4-dir "$R/prefix_fix_001" --source-dir "$SRC" --source-lock "$SL"  --teacher-model "$TM" --student-model "$SM" --dataset-path "$DS" --validation-ledger "$LED/validation.csv"  --validation-prior-cache "$PRIOR" --checkpoint "$CK" --device cuda:0 --target-max-memory 8GiB  --target-dtype bfloat16 --max-new-tokens $MAXNEW --shard-lo "$lo" --shard-hi "$hi" --shard-name "k$i" --out "$O"; done
CUDA_VISIBLE_DEVICES="" $PY -u four_arm_aggregate.py --harness-root "$H" --config "$CFG" --out "$O"
