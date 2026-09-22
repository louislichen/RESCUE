#!/usr/bin/env bash
# RESCUE-PRE-EXPERIMENT-CORRECTNESS-GATE-V3-001 - single-shot orchestrator (handoff copy).
set -u
R=${RESCUE_ROOT:?set RESCUE_ROOT}
H=${HARNESS_ROOT:-$R/harness_freeze_001}
A=${ATTEMPT_ROOT:?set ATTEMPT_ROOT}
SL=${SOURCE_LOCK:?set SOURCE_LOCK}
LED=${LEDGER_ROOT:?set LEDGER_ROOT}
PY=${PYTHON_BIN:-python}
TM=${TEACHER_MODEL:?set TEACHER_MODEL}
SM=${STUDENT_MODEL:?set STUDENT_MODEL}
DS=${DATASET_PATH:?set DATASET_PATH}
CK=${CHECKPOINT:?set CHECKPOINT}
PRIOR=${PRIOR_CACHE:?set PRIOR_CACHE}
O=${OUTPUT_ROOT:-$R/pre_gate_v3_001/out}
G=${RUN_DIR:-$R/pre_gate_v3_001}
REAL_SRC="$A/source,$R/prefix_fix_001"
U=$G/run_source_union
mkdir -p "$O"; rm -rf "$U"; mkdir -p "$U"
for f in "$A"/source/*.py "$R"/prefix_fix_001/*.py; do ln -sf "$(readlink -f "$f")" "$U/$(basename "$f")"; done
SRC="$U"
cd "$G" || exit 1
CUDA_VISIBLE_DEVICES="" $PY -u v3_identity_preflight.py   --harness-root "$H" --source-lock "$SL" --run-source-dir "$REAL_SRC"   --checkpoint "$CK" --teacher-model "$TM" --student-model "$SM"   --train-ledger "$LED/train.csv" --validation-ledger "$LED/validation.csv"   --provenance-envelope "$H/results/run_provenance_080.json"   --direct-training-config "$H/results/training_config_080.json"   --recorded-contract "$H/results/checkpoint_identity_contract.json"   --python "$PY" --out "$O" || exit 2
$PY -u v3_gpu_preflight.py --out "$O"
RESCUE_E_ARTIFACTS=$H/results/e_artifacts.json CUDA_VISIBLE_DEVICES=0,1,2,3   $PY -u "$H/PRE_EXPERIMENT_CORRECTNESS_GATE.py" --out "$O/frozen_runner"   --python "$PY" --source-dir "$SRC" --source-lock "$SL" --k4-dir "$R/prefix_fix_001"   --teacher-model "$TM" --student-model "$SM" --dataset-path "$DS"   --validation-ledger "$LED/validation.csv" --validation-prior-cache "$PRIOR"   --checkpoint "$CK" --training-config "$H/results/training_config_080.json" --samples 2
for g in empty_hook_decode intervention_effect; do
  CUDA_VISIBLE_DEVICES=0 $PY -u v3_gate_ext.py --gate "$g" --harness-root "$H" --out "$O"     --source-dir "$SRC" --source-lock "$SL" --teacher-model "$TM" --student-model "$SM"     --validation-ledger "$LED/validation.csv" --validation-prior-cache "$PRIOR"     --dataset-path "$DS" --checkpoint "$CK" --device cuda:0 --samples 2 --decode-steps 5
done
CUDA_VISIBLE_DEVICES="" $PY -u v3_integrity_smoke.py --harness-root "$H" --out "$O"
$PY -u v3_rollup.py --out "$O" --frozen-runner-out "$O/frozen_runner"   --manifest-sha256 0e569ea84d36ae91d7ad73dd7299b60c9dcab1ff0abb14e270f447449ff43788
