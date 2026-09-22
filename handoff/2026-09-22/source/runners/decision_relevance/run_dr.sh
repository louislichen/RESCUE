#!/usr/bin/env bash
# RESCUE-TEACHER-PRIOR-DECISION-RELEVANCE-DIAGNOSTIC-001 - orchestrator (handoff copy).
# Diagnostic only: no training, no method change, frozen harness V3, frozen step80.
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

#!/usr/bin/env bash
# RESCUE-TEACHER-PRIOR-DECISION-RELEVANCE-DIAGNOSTIC-001 — orchestrator (remote host).
# Diagnostic only: no training, no method change, frozen harness V3, frozen step80.
set -u

G=${RUN_DIR:-$R/dr_run_001}
O=${OUTPUT_ROOT:-$G/out}
N=${PLANNED_N:-128}
MAXNEW=${MAX_NEW_TOKENS:-16}
PARENT=${PARENT_RUN_OUT:-$R/four_arm_run_001/out}

mkdir -p "$O"
cd "$G" || exit 1
U=$G/run_source_union
rm -rf "$U"; mkdir -p "$U"
for f in "$A"/source/*.py "$R"/prefix_fix_001/*.py; do
  ln -sf "$(readlink -f "$f")" "$U/$(basename "$f")"
done
SRC=$U

echo "=== decision-relevance diagnostic start $(date -Is) ==="

echo "--- prepare: dataset identity + frozen diagnostic config ---"
CUDA_VISIBLE_DEVICES="" $PY -u dr_prepare.py --harness-root "$H" --source-lock "$SL" \
  --checkpoint "$CK" --teacher-model "$TM" --student-model "$SM" --dataset-path "$DS" \
  --validation-ledger "$LED/validation.csv" --validation-prior-cache "$PRIOR" \
  --manifest-sha256 "$MANIFEST_SHA" --parent-summary "$PARENT/four_arm_results.json" \
  --planned-n $N --max-new-tokens $MAXNEW --out "$O" 2>&1 | tee "$O/log_prepare.txt"
PREP=${PIPESTATUS[0]}
[ "$PREP" = "0" ] || { echo "PREPARE FAILED"; exit 2; }
CFG=$O/dr_experiment_config.json

echo "--- identity preflight (frozen fail-closed path) ---"
CUDA_VISIBLE_DEVICES="" $PY -u "$R/pre_gate_v3_001/v3_identity_preflight.py" \
  --harness-root "$H" --source-lock "$SL" --run-source-dir "$A/source,$R/prefix_fix_001" \
  --checkpoint "$CK" --teacher-model "$TM" --student-model "$SM" \
  --train-ledger "$LED/train.csv" --validation-ledger "$LED/validation.csv" \
  --provenance-envelope "$H/results/run_provenance_080.json" \
  --direct-training-config "$H/results/training_config_080.json" \
  --recorded-contract "$H/results/checkpoint_identity_contract.json" \
  --python "$PY" --out "$O/identity" 2>&1 | tee "$O/log_identity.txt"
ID=${PIPESTATUS[0]}
[ "$ID" = "0" ] || { echo "IDENTITY_DRIFT -> stopping"; exit 3; }

echo "--- measurement pass: 4 shards, one process (target on all four cards) ---"
for i in 0 1 2 3; do
  lo=$((i * N / 4)); hi=$(((i + 1) * N / 4))
  CUDA_VISIBLE_DEVICES=0,1,2,3 $PY -u dr_stream.py --harness-root "$H" --config "$CFG" \
    --k4-dir "$R/prefix_fix_001" --source-dir "$SRC" --source-lock "$SL" \
    --teacher-model "$TM" --student-model "$SM" --dataset-path "$DS" \
    --validation-ledger "$LED/validation.csv" --validation-prior-cache "$PRIOR" \
    --checkpoint "$CK" --device cuda:0 --target-max-memory 8GiB --target-dtype bfloat16 \
    --max-new-tokens $MAXNEW --shard-lo "$lo" --shard-hi "$hi" --shard-name "d$i" \
    --out "$O" > "$O/log_stream_d$i.txt" 2>&1
  echo "shard d$i exit=$? $(date -Is)"
done

echo "--- aggregate ---"
CUDA_VISIBLE_DEVICES="" $PY -u dr_aggregate.py --harness-root "$H" --config "$CFG" \
  --out "$O" --parent-qa "$PARENT/qa_four_arm.jsonl" 2>&1 | tee "$O/log_aggregate.txt"
AGG=${PIPESTATUS[0]}

cd "$O" && find . -maxdepth 3 -type f ! -name SHA256SUMS.txt -print0 \
  | sort -z | xargs -0 sha256sum > SHA256SUMS.txt 2>/dev/null
echo "=== diagnostic done $(date -Is) aggregate_exit=$AGG ==="
