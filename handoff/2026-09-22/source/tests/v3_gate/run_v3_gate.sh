#!/usr/bin/env bash
# RESCUE-PRE-EXPERIMENT-CORRECTNESS-GATE-V3-001 - single-shot orchestrator (handoff copy).
#
# Frozen tree is only READ.  Every new artefact lands in $O (outside the frozen tree).
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
# RESCUE-PRE-EXPERIMENT-CORRECTNESS-GATE-V3-001 — single-shot orchestrator (remote host).
#
# Frozen tree is only READ.  Every new artefact lands in $O (outside the frozen tree).
set -u

O=${OUTPUT_ROOT:-$R/pre_gate_v3_001/out}
G=${RUN_DIR:-$R/pre_gate_v3_001}
# FIX-E observations need the run-source files that actually produced the checkpoint, and
# the frozen k4 gate runs its CPU boundary suite with cwd=<source-dir>.  The training
# wrapper (sequence_prefix_runner.py) lives in prefix_fix_001 while the model/backend code
# lives in $A/source, so the harness is given ONE real directory holding symlinks to both
# (contents byte-identical; manifest recorded in $O/run_source_union_sha256.txt).
REAL_SRC="$A/source,$R/prefix_fix_001"
U=$G/run_source_union
mkdir -p "$O"
rm -rf "$U"; mkdir -p "$U"
for f in "$A"/source/*.py "$R"/prefix_fix_001/*.py; do
  ln -sf "$(readlink -f "$f")" "$U/$(basename "$f")"
done
SRC="$U"
{
  echo "# run_source_union — symlinks to the real run sources"
  for f in "$U"/*.py; do
    real=$(readlink -f "$f")
    printf '%s  %s  <- %s\n' "$(sha256sum "$real" | cut -d' ' -f1)" "$(basename "$f")" "$real"
  done
} > "$O/run_source_union_sha256.txt"

cd "$G" || exit 1
echo "=== V3 pre-experiment correctness gate start $(date -Is) ==="

echo "--- §1 identity preflight (CPU only) ---"
CUDA_VISIBLE_DEVICES="" $PY -u v3_identity_preflight.py \
  --harness-root "$H" --source-lock "$SL" --run-source-dir "$REAL_SRC" \
  --checkpoint "$CK" --teacher-model "$TM" --student-model "$SM" \
  --train-ledger "$LED/train.csv" --validation-ledger "$LED/validation.csv" \
  --provenance-envelope "$H/results/run_provenance_080.json" \
  --direct-training-config "$H/results/training_config_080.json" \
  --recorded-contract "$H/results/checkpoint_identity_contract.json" \
  --python "$PY" --out "$O" 2>&1 | tee "$O/log_identity_preflight.txt"
ID_RC=${PIPESTATUS[0]}
if [ "$ID_RC" != "0" ]; then
  echo "IDENTITY_DRIFT -> stopping before any GPU work"
  exit 2
fi

echo "--- §2 GPU resource preflight ---"
$PY -u v3_gpu_preflight.py --out "$O" 2>&1 | tee "$O/log_gpu_preflight.txt"

echo "--- §3,§5,§7,§8,§9,§11 frozen harness gates (real GPU, incremental K4) ---"
RESCUE_E_ARTIFACTS=$H/results/e_artifacts.json \
CUDA_VISIBLE_DEVICES=0,1,2,3 $PY -u "$H/PRE_EXPERIMENT_CORRECTNESS_GATE.py" \
  --out "$O/frozen_runner" --python "$PY" --source-dir "$SRC" --source-lock "$SL" \
  --k4-dir "$R/prefix_fix_001" --teacher-model "$TM" --student-model "$SM" \
  --dataset-path "$DS" --validation-ledger "$LED/validation.csv" \
  --validation-prior-cache "$PRIOR" --checkpoint "$CK" \
  --training-config "$H/results/training_config_080.json" --samples 2 \
  2>&1 | tee "$O/log_frozen_runner.txt"
FR_RC=${PIPESTATUS[0]}
echo "frozen runner exit=$FR_RC"

echo "--- §4 (decode) + §6 intervention effect (one process per gate: 11 GiB cards) ---"
for g in empty_hook_decode intervention_effect; do
  CUDA_VISIBLE_DEVICES=0 $PY -u v3_gate_ext.py --gate "$g" --harness-root "$H" --out "$O" \
    --source-dir "$SRC" --source-lock "$SL" --teacher-model "$TM" --student-model "$SM" \
    --validation-ledger "$LED/validation.csv" \
    --validation-prior-cache "$PRIOR" --dataset-path "$DS" \
    --checkpoint "$CK" --device cuda:0 --samples 2 --decode-steps 5 \
    2>&1 | tee -a "$O/log_gate_ext.txt"
  echo "ext gate $g exit=${PIPESTATUS[0]}"
done

echo "--- §10 result integrity smoke (CPU only) ---"
CUDA_VISIBLE_DEVICES="" $PY -u v3_integrity_smoke.py --harness-root "$H" --out "$O" \
  2>&1 | tee "$O/log_result_integrity.txt"
RI_RC=${PIPESTATUS[0]}
echo "integrity exit=$RI_RC"

echo "--- §12 roll-up ---"
$PY -u v3_rollup.py --out "$O" --frozen-runner-out "$O/frozen_runner" \
  --manifest-sha256 0e569ea84d36ae91d7ad73dd7299b60c9dcab1ff0abb14e270f447449ff43788 \
  2>&1 | tee "$O/log_rollup.txt"
RO_RC=${PIPESTATUS[0]}

cd "$O" && find . -maxdepth 3 -type f ! -path "./_integrity_work/*" -print0 \
  | sort -z | xargs -0 sha256sum > SHA256SUMS.txt 2>/dev/null
echo "=== V3 gate done $(date -Is) rollup_exit=$RO_RC ==="
