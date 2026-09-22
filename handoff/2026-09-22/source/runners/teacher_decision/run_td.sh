#!/usr/bin/env bash
# RESCUE-DECISION-AWARE-TEACHER-SIGNAL-SPRINT-001 - Phase C runner (handoff copy).
# Oracle-style diagnostic only: no training, no method change, no parameter update,
# frozen harness V3, frozen step80, frozen 67-position population, frozen smoke/pilot set.
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
# RESCUE-DECISION-AWARE-TEACHER-SIGNAL-SPRINT-001 — Phase C runner (remote host).
# Oracle-style diagnostic only: no training, no method change, no parameter update,
# frozen harness V3, frozen step80, frozen 67-position population, frozen smoke/pilot set.
set -u

G=${RUN_DIR:-$R/td_run_001}
O=${OUTPUT_ROOT:-$G/out}
POP=${POPULATION:-$R/review_run_001/out/teacher_attribution_population.json}
SET=${POSITIONS_SET:-$R/review_run_001/out/predetermined_smoke_and_pilot_set.json}
SRC=${RUN_SOURCE_DIR:-$R/mech_run_001/run_source_union}
MECHCFG=${PARENT_CONFIG:-$R/mech_run_001/out/mech_experiment_config.json}
STAGE=${1:-help}

mkdir -p "$O"
cd "$G" || exit 1

T_COMMON="--harness-root $H --config $O/td_experiment_config.json --population $POP --positions-set $SET --k4-dir $R/prefix_fix_001 --source-dir $SRC --source-lock $SL --teacher-model $TM --student-model $SM --dataset-path $DS --validation-ledger $LED/validation.csv --validation-prior-cache $PRIOR --checkpoint $CK --out $O"
S_COMMON="--harness-root $H --config $O/td_experiment_config.json --population $POP --positions-set $SET --k4-dir $R/prefix_fix_001 --source-dir $SRC --source-lock $SL --teacher-model $TM --student-model $SM --dataset-path $DS --validation-ledger $LED/validation.csv --validation-prior-cache $PRIOR --checkpoint $CK --out $O"
FAIL=0

run_teacher() {   # $1 set-name  $2 shard-name  $3 extra flags
  echo "--- teacher attribution set=$1 shard=$2 $(date -Is)"
  CUDA_VISIBLE_DEVICES=0,1,2,3 $PY -u td_teacher_attribution.py $T_COMMON \
    --set-name "$1" --shard-name "$2" $3
  local rc=$?
  echo "--- teacher set=$1 exit=$rc"
  [ "$rc" -eq 0 ] || FAIL=1
}

run_student() {   # $1 set-name  $2 shard-name  $3 lo  $4 hi  $5 gpu
  echo "--- student alignment set=$1 shard=$2 $(date -Is)"
  CUDA_VISIBLE_DEVICES=$5 $PY -u td_student_alignment.py $S_COMMON \
    --teacher-scores "$O/td_scores_$2.pt" --set-name "$1" --shard-name "$2" \
    --shard-lo "$3" --shard-hi "$4"
  local rc=$?
  echo "--- student set=$1 exit=$rc"
  [ "$rc" -eq 0 ] || FAIL=1
}

case "$STAGE" in
prepare)
  CUDA_VISIBLE_DEVICES="" $PY -u td_prepare.py --harness-root "$H" \
    --manifest-sha256 "$MANIFEST_SHA" --checkpoint "$CK" --teacher-model "$TM" \
    --student-model "$SM" --source-lock "$SL" --source-dir "$SRC" --population "$POP" \
    --positions-set "$SET" --parent-config "$MECHCFG" --out "$O" 2>&1 | tee "$O/log_td_prepare.txt"
  ;;
smoke_short)
  run_teacher short  smoke_short "--smoke --framing-check --observer-check" \
    > "$O/log_td_teacher_smoke_short.txt" 2>&1
  tail -n 4 "$O/log_td_teacher_smoke_short.txt"
  ;;
smoke_long)
  run_teacher long  smoke_long "--smoke --framing-check --observer-check" \
    > "$O/log_td_teacher_smoke_long.txt" 2>&1
  tail -n 4 "$O/log_td_teacher_smoke_long.txt"
  ;;
pilot_teacher)
  run_teacher pilot pilot_teacher "--framing-check" \
    > "$O/log_td_teacher_pilot.txt" 2>&1
  tail -n 4 "$O/log_td_teacher_pilot.txt"
  ;;
pilot_student)
  run_student pilot pilot_teacher 0 8 0 > "$O/log_td_student_pilot.txt" 2>&1
  tail -n 4 "$O/log_td_student_pilot.txt"
  ;;
expand_teacher)
  run_teacher all_population expand_teacher "" > "$O/log_td_teacher_expand.txt" 2>&1
  tail -n 4 "$O/log_td_teacher_expand.txt"
  ;;
expand_student)
  PIDS=()
  for i in 0 1 2 3; do
    lo=$((i * 67 / 4)); hi=$(((i + 1) * 67 / 4))
    echo "--- student expand shard e$i positions $lo..$hi on cuda:$i"
    CUDA_VISIBLE_DEVICES=$i $PY -u td_student_alignment.py $S_COMMON \
      --teacher-scores "$O/td_scores_expand_teacher.pt" --set-name all_population \
      --shard-name "e$i" --shard-lo "$lo" --shard-hi "$hi" \
      > "$O/log_td_student_expand_e$i.txt" 2>&1 &
    PIDS+=($!)
  done
  for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done
  for i in 0 1 2 3; do tail -n 2 "$O/log_td_student_expand_e$i.txt"; done
  ;;
smoke_student)
  run_student short smoke_short 0 1 0 > "$O/log_td_student_smoke_short.txt" 2>&1
  tail -n 4 "$O/log_td_student_smoke_short.txt"
  ;;
*)
  echo "usage: run_td.sh {prepare|smoke_short|smoke_long|pilot_teacher|pilot_student|expand_teacher|expand_student}"
  ;;
esac

echo "=== stage $STAGE done $(date -Is) fail=$FAIL ==="
exit "$FAIL"
