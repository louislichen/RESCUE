#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# RESCUE handoff helper: materialise the historical run-tree layout that the
# frozen run orchestrators expect, using symlinks into THIS package.
#
# The ten orchestrators were written against a flat run root:
#
#   $RESCUE_ROOT/
#     harness_freeze_001/   prefix_fix_001/   four_arm_run_001/
#     dr_run_001/           r1_run_001/       td_run_001/
#     mech_run_001/         review_run_001/   pre_gate_v3_001/
#     diagnostics_001/      assemble_evidence.py
#
# This package stores the same sources under harness/ runtime/ runners/
# diagnostics/ tests/.  This script bridges the two without copying sources.
#
# It also stages what IS shipped (the frozen step-80 checkpoint, the frozen
# configs, the small result snapshots) and reports what is NOT shipped and must
# be supplied by you -- see EXTERNAL_ARTIFACTS.md.
#
#   usage:  bash make_run_tree.sh <package-root> <rescue-root>
# ---------------------------------------------------------------------------
set -u

PKG=${1:?usage: make_run_tree.sh <package-root> <rescue-root>}
R=${2:?usage: make_run_tree.sh <package-root> <rescue-root>}
PKG=$(cd "$PKG" && pwd)
mkdir -p "$R"
R=$(cd "$R" && pwd)

link () {  # link <pkg-relative-source> <run-root-relative-target>
  local src="$PKG/$1" dst="$R/$2"
  mkdir -p "$(dirname "$dst")"
  rm -rf "$dst"
  ln -s "$src" "$dst"
  echo "  link  $2  ->  $1"
}

echo "== mapping packaged sources into $R =="
link harness             harness_freeze_001
link runtime/prefix_fix  prefix_fix_001
link runtime/source_lock source_lock
link runtime/frozen      frozen_contracts
link tests/v3_gate       pre_gate_v3_001
link runners/four_arm            four_arm_run_001
link runners/decision_relevance  dr_run_001
link runners/r1_pilot            r1_run_001
link runners/teacher_decision    td_run_001
link diagnostics/mechanism_sprint              mech_run_001
link diagnostics/teacher_decision_attribution  review_run_001
link diagnostics/adapter_path                  diagnostics_001
link diagnostics/assemble_evidence.py          assemble_evidence.py
link data                data_authority_and_role_ledgers

echo "== rebuilding the run-source union (symlinks, as the frozen gate did) =="
for U in "$R/mech_run_001/run_source_union" "$R/four_arm_run_001/run_source_union"; do
  mkdir -p "$U"
  for f in "$PKG"/runtime/source/*.py "$PKG"/runtime/prefix_fix/*.py; do
    ln -sf "$(readlink -f "$f")" "$U/$(basename "$f")"
  done
  echo "  $U  ($(ls -1 "$U" | wc -l) files)"
done

echo "== staging the frozen checkpoint (bundled with this package) =="
if [ -f "$PKG/external/checkpoint_080.pt" ]; then
  mkdir -p "$R/prefix_fix_001/run_001"
  cp -n "$PKG/external/checkpoint_080.pt" "$R/prefix_fix_001/run_001/checkpoint_080.pt"
  echo "  checkpoint_080.pt -> prefix_fix_001/run_001/  (sha256 6532daad…)"
else
  echo "  MISSING: external/checkpoint_080.pt"
fi

echo "== staging the frozen experiment configs the orchestrators read =="
mkdir -p "$R/four_arm_run_001/out" "$R/dr_run_001/out" "$R/r1_run_001/out" \
         "$R/td_run_001/out" "$R/mech_run_001/out" "$R/review_run_001/out" \
         "$R/pre_gate_v3_001/out"
cp -n "$PKG"/configs/four_arm_experiment_config.json "$R/four_arm_run_001/out/" 2>/dev/null
cp -n "$PKG"/configs/dr_experiment_config.json       "$R/dr_run_001/out/"       2>/dev/null
cp -n "$PKG"/configs/r1_experiment_config.json       "$R/r1_run_001/out/"       2>/dev/null
cp -n "$PKG"/configs/td_experiment_config.json       "$R/td_run_001/out/"       2>/dev/null
cp -n "$PKG"/configs/mech_experiment_config.json     "$R/mech_run_001/out/"     2>/dev/null
cp -n "$PKG"/configs/populations/*.json              "$R/review_run_001/out/"   2>/dev/null
cp -n "$PKG"/results/four_arm/*.json "$PKG"/results/four_arm/*.csv "$R/four_arm_run_001/out/" 2>/dev/null
echo "  configs + small result snapshots staged"

echo
echo "== bundled and ready =="
echo "  frozen step-80 checkpoint   external/checkpoint_080.pt"
echo "  frozen experiment configs   configs/"
echo "  frozen TD population +      configs/populations/"
echo "  pre-registered pilot set"
echo
echo "== still required from OUTSIDE this package (see EXTERNAL_ARTIFACTS.md) =="
echo "  PRIOR_CACHE   <external>/validation_teacher_priors.pt"
echo "                  sha256 b820317774791c48439f8caa52d1410c9c744018524e92bb5ce02d033e0256c7  (82 951 959 B)"
echo "  TEACHER_MODEL / STUDENT_MODEL   the locked Qwen2-7B / Qwen2-1.5B snapshots"
echo "  DATASET_PATH  HotpotQA distractor validation directory"
echo
echo "  row-level '*_rows*.jsonl' bodies (MB-scale) are NOT shipped.  These run-dir inputs"
echo "  default to \$RESCUE_ROOT/... and have env overrides:"
echo "    DR_ROWS           -> dr_run_001/out/decision_relevance_rows.jsonl"
echo "    TA_ROWS           -> mech_run_001/out/student_controllability_rows.jsonl"
echo "    PARENT_RUN_OUT    -> four_arm_run_001/out   (needs qa_four_arm.jsonl)"
echo
echo "done.  export RESCUE_ROOT=$R and see REPRODUCTION.md sections B-F."
