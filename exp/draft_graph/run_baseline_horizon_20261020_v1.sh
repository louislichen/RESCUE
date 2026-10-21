#!/usr/bin/env bash
# Separate old-data development study. Never resumes stage1 or opens new data.
set -Eeuo pipefail
run_root=/mnt/8T/cl/rescue_experiment_cache/baseline_horizon_20261020_v1
run_python=/mnt/8T/cl/rescue_experiment_cache/specextend_match_20261020_v1/venv/bin/python
expected_gpu=GPU-9db4f7b5-1447-9a2a-5c1b-80379d909fd9
stage1_root="$run_root/stage1_evidence"
validate_only=false
if [[ $# -eq 1 && $1 == --validate-only ]]; then
  validate_only=true
elif [[ $# -ne 0 ]]; then
  printf '%s\n' 'Usage: run_baseline_horizon_20261020_v1.sh [--validate-only]' >&2
  exit 64
fi
test -d "$run_root/after"
test -x "$run_python"
test ! -e "$run_root/RUN_STARTED.json"
test ! -e "$run_root/results"
test ! -e "$run_root/PIPELINE_EXIT.json"
export CUDA_VISIBLE_DEVICES="$expected_gpu"
export PYTHONHASHSEED=20261025
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONOPTIMIZE
cd "$run_root/after"
verify_staged() {
  "$run_python" -c 'import hashlib,json,pathlib; root=pathlib.Path("/mnt/8T/cl/rescue_experiment_cache/baseline_horizon_20261020_v1"); expected=json.loads((root/"LAUNCH_MANIFEST.json").read_text())["source_sha256"]; assert expected; actual={p:hashlib.sha256((root/"after"/p).read_bytes()).hexdigest() for p in expected}; assert actual==expected,"Staged source changed"; print("FROZEN_SOURCE_VERIFIED",len(actual),flush=True)'
}
verify_stage1() {
  "$run_python" -c 'from pathlib import Path; from rescue.fair_confirmation_data import validate_baseline_freeze; p=Path("/mnt/8T/cl/rescue_experiment_cache/baseline_horizon_20261020_v1/stage1_evidence"); proof=validate_baseline_freeze(p/"results",p/"results/analysis_final_v1",p/"VERIFIED_COMPLETE.json",p/"FINAL_ARTIFACTS_MANIFEST.json"); print("STAGE1_COMPLETE_538_76_VERIFIED",proof["selection_freeze_sha256"],flush=True)'
}
verify_staged
verify_stage1
test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)" = "$expected_gpu"
if "$validate_only"; then
  printf '%s\n' 'VALIDATED_ONLY: no lock, output directory, model generation or launch marker created'
  exit 0
fi
if tmux has-session -t rescue_baseline_selection_v1 2>/dev/null; then
  printf '%s\n' 'Stage1 tmux is still present; refusing to start another experiment' >&2
  exit 75
fi
exec 200>"$run_root/pipeline.lock"
flock -n 200 || exit 73
test ! -e "$run_root/RUN_STARTED.json"
test ! -e "$run_root/results"
test ! -e "$run_root/PIPELINE_EXIT.json"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
"$run_python" -c 'import datetime,json,pathlib; p=pathlib.Path("/mnt/8T/cl/rescue_experiment_cache/baseline_horizon_20261020_v1/RUN_STARTED.json"); p.open("x").write(json.dumps({"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"scope":"2540-request old-data horizon-matched development selection; not confirmation; local report required"})+"\n")'
finish() {
  pipeline_rc=$?
  trap - EXIT
  "$run_python" -c 'import datetime,json,pathlib,sys; p=pathlib.Path("/mnt/8T/cl/rescue_experiment_cache/baseline_horizon_20261020_v1/PIPELINE_EXIT.json"); p.open("x").write(json.dumps({"exit_code":int(sys.argv[1]),"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"analysis_required_locally":True})+"\n")' "$pipeline_rc"
  exit "$pipeline_rc"
}
trap finish EXIT
for test_name in test_rescue_draft_graph_primitives.py test_rescue_draft_runtime.py test_rescue_baseline_runtime.py test_rescue_baseline_selection.py test_rescue_baseline_horizon_data.py test_rescue_baseline_horizon.py; do
  test -f "tests/$test_name"
  "$run_python" -m unittest discover -s tests -p "$test_name" -v
done >"$run_root/gpu_tests.log" 2>&1
verify_staged
verify_stage1
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
"$run_python" -m rescue.baseline_horizon \
  --teacher /mnt/8T/cl/models/Qwen2-7B-Instruct \
  --student /mnt/8T/cl/models/Qwen2-1.5B-Instruct \
  --train-file /home/wanghl/cl/RESCUE/exp/long_system/data/train.jsonl \
  --opt-confirmation-file /home/wanghl/cl/RESCUE/exp/module_opt/data_v1_20261018/confirmation.jsonl \
  --arch-confirmation-file /home/wanghl/cl/RESCUE/exp/module_arch/data_v1_20261018/confirmation.jsonl \
  --prime-file /home/wanghl/cl/RESCUE/exp/module_opt/data_v1_20261018/tune.jsonl \
  --stage1-results "$stage1_root/results" \
  --stage1-analysis "$stage1_root/results/analysis_final_v1" \
  --stage1-verified "$stage1_root/VERIFIED_COMPLETE.json" \
  --stage1-manifest "$stage1_root/FINAL_ARTIFACTS_MANIFEST.json" \
  --output "$run_root/results" >"$run_root/benchmark.log" 2>&1
