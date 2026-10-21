#!/usr/bin/env bash
# Baseline-only selection. No training, Mapper update or confirmation access.
set -Eeuo pipefail
run_root=/mnt/8T/cl/rescue_experiment_cache/baseline_selection_20261020_v1
run_python=/mnt/8T/cl/rescue_experiment_cache/specextend_match_20261020_v1/venv/bin/python
expected_gpu=GPU-9db4f7b5-1447-9a2a-5c1b-80379d909fd9
test -d "$run_root/after"
test -x "$run_python"
exec 200>"$run_root/pipeline.lock"
flock -n 200 || exit 73
test ! -e "$run_root/RUN_STARTED.json"
test ! -e "$run_root/results"
export CUDA_VISIBLE_DEVICES="$expected_gpu"
export PYTHONHASHSEED=20261023
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export MPLBACKEND=Agg
cd "$run_root/after"
verify_staged() {
  "$run_python" -c 'import hashlib,json,pathlib; root=pathlib.Path("/mnt/8T/cl/rescue_experiment_cache/baseline_selection_20261020_v1"); expected=json.loads((root/"LAUNCH_MANIFEST.json").read_text())["source_sha256"]; assert expected; actual={p:hashlib.sha256((root/"after"/p).read_bytes()).hexdigest() for p in expected}; assert actual==expected,"Staged source changed"; print("FROZEN_SOURCE_VERIFIED",len(actual),flush=True)'
}
verify_staged
test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)" = "$expected_gpu"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
"$run_python" -c 'import datetime,json,pathlib; p=pathlib.Path("/mnt/8T/cl/rescue_experiment_cache/baseline_selection_20261020_v1/RUN_STARTED.json"); p.open("x").write(json.dumps({"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"scope":"538-request baseline-only development selection; local CPU report required"})+"\n")'
finish() {
  pipeline_rc=$?
  trap - EXIT
  "$run_python" -c 'import datetime,json,pathlib,sys; p=pathlib.Path("/mnt/8T/cl/rescue_experiment_cache/baseline_selection_20261020_v1/PIPELINE_EXIT.json"); p.open("x").write(json.dumps({"exit_code":int(sys.argv[1]),"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"analysis_required_locally":True})+"\n")' "$pipeline_rc"
  exit "$pipeline_rc"
}
trap finish EXIT
for test_name in test_rescue_draft_graph_primitives.py test_rescue_draft_runtime.py test_rescue_baseline_runtime.py test_rescue_baseline_selection.py; do
  test -f "tests/$test_name"
  "$run_python" -m unittest discover -s tests -p "$test_name" -v
done >"$run_root/gpu_tests.log" 2>&1
verify_staged
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
"$run_python" -m rescue.baseline_selection \
  --teacher /mnt/8T/cl/models/Qwen2-7B-Instruct \
  --student /mnt/8T/cl/models/Qwen2-1.5B-Instruct \
  --data-file /home/wanghl/cl/RESCUE/exp/module_opt/data_v1_20261018/tune.jsonl \
  --output "$run_root/results" >"$run_root/benchmark.log" 2>&1
