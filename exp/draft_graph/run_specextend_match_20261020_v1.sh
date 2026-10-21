#!/usr/bin/env bash
# Fresh isolated matched-gap pilot; never retry or overwrite partial evidence.
set -u
set -o pipefail
experiment_root=/mnt/8T/cl/rescue_experiment_cache/specextend_match_20261020_v1
original_root=/home/wanghl/cl/RESCUE
runtime="$experiment_root/venv/bin/python"
test -d "$experiment_root/after/rescue" || exit 10
test ! -e "$experiment_root/results" || exit 11
test ! -e "$experiment_root/PIPELINE_EXIT.json" || exit 12
test ! -e "$experiment_root/GPU_TEST_EXIT.json" || exit 13
exec 9>"$experiment_root/pipeline.lock"
flock -n 9 || exit 14
cd "$experiment_root/after" || exit 15
export CUDA_VISIBLE_DEVICES=GPU-9db4f7b5-1447-9a2a-5c1b-80379d909fd9
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

"$runtime" -c 'import hashlib,json,pathlib,sys; r=pathlib.Path(sys.argv[1]); m=json.loads((r/"LAUNCH_MANIFEST.json").read_text()); assert all(hashlib.sha256((r/k).read_bytes()).hexdigest()==v for k,v in m["file_sha256"].items()),"Frozen source changed; do not retry or overwrite"' "$experiment_root" || exit 16
"$runtime" -c 'import flash_attn,termcolor; from rescue.round_overhead_run import gpu_snapshot; assert flash_attn.__version__=="2.8.3"; gpu_snapshot(); print("Dependency and idle-GPU gate passed",flush=True)' || exit 18
PYTHONPATH="$experiment_root/after/tests:$experiment_root/after" "$runtime" -u -m unittest -v \
  test_rescue_draft_graph_primitives test_rescue_draft_runtime test_rescue_draft_benchmark \
  test_rescue_baseline_runtime test_rescue_specextend_match_pilot \
  > "$experiment_root/gpu_tests.log" 2>&1
test_status=$?
"$runtime" -c 'import json,sys,time; f=open(sys.argv[1],"x"); json.dump({"returncode":int(sys.argv[2]),"finished_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())},f,indent=2); f.write("\n"); f.close()' "$experiment_root/GPU_TEST_EXIT.json" "$test_status"
if [ "$test_status" -ne 0 ]; then
  exit "$test_status"
fi
"$runtime" -c 'import hashlib,json,pathlib,sys; r=pathlib.Path(sys.argv[1]); m=json.loads((r/"LAUNCH_MANIFEST.json").read_text()); assert all(hashlib.sha256((r/k).read_bytes()).hexdigest()==v for k,v in m["file_sha256"].items()),"Tested source changed"' "$experiment_root" || exit 17
"$runtime" -u -m rescue.specextend_match_pilot \
  --teacher /mnt/8T/cl/models/Qwen2-7B-Instruct \
  --student /mnt/8T/cl/models/Qwen2-1.5B-Instruct \
  --mapper "$original_root/exp/module_arch/out_v1_20261018/full_final.pt" \
  --policy "$original_root/exp/module_arch/out_v1_20261018/budget_selected.pt" \
  --cost "$original_root/exp/module_arch/out_v1_20261018/cost.json" \
  --data-file "$original_root/exp/module_opt/data_v1_20261018/confirmation.jsonl" \
  --output "$experiment_root/results" \
  > "$experiment_root/benchmark.log" 2>&1
benchmark_status=$?
"$runtime" -c 'import json,sys,time; f=open(sys.argv[1],"x"); json.dump({"returncode":int(sys.argv[2]),"finished_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"analysis_required":True},f,indent=2); f.write("\n"); f.close()' "$experiment_root/PIPELINE_EXIT.json" "$benchmark_status"
exit "$benchmark_status"
