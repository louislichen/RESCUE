#!/usr/bin/env bash
set -u
set -o pipefail
experiment_root=/mnt/8T/cl/rescue_experiment_cache/draft_graph_20261019_v1
runtime=/home/wanghl/miniconda3/envs/rescue/bin/python
test -d "$experiment_root/after/rescue" || exit 10
test ! -e "$experiment_root/GPU_TEST_EXIT.json" || exit 11
exec 9>"$experiment_root/gpu_test.lock"
flock -n 9 || exit 12
cd "$experiment_root/after" || exit 13
export CUDA_VISIBLE_DEVICES=GPU-9db4f7b5-1447-9a2a-5c1b-80379d909fd9
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=tests:.
"$runtime" -c 'import subprocess; active=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip(); assert not active, "GPU is occupied; no other job was interrupted"' || exit 14
"$runtime" -m unittest -v test_rescue_draft_graph_primitives test_rescue_draft_runtime
test_status=$?
"$runtime" -c 'import json,sys; f=open(sys.argv[1],"x"); json.dump({"returncode":int(sys.argv[2]),"scope":"Toy correctness tests, not a pretrained-model speed result"},f,indent=2); f.write("\n"); f.close()' \
  "$experiment_root/GPU_TEST_EXIT.json" "$test_status"
exit "$test_status"
