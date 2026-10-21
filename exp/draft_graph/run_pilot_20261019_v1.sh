#!/usr/bin/env bash
# One registered development document only; not a held-out/full-system claim.
set -u
set -o pipefail
experiment_root=/mnt/8T/cl/rescue_experiment_cache/draft_graph_20261019_v1
original_root=/home/wanghl/cl/RESCUE
runtime=/home/wanghl/miniconda3/envs/rescue/bin/python
test -d "$experiment_root/after/rescue" || exit 10
test ! -e "$experiment_root/pilot" || exit 11
test ! -e "$experiment_root/PILOT_EXIT.json" || exit 12
exec 9>"$experiment_root/pilot.lock"
flock -n 9 || exit 13
"$runtime" -c 'import json,pathlib,hashlib,sys; p=pathlib.Path(sys.argv[1]); assert json.loads((p/"GPU_TEST_EXIT.json").read_text())["returncode"]==0; m=json.loads((p/"GPU_TEST_SOURCE_MANIFEST.json").read_text()); assert all(hashlib.sha256((p/"after"/k).read_bytes()).hexdigest()==v for k,v in m["file_sha256"].items()),"Previously tested code changed; stop without restarting"' "$experiment_root" || exit 14
cd "$experiment_root/after" || exit 15
export CUDA_VISIBLE_DEVICES=GPU-9db4f7b5-1447-9a2a-5c1b-80379d909fd9
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
"$runtime" -u -m rescue.draft_graph_smoke \
  --baseline-root "$experiment_root/before" \
  --teacher /mnt/8T/cl/models/Qwen2-7B-Instruct \
  --student /mnt/8T/cl/models/Qwen2-1.5B-Instruct \
  --mapper "$original_root/exp/module_arch/out_v1_20261018/full_final.pt" \
  --policy "$original_root/exp/module_arch/out_v1_20261018/budget_selected.pt" \
  --cost "$original_root/exp/module_arch/out_v1_20261018/cost.json" \
  --data-file "$original_root/exp/module_opt/data_v1_20261018/confirmation.jsonl" \
  --output "$experiment_root/pilot"
pilot_status=$?
"$runtime" -c 'import json,sys,time; f=open(sys.argv[1],"x"); json.dump({"returncode":int(sys.argv[2]),"finished_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"analysis_required":True},f,indent=2); f.write("\n"); f.close()' \
  "$experiment_root/PILOT_EXIT.json" "$pilot_status"
exit "$pilot_status"
