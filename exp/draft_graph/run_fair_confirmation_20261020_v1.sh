#!/usr/bin/env bash
# Fresh confirmation only. Never resume data, fit models or select a weaker baseline.
set -Eeuo pipefail
# This unexecuted candidate was superseded before any fresh data access.
# Six-warm configuration selection does not establish the strongest challenger
# for32warm. Re-audit/rewrite this launcher only after horizon-matched selection.
printf '%s\n' 'DISABLED: pending horizon-matched baseline selection; no confirmation data may be opened by this draft.' >&2
exit 64
run_root=/mnt/8T/cl/rescue_experiment_cache/fair_confirmation_20261020_v1
baseline_root=/mnt/8T/cl/rescue_experiment_cache/baseline_selection_20261020_v1
old_repository=/home/wanghl/cl/RESCUE
run_python=/mnt/8T/cl/rescue_experiment_cache/specextend_match_20261020_v1/venv/bin/python
expected_gpu=GPU-9db4f7b5-1447-9a2a-5c1b-80379d909fd9
test -d "$run_root/after"
test -x "$run_python"
test ! -e "$run_root/RUN_STARTED.json"
test ! -e "$run_root/final_protocol.json"
test ! -e "$run_root/data_gate"
test ! -e "$run_root/results"
exec 200>"$run_root/pipeline.lock"
flock -n 200 || exit 73
export CUDA_VISIBLE_DEVICES="$expected_gpu"
export PYTHONHASHSEED=20261024
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH
cd "$run_root/after"
verify_staged() {
  "$run_python" -c 'import hashlib,json,pathlib; root=pathlib.Path("/mnt/8T/cl/rescue_experiment_cache/fair_confirmation_20261020_v1"); expected=json.loads((root/"LAUNCH_MANIFEST.json").read_text())["source_sha256"]; assert expected; actual={p:hashlib.sha256((root/"after"/p).read_bytes()).hexdigest() for p in expected}; assert actual==expected,"Staged source changed"; print("FROZEN_SOURCE_VERIFIED",len(actual),flush=True)'
}
verify_staged
test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)" = "$expected_gpu"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
"$run_python" -c 'import json,pathlib; p=pathlib.Path("/mnt/8T/cl/rescue_experiment_cache/baseline_selection_20261020_v1/PIPELINE_EXIT.json"); assert json.loads(p.read_text())["exit_code"]==0'
"$run_python" -c 'import datetime,json,pathlib; p=pathlib.Path("/mnt/8T/cl/rescue_experiment_cache/fair_confirmation_20261020_v1/RUN_STARTED.json"); p.open("x").write(json.dumps({"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"scope":"Fresh16 same-Qwen confirmation; data/source gate precedes first candidate access; local analysis required"})+"\n")'
finish() {
  pipeline_rc=$?
  trap - EXIT
  "$run_python" -c 'import datetime,json,pathlib,sys; p=pathlib.Path("/mnt/8T/cl/rescue_experiment_cache/fair_confirmation_20261020_v1/PIPELINE_EXIT.json"); p.open("x").write(json.dumps({"exit_code":int(sys.argv[1]),"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"analysis_required_locally":True})+"\n")' "$pipeline_rc"
  exit "$pipeline_rc"
}
trap finish EXIT

# No rendering/report tests here: server lacks Matplotlib. All local report
# tests must already pass before the transport source manifest is sealed.
for test_name in test_rescue_draft_graph_primitives.py test_rescue_draft_runtime.py test_rescue_baseline_runtime.py test_rescue_baseline_selection.py test_rescue_fair_confirmation_data.py test_rescue_fair_confirmation.py; do
  test -f "tests/$test_name"
  "$run_python" -m unittest discover -s tests -p "$test_name" -v
done >"$run_root/gpu_tests.log" 2>&1
"$run_python" -c 'import pyarrow.parquet,torch; from rescue.baseline_selection import flash_receipt; assert torch.cuda.device_count()==1; print(flash_receipt())' >"$run_root/dependency_check.log" 2>&1
verify_staged
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"

"$run_python" -c 'from pathlib import Path; from rescue.fair_confirmation_data import validate_baseline_freeze,make_final_protocol,write_json; r=Path("/mnt/8T/cl/rescue_experiment_cache/fair_confirmation_20261020_v1"); b=Path("/mnt/8T/cl/rescue_experiment_cache/baseline_selection_20261020_v1"); validated=validate_baseline_freeze(b/"results",b/"results/analysis_final_v1",b/"VERIFIED_COMPLETE.json",b/"FINAL_ARTIFACTS_MANIFEST.json"); write_json(r/"final_protocol.json",make_final_protocol(r/"after/exp/draft_graph/fair_confirmation_20261020_preregistration.json",validated))'
metadata_args=(
  --metadata-file "$old_repository/exp/long_system/data/train.jsonl"
  --metadata-file "$old_repository/exp/long_system/data/development.jsonl"
  --metadata-file "$old_repository/exp/long_system/data/test.jsonl"
  --metadata-file "$old_repository/exp/module_opt/data_v1_20261018/fit.jsonl"
  --metadata-file "$old_repository/exp/module_opt/data_v1_20261018/tune.jsonl"
  --metadata-file "$old_repository/exp/module_opt/data_v1_20261018/warmup.jsonl"
  --metadata-file "$old_repository/exp/module_opt/data_v1_20261018/confirmation.jsonl"
  --metadata-file "$old_repository/exp/module_arch/data_v1_20261018/fit.jsonl"
  --metadata-file "$old_repository/exp/module_arch/data_v1_20261018/tune.jsonl"
  --metadata-file "$old_repository/exp/module_arch/data_v1_20261018/warmup.jsonl"
  --metadata-file "$old_repository/exp/module_arch/data_v1_20261018/confirmation.jsonl"
)
"$run_python" -m rescue.fair_confirmation_data freeze \
  --output "$run_root/data_gate" \
  --baseline-results "$baseline_root/results" \
  --analysis-dir "$baseline_root/results/analysis_final_v1" \
  --verified-complete "$baseline_root/VERIFIED_COMPLETE.json" \
  --final-manifest "$baseline_root/FINAL_ARTIFACTS_MANIFEST.json" \
  --raw-source-manifest "$old_repository/exp/module_arch/data_v1_20261018/manifest.json" \
  --raw-source /mnt/8T/cl/rescue_experiment_cache/long_system_v1/raw/validation-00000-of-00001.parquet \
  --tokenizer-root /mnt/8T/cl/models/Qwen2-7B-Instruct \
  --source-root "$run_root/after" \
  --final-protocol "$run_root/final_protocol.json" "${metadata_args[@]}" \
  >"$run_root/data_preparation.log" 2>&1
"$run_python" -m rescue.fair_confirmation_data prepare --gate "$run_root/data_gate/GATE_SOURCE_FREEZE.json" >>"$run_root/data_preparation.log" 2>&1
"$run_python" -m rescue.fair_confirmation_data verify --data-root "$run_root/data_gate/data" >>"$run_root/data_preparation.log" 2>&1
verify_staged
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
"$run_python" -m rescue.fair_confirmation \
  --teacher /mnt/8T/cl/models/Qwen2-7B-Instruct \
  --student /mnt/8T/cl/models/Qwen2-1.5B-Instruct \
  --mapper "$old_repository/exp/module_arch/out_v1_20261018/full_final.pt" \
  --policy "$old_repository/exp/module_arch/out_v1_20261018/budget_selected.pt" \
  --cost "$old_repository/exp/module_arch/out_v1_20261018/cost.json" \
  --freeze-receipt "$run_root/data_gate/GATE_SOURCE_FREEZE.json" \
  --dataset-manifest "$run_root/data_gate/data/DATASET_MANIFEST.json" \
  --data-file "$run_root/data_gate/data/confirmation.jsonl" \
  --prime-file "$old_repository/exp/module_opt/data_v1_20261018/tune.jsonl" \
  --output "$run_root/results" >"$run_root/confirmation.log" 2>&1
