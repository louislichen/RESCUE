#!/usr/bin/env bash
# New v2 execution only; v1 remains a read-only, hard-disabled source capsule.
# Initial v2 contains only tools/ and EXECUTION_PLAN.json. Do not pre-create
# v2/pipeline.log via shell redirection; use tmux stdout or an external stage log.
set -Eeuo pipefail
run_root=/mnt/8T/cl/rescue_experiment_cache/fair_union_confirmation_20261020_v2
run_python=/mnt/8T/cl/rescue_experiment_cache/specextend_match_20261020_v1/venv/bin/python
if [[ $# -gt 1 || ( $# -eq 1 && $1 != --validate-only ) ]]; then
  printf '%s\n' 'Usage: run_fair_union_confirmation_20261020_v2.sh [--validate-only]' >&2
  exit 64
fi
export CUDA_VISIBLE_DEVICES=GPU-9db4f7b5-1447-9a2a-5c1b-80379d909fd9
export PYTHONHASHSEED=20261026
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONOPTIMIZE
exec "$run_python" "$run_root/tools/fair_union_execution.py" --plan "$run_root/EXECUTION_PLAN.json" "$@"
