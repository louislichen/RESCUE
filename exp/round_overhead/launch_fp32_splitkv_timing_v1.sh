#!/bin/bash
# One isolated792-call matched development timing; no retries or cleanup.
set -euo pipefail
umask 077
if [[ "$#" != 1 || "$1" != "--execute-timing" ]]; then
  printf '%s\n' 'Not launched. Explicit --execute-timing is required.'
  exit 64
fi
ROOT=/mnt/8T/cl/rescue_experiment_cache/fp32_splitkv_timing_20261021_v1
PYTHON=/mnt/8T/cl/rescue_experiment_cache/specextend_match_20261020_v1/venv/bin/python
cd "$ROOT/capsule"
sha256sum -c "$ROOT/SOURCE_SHA256SUMS"
printf '%s  %s\n' 8119915f96beb2ae0797f1b92434b15f4a0f4a55900297fe3c713c3f1e5a5a3d "$PYTHON" | sha256sum -c -
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader)" = GPU-9db4f7b5-1447-9a2a-5c1b-80379d909fd9
"$PYTHON" -B - "$ROOT" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
with (root/'LAUNCH_STARTED.json').open('x') as handle:
    json.dump(dict(status='MATCHED_TIMING_STARTED', started_at_utc=datetime.now(timezone.utc).isoformat(),
        expected_generation_calls=792, expected_private_owners=24, expected_warm_calls=768,
        frozen_mapper_policy_cost=True, both_methods_same_candidate=True, goal_complete=False), handle, indent=2)
    handle.write('\n')
PY
if env CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR="$ROOT/cache" CUDA_CACHE_DISABLE=1 \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
    "$PYTHON" -B "$ROOT/capsule/exp/round_overhead/fp32_splitkv_timing_v1.py" \
      --execute-timing --output "$ROOT/results" --cache-dir "$ROOT/cache" \
      --preregistration "$ROOT/PREREGISTRATION.json" \
      --prior-seal-root "$ROOT/inputs/prior_local_seal" \
      --model-terminal /mnt/8T/cl/rescue_experiment_cache/fp32_splitkv_model_20261021_v1/results/TERMINAL.json \
      --model-terminal-sha256 ae9f5870b37097699c791ee1fa139bc9de7fe0f3b50ab70d1666fd2deb9c8993 \
      --terminal-barrier "$ROOT/inputs/MODEL_TERMINAL_BARRIER.json" \
      --terminal-barrier-sha256 d7646b2f8e05550be7929574d0257f6dad7e531223406c69f127b1650841bd83 \
      >"$ROOT/timing.log" 2>&1; then
  timing_exit=0
else
  timing_exit=$?
fi
"$PYTHON" -B - "$ROOT" "$timing_exit" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
root, code = Path(sys.argv[1]), int(sys.argv[2])
terminal = root/'results/TERMINAL.json'
with (root/'PIPELINE_EXIT.json').open('x') as handle:
    json.dump(dict(status='MATCHED_TIMING_PIPELINE_EXIT', exit_code=code,
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
        terminal_sha256=hashlib.sha256(terminal.read_bytes()).hexdigest() if terminal.is_file() else None,
        preserve_partial=True, automatic_retry=False, analysis_required=True, goal_complete=False), handle, indent=2)
    handle.write('\n')
PY
exit "$timing_exit"
