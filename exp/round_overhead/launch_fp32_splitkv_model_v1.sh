#!/bin/bash
# Isolated 48-call model Token gate; no performance acceptance or retry.
set -euo pipefail
umask 077
if [[ "$#" != 1 || "$1" != "--execute-model" ]]; then
  printf '%s\n' 'Not launched. Explicit --execute-model is required.'
  exit 64
fi
ROOT=/mnt/8T/cl/rescue_experiment_cache/fp32_splitkv_model_20261021_v1
PYTHON=/mnt/8T/cl/rescue_experiment_cache/specextend_match_20261020_v1/venv/bin/python
cd "$ROOT/capsule"
sha256sum -c "$ROOT/SOURCE_SHA256SUMS"
printf '%s  %s\n' 8119915f96beb2ae0797f1b92434b15f4a0f4a55900297fe3c713c3f1e5a5a3d "$PYTHON" | sha256sum -c -
"$PYTHON" -B - "$ROOT" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
with (root/'LAUNCH_STARTED.json').open('x') as handle:
    json.dump(dict(status='MODEL_TOKEN_GATE_STARTED', started_at_utc=datetime.now(timezone.utc).isoformat(),
        expected_model_calls=48, performance_measured=False, goal_complete=False,
        intermediate_bitwise_required=False, evaluated_target_tokens_exact_required=True,
        frozen_mapper_policy_cost=True, both_arms_same_candidate=True), handle, indent=2)
    handle.write('\n')
PY
if env CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR="$ROOT/cache" CUDA_CACHE_DISABLE=1 \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
    "$PYTHON" -B "$ROOT/capsule/exp/round_overhead/fp32_splitkv_model_gate_v1.py" \
      --execute-model --output "$ROOT/results" --cache-dir "$ROOT/cache" \
      --preregistration "$ROOT/PREREGISTRATION.json" \
      --prior-seal-root "$ROOT/inputs/prior_local_seal" \
      --operator-terminal /mnt/8T/cl/rescue_experiment_cache/fp32_splitkv_operator_20261021_v3/results/TERMINAL.json \
      --operator-terminal-sha256 184674b6a991601378fb3e10ee4b7b4125477efad1400fd21599a5ac98e8c7b6 \
      >"$ROOT/model_gate.log" 2>&1; then
  gate_exit=0
else
  gate_exit=$?
fi
"$PYTHON" -B - "$ROOT" "$gate_exit" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
root, code = Path(sys.argv[1]), int(sys.argv[2])
terminal = root/'results/TERMINAL.json'
with (root/'PIPELINE_EXIT.json').open('x') as handle:
    json.dump(dict(status='MODEL_TOKEN_PIPELINE_EXIT', exit_code=code,
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
        terminal_sha256=hashlib.sha256(terminal.read_bytes()).hexdigest() if terminal.is_file() else None,
        preserve_partial=True, automatic_retry=False, performance_measured=False, goal_complete=False), handle, indent=2)
    handle.write('\n')
PY
exit "$gate_exit"
