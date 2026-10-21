#!/bin/bash
# Fixture-only v2; v1 failure is preserved. No models or timing.
set -euo pipefail
umask 077
if [[ "$#" != 1 || "$1" != "--execute-cuda" ]]; then
  printf '%s\n' 'Not launched. Explicit --execute-cuda is required.'
  exit 64
fi
ROOT=/mnt/8T/cl/rescue_experiment_cache/fp32_splitkv_operator_20261021_v2
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
    json.dump(dict(status='OPERATOR_GATE_STARTED', fixture_version=2,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        model_calls=0, performance_measured=False, goal_complete=False,
        prior_v1_failure_preserved=True,
        human_authorized_numeric_contract='FP32 and TF32 off; intermediate last-bit differences allowed; future evaluated Target Tokens must match frozen original Target.'), handle, indent=2)
    handle.write('\n')
PY
if env CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR="$ROOT/cache" CUDA_CACHE_DISABLE=1 \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
    "$PYTHON" -B "$ROOT/capsule/exp/round_overhead/fp32_splitkv_cuda_gate_v2.py" \
      --execute-cuda --output "$ROOT/results" --cache-dir "$ROOT/cache" \
      >"$ROOT/operator.log" 2>&1; then
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
    json.dump(dict(status='OPERATOR_PIPELINE_EXIT', fixture_version=2, exit_code=code,
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
        terminal_sha256=hashlib.sha256(terminal.read_bytes()).hexdigest() if terminal.is_file() else None,
        preserve_partial=True, automatic_retry=False, model_calls=0,
        performance_measured=False, goal_complete=False), handle, indent=2)
    handle.write('\n')
PY
exit "$gate_exit"
