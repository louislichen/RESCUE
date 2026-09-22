"""§2 GPU RESOURCE PREFLIGHT — RESCUE-PRE-EXPERIMENT-CORRECTNESS-GATE-V3-001.

Read-only.  Verifies 4 x RTX 2080 Ti are present and that no residual training or
inference process is holding them, and records the CUDA-visible-device mapping.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

EXPECTED_COUNT = 4
EXPECTED_NAME = "NVIDIA GeForce RTX 2080 Ti"
#: a real training/inference worker would show up as one of these; anything smaller is
#: the preflight's own bookkeeping, not a residual run.
RESIDUAL_PATTERNS = ("sequence_prefix_runner", "real_k4_runner", "window_k4", "window_qa",
                     "window_pair", "train.py", "q3_teacher_prior_worker")


def sh(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    payload: dict = {"stage": "gpu_resource_preflight",
                     "generated_at": datetime.now(timezone(timedelta(hours=8)))
                     .strftime("%Y-%m-%d %H:%M:%S %z"),
                     "errors": []}

    rc, txt = sh(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,"
                  "memory.free,utilization.gpu,temperature.gpu",
                  "--format=csv,noheader,nounits"])
    payload["nvidia_smi_returncode"] = rc
    gpus = []
    for line in txt.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 7:
            continue
        gpus.append({"index": int(parts[0]), "name": parts[1],
                     "memory_total_mib": int(parts[2]), "memory_used_mib": int(parts[3]),
                     "memory_free_mib": int(parts[4]), "utilization_pct": int(parts[5]),
                     "temperature_c": int(parts[6])})
    payload["gpus"] = gpus
    payload["gpu_count"] = len(gpus)
    payload["all_expected_model"] = all(g["name"] == EXPECTED_NAME for g in gpus) and \
        len(gpus) == EXPECTED_COUNT

    rc2, apps_txt = sh(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                        "--format=csv,noheader"])
    apps = [a.strip() for a in apps_txt.strip().splitlines() if a.strip()]
    payload["compute_apps"] = apps

    rc3, ps_txt = sh(["ps", "-eo", "pid,etime,args"])
    residual = [l.strip() for l in ps_txt.splitlines()
                if any(pat in l for pat in RESIDUAL_PATTERNS) and "preflight" not in l]
    payload["residual_process_matches"] = residual

    payload["cuda_visible_devices_env"] = args.cuda_visible_devices
    payload["cuda_visible_device_mapping"] = (
        list(range(len(gpus))) if args.cuda_visible_devices in ("", None)
        else [int(x) for x in str(args.cuda_visible_devices).split(",") if x.strip()])

    if payload["gpu_count"] != EXPECTED_COUNT:
        payload["errors"].append(f"GPU_COUNT_MISMATCH: {payload['gpu_count']}")
    if not payload["all_expected_model"]:
        payload["errors"].append("GPU_MODEL_MISMATCH")
    if apps:
        payload["errors"].append(f"RESIDUAL_COMPUTE_APPS: {apps}")
    if residual:
        payload["errors"].append(f"RESIDUAL_PROCESSES: {len(residual)}")
    payload["idle"] = bool(gpus) and all(g["utilization_pct"] == 0 for g in gpus)
    payload["free_vram_mib_total"] = sum(g["memory_free_mib"] for g in gpus)
    payload["status"] = "PASS" if not payload["errors"] else "FAIL"
    payload["errors"] = sorted(set(payload["errors"]))

    Path(out / "v3_gpu_preflight.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gate": "gpu_preflight", "status": payload["status"],
                      "gpu_count": payload["gpu_count"],
                      "models": sorted({g["name"] for g in gpus}),
                      "free_vram_mib_total": payload["free_vram_mib_total"],
                      "idle": payload["idle"],
                      "errors": payload["errors"]}, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
