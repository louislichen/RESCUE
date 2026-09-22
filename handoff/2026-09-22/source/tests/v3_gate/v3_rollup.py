"""§12 GATE ROLL-UP — RESCUE-PRE-EXPERIMENT-CORRECTNESS-GATE-V3-001.

Combines the frozen harness gate results with the supplementary §4-decode / §6 / §10
evidence and applies the §12 rule: all ten required gates must PASS.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


def load(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def status_of(node, default="MISSING"):
    if not isinstance(node, dict):
        return default
    return node.get("status", default)


def verdict_of(node, default="MISSING"):
    """Gate artefacts that report `verdict` instead of `status` (identity preflight)."""
    if not isinstance(node, dict):
        return default
    return node.get("verdict") or node.get("status") or default


def combine(*statuses):
    if any(s != "PASS" for s in statuses):
        return "FAIL"
    return "PASS"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frozen-runner-out", required=True)
    ap.add_argument("--manifest-sha256", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    frozen_dir = Path(args.frozen_runner_out)

    identity = load(out / "v3_identity_preflight.json")
    gpu_pre = load(out / "v3_gpu_preflight.json")
    ext_decode = load(out / "v3_empty_hook_decode.json")
    intervention = load(out / "v3_intervention_effect.json")
    integrity = load(out / "v3_result_integrity.json")
    frozen = load(frozen_dir / "pre_experiment_gate.json")

    gpu_smokes = ((frozen or {}).get("stages") or {}).get("gpu_smokes") or {}

    def g(name):
        return status_of(gpu_smokes.get(name))

    required = {
        "identity": verdict_of(identity),
        "runtime_signature": g("signature"),
        "empty_hook": combine(g("empty_hook"), status_of(ext_decode)),
        "cache_lifecycle": g("cache_lifecycle"),
        "intervention_effect": status_of(intervention),
        "sequence_gradient": g("gradient"),
        "numerics": g("numerics"),
        "paired_isolation": combine(g("arm_isolation"), g("stock_purity")),
        "result_integrity": status_of(integrity),
        "k4_correctness": g("k4"),
    }
    supporting = {
        "cpu_regression": status_of(((frozen or {}).get("stages") or {}).get("cpu_regression")),
        "checkpoint_identity_contract_runner":
            status_of(((frozen or {}).get("stages") or {}).get("checkpoint_identity")),
        "targeted_fix_regressions":
            status_of(((frozen or {}).get("stages") or {}).get("targeted_regressions")),
        "gpu_resource_preflight": status_of(gpu_pre),
    }
    failing = sorted(k for k, v in required.items() if v != "PASS")
    supporting_failing = sorted(k for k, v in supporting.items() if v != "PASS")

    report = {
        "task_id": "RESCUE-PRE-EXPERIMENT-CORRECTNESS-GATE-V3-001",
        "gate_version": 3,
        "harness": "FROZEN_V3",
        "generated_at": datetime.now(timezone(timedelta(hours=8)))
        .strftime("%Y-%m-%d %H:%M:%S %z"),
        "harness_manifest_sha256": args.manifest_sha256,
        "required_gates": required,
        "supporting_checks": supporting,
        "failing_gates": failing,
        "failing_supporting_checks": supporting_failing,
        "pre_experiment_correctness_gate": "PASS" if not failing else "FAIL",
        "gpu_smoke_detail": {k: v for k, v in gpu_smokes.items()},
        "identity": identity,
        "gpu_resource": gpu_pre,
        "evidence": {
            "identity_preflight": str(out / "v3_identity_preflight.json"),
            "gpu_resource_preflight": str(out / "v3_gpu_preflight.json"),
            "frozen_runner_summary": str(frozen_dir / "pre_experiment_gate.json"),
            "frozen_runner_logs": str(frozen_dir),
            "empty_hook_decode": str(out / "v3_empty_hook_decode.json"),
            "intervention_effect": str(out / "v3_intervention_effect.json"),
            "result_integrity": str(out / "v3_result_integrity.json"),
        },
    }
    (out / "pre_experiment_gate_v3.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pre_experiment_correctness_gate":
                      report["pre_experiment_correctness_gate"],
                      "required_gates": required, "failing": failing,
                      "supporting_failing": supporting_failing}, ensure_ascii=False, indent=2))
    return 0 if not failing else 1


if __name__ == "__main__":
    raise SystemExit(main())
