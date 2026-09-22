"""PRE_EXPERIMENT_CORRECTNESS_GATE — run this before any formal RESCUE experiment.

It executes, in order:
  1. the CPU regression suite (runtime contract, gradient fail-closed, prior identity,
     checkpoint identity, denominator, output integrity, dtype policy, arm isolation);
  2. the real-model GPU smokes (signature, empty hook, cache lifecycle, arm isolation,
     stock purity, sequence gradient, numerics, K4 correctness);
  3. the checkpoint identity contract for the checkpoint actually in use;
  4. a summary with one PASS/FAIL per required gate.

Exit code 0 only if every required gate passes.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rescue_harness import harness as H  # noqa: E402

REQUIRED_GATES = [
    ("runtime_api", "signature"),
    ("stock_vs_empty_hook", "empty_hook"),
    ("cache_lifecycle", "cache_lifecycle"),
    ("arm_isolation", "arm_isolation"),
    ("stock_purity", "stock_purity"),
    ("sequence_task_gradient", "gradient"),
    ("numerics", "numerics"),
    ("k4_correctness_smoke", "k4"),
]


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--source-lock", required=True)
    ap.add_argument("--k4-dir", required=True)
    ap.add_argument("--teacher-model", required=True)
    ap.add_argument("--student-model", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--validation-ledger", required=True)
    ap.add_argument("--validation-prior-cache", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--training-config", default="")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--skip-gpu", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary = {"task_id": "RESCUE-CORRECTNESS-CLOSURE-AND-HARNESS-FREEZE-001",
               "gate_version": 1, "stages": {}, "required": {}}

    # ---- 1. CPU suite ----
    cpu = run([args.python, "-m", "unittest", "discover", "-s", "rescue_harness",
               "-p", "test_*.py"], cwd=str(HERE))
    tail = (cpu.stderr or "").strip().splitlines()[-3:]
    summary["stages"]["cpu_regression"] = {
        "returncode": cpu.returncode, "tail": tail,
        "status": "PASS" if cpu.returncode == 0 else "FAIL"}
    H.atomic_write_text(out / "cpu_regression.log",
                        (cpu.stdout or "") + (cpu.stderr or ""))

    # ---- 2. checkpoint identity contract (real files) ----
    # FIX-B: a missing/placeholder training config must FAIL this gate, not silently
    # produce a weak identity.  Supply a canonical config file built from the real
    # frozen artefacts (see make_run_provenance.py).
    ident_status, ident_detail = "FAIL", {}
    try:
        contract = H.checkpoint_identity_contract(
            teacher_model_path=Path(args.teacher_model),
            student_model_path=Path(args.student_model),
            source_lock=Path(args.source_lock),
            training_config=(Path(args.training_config) if args.training_config
                             else "training_config not supplied"),
            checkpoint=Path(args.checkpoint),
            validation_ledger=Path(args.validation_ledger) if args.validation_ledger else None,
            run_source_dirs=[Path(p) for p in str(args.source_dir).split(",") if p.strip()],
        )
        H.atomic_write_text(out / "checkpoint_identity_contract.json",
                            json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
        ident_status = "PASS"
        ident_detail = {
            "teacher_weights_manifest_sha256": contract["teacher_weights_manifest_sha256"][:16],
            "student_weights_manifest_sha256": contract["student_weights_manifest_sha256"][:16],
            "student_config_sha256": contract["student_config_sha256"][:16],
            "student_tokenizer_sha256": (contract["student_tokenizer_sha256"] or "")[:16],
            "source_lock_manifest_sha256": contract["source_lock_manifest_sha256"][:16],
            "projection_code_sha256": contract["projection_code_sha256"][:16],
            "training_config_sha256": contract["training_config_sha256"][:16],
            "checkpoint_sha256": (contract["checkpoint_sha256"] or "")[:16],
        }
    except H.CheckpointIdentityIncomplete as exc:
        ident_detail = {"reason": str(exc)}
    except H.ProvenanceError as exc:
        ident_detail = {"reason": f"{type(exc).__name__}: {exc}"}
    summary["stages"]["checkpoint_identity"] = {"status": ident_status, **ident_detail}

    # ---- 3. targeted FIX-A/B/C regressions ----
    reg = run([args.python, "-u", str(HERE / "targeted_regressions_002.py"),
               str(out / "targeted")], cwd=str(HERE))
    H.atomic_write_text(out / "targeted_regressions.log",
                        (reg.stdout or "") + (reg.stderr or ""))
    reg_summary = {}
    for name in ("checkpoint_student_identity_regression.json",
                 "training_config_identity_regression.json",
                 "completion_integrity_regression.json"):
        p = out / "targeted" / name
        reg_summary[name] = (json.loads(p.read_text(encoding="utf-8"))["status"]
                             if p.exists() else "MISSING")
    summary["stages"]["targeted_regressions"] = {
        "returncode": reg.returncode, "status": "PASS" if reg.returncode == 0 else "FAIL",
        "reports": reg_summary}

    # ---- 4. GPU smokes ----
    gpu_results = {}
    if args.skip_gpu:
        for _, gate in REQUIRED_GATES:
            gpu_results[gate] = {"status": "BLOCKED", "detail": {"reason": "--skip-gpu"}}
    else:
        common = ["--source-dir", args.source_dir, "--source-lock", args.source_lock,
                  "--k4-dir", args.k4_dir, "--teacher-model", args.teacher_model,
                  "--student-model", args.student_model, "--dataset-path", args.dataset_path,
                  "--validation-ledger", args.validation_ledger,
                  "--validation-prior-cache", args.validation_prior_cache,
                  "--checkpoint", args.checkpoint, "--samples", str(args.samples),
                  "--out", str(out)]
        for _, gate in REQUIRED_GATES:
            proc = run([args.python, "-u", str(HERE / "rescue_harness" / "gpu_smokes.py"),
                        "--gate", gate] + common, cwd=str(HERE))
            H.atomic_write_text(out / f"gpu_{gate}.log", (proc.stdout or "") + (proc.stderr or ""))
            last = [l for l in (proc.stdout or "").strip().splitlines() if l.startswith("{")]
            try:
                gpu_results[gate] = json.loads(last[-1]) if last else {
                    "gate": gate, "status": "FAIL", "detail": {"reason": "no json emitted"}}
            except Exception as exc:
                gpu_results[gate] = {"gate": gate, "status": "FAIL",
                                     "detail": {"reason": f"unparseable: {exc}"}}
            print(json.dumps(gpu_results[gate], ensure_ascii=False), flush=True)
    summary["stages"]["gpu_smokes"] = gpu_results

    # ---- 5. required gate roll-up ----
    for name, gate in REQUIRED_GATES:
        status = gpu_results.get(gate, {}).get("status", "BLOCKED")
        summary["required"][name] = status
    summary["required"]["checkpoint_identity"] = summary["stages"]["checkpoint_identity"]["status"]
    summary["required"]["targeted_fix_regressions"] = (
        summary["stages"]["targeted_regressions"]["status"])
    summary["required"]["prior_identity_mismatch_rejection"] = (
        summary["stages"]["cpu_regression"]["status"])
    summary["required"]["denominator_audit"] = summary["stages"]["cpu_regression"]["status"]
    summary["required"]["output_atomicity"] = summary["stages"]["cpu_regression"]["status"]
    summary["required"]["gradient_fail_closed"] = summary["stages"]["cpu_regression"]["status"]

    failing = sorted(k for k, v in summary["required"].items() if v != "PASS")
    summary["failing_gates"] = failing
    summary["verdict"] = ("CORRECTNESS_CLOSURE_PASS" if not failing else
                          ("CORRECTNESS_CLOSURE_PARTIAL" if len(failing) < len(summary["required"])
                           else "CORRECTNESS_CLOSURE_BLOCKED"))
    summary["harness_freeze_candidate"] = "YES" if not failing else "NO"
    H.atomic_write_text(out / "pre_experiment_gate.json",
                        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"verdict": summary["verdict"], "failing": failing,
                      "freeze_candidate": summary["harness_freeze_candidate"]}, indent=2))
    return 0 if not failing else 1


if __name__ == "__main__":
    raise SystemExit(main())
