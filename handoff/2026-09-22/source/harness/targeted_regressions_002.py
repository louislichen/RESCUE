"""Run the targeted regressions and emit the required evidence files.

  checkpoint_student_identity_regression.json   (FIX-A, A1..A5)
  training_config_identity_regression.json      (FIX-B, B1..B7)
  completion_integrity_regression.json          (FIX-C, C1..C8)
  training_config_schema_regression.json        (FIX-D, D1..D10)
  provenance_value_match_regression.json        (FIX-E, E0..E6 + E2E)
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rescue_harness import harness as H
from rescue_harness import regression_scenarios as R
OUT_FILES = {
    "A": "checkpoint_student_identity_regression.json",
    "B": "training_config_identity_regression.json",
    "C": "completion_integrity_regression.json",
    "D": "training_config_schema_regression.json",
    "E": "provenance_value_match_regression.json",
}
def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("out", nargs="?", default=str(HERE/"results"))
    ap.add_argument("--artifacts-json", default=""); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    artifacts=(json.loads(Path(args.artifacts_json).read_text(encoding="utf-8")) if args.artifacts_json else None)
    now=datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S %z")
    summary={"task_id":"RESCUE-HARNESS-TRAINING-IDENTITY-FINAL-CLOSURE-003","generated_at":now,"fixes":{},"status":"PASS"}
    for key,fn in R.ALL.items():
        report=fn(artifacts) if key=="E" else fn()
        report["generated_at"]=now
        report["harness_sha256"]=H.sha256_file(HERE/"rescue_harness"/"harness.py")
        report["training_identity_sha256"]=H.sha256_file(HERE/"rescue_harness"/"training_identity.py")
        H.atomic_write_text(out/OUT_FILES[key], json.dumps(report,ensure_ascii=False,indent=2)+"\n")
        summary["fixes"][key]={"file":OUT_FILES[key],"status":report["status"],
          "scenarios":{r["id"]:r["status"] for r in report["scenarios"]}}
        if report["status"]=="FAIL": summary["status"]="FAIL"
        elif report["status"]=="BLOCKED" and summary["status"]=="PASS": summary["status"]="PARTIAL"
    H.atomic_write_text(out/"targeted_regressions_summary.json",json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    return 1 if summary["status"]=="FAIL" else 0
if __name__=="__main__": raise SystemExit(main())
