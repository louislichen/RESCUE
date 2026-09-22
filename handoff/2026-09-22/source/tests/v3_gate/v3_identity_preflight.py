"""§1 IDENTITY PREFLIGHT — RESCUE-PRE-EXPERIMENT-CORRECTNESS-GATE-V3-001.

Read-only, CPU only. Proves before GPU work that frozen V3 harness, source-lock,
checkpoint, model sides and training-config identity are the accepted artefacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

FROZEN_MANIFEST_NAME = "HARNESS_FREEZE_CANDIDATE_MANIFEST_V3.json"
FROZEN_MANIFEST_SHA256 = "0e569ea84d36ae91d7ad73dd7299b60c9dcab1ff0abb14e270f447449ff43788"
EXPECTED_SOURCE_LOCK = {
    "adapter.py": "7253fa9fb1e5f7d513afdbc7d8863cb5113fb98d90f79f77284b3879f4a32722",
    "common.py": "e8f7361c766bd3ee5dd05aa01630eb51c21cb890e795c093f366bea800278581",
    "steering.py": "24eaeed620e3a04702ab7b61b1d0b5756d588dbc1b54ae444710a79e2cfb5791",
}
EXPECTED_CONTRACT = {
    "teacher_config_sha256": "d318b49c4a29771e259d93053084eb752f5bfe69c20c4f2e041bd1017884f810",
    "teacher_weights_manifest_sha256": "23dc5f5369f8349fbd7ead3e7f9b4be98bbd76fd9895ef6a205f4a49b045a98d",
    "student_config_sha256": "f8c6c681f7c28386e333775bb78de28f9de3ce6a676059dc35792022adcb7e36",
    "student_weights_manifest_sha256": "5f945a6025ee13b0db77fd67ef8e71938bb9ded3e52a752f4bab5a0efb4d2757",
    "student_tokenizer_sha256": "f7c9b2dba4a296b1aa76c16a34b8225c0c118978400d4bb66bff0902d702f5b8",
    "source_lock_manifest_sha256": "713a2adaf001583a1df0335d7d979e39d798ed9dbc701ebe85b45203e47b36fe",
    "projection_code_sha256": "69b717914ee182b75a627c457583efeb1461969411320ea33576276f4e90089c",
    "checkpoint_sha256": "6532daad4e3ae54a7e76063e6aef3fef9b59ba6ef709394bf361c30228f19ace",
}
EXPECTED_TRAINING_CONFIG_SHA256 = "81dd4b1a8c2db6ffb3bd1afd6539b24d5d9a5750eebec74bc0916a1fad26413a"
EXPECTED_IDENTITY_SHA256 = "5b7a4dab400a0c8782e09124fe9bdbc802b193082de12ed8f5b118049f222cd8"
EXPECTED_RUNTIME = {"torch": "2.9.1+cu128", "transformers": "4.57.1", "cuda": "12.8", "cudnn": 91002}
CANONICAL_DTYPE = "bfloat16"
REJECTED_DTYPE = "float16"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compare(label: str, expected, observed) -> dict:
    return {"check": label, "expected": expected, "observed": observed, "match": expected == observed}


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("harness-root","source-lock","run-source-dir","checkpoint","teacher-model",
                 "student-model","train-ledger","validation-ledger","provenance-envelope",
                 "direct-training-config","recorded-contract","out"):
        ap.add_argument("--"+name, required=True)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    H = Path(args.harness_root); SL = Path(args.source_lock); out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(H))
    run_source_dirs = [Path(p) for p in str(args.run_source_dir).split(",") if p.strip()]
    payload = {"stage":"identity_preflight",
               "generated_at":datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S %z"),
               "checks":[],"errors":[]}
    def fail(msg): payload["errors"].append(msg)

    manifest_path = H / FROZEN_MANIFEST_NAME
    manifest_sha = sha256_file(manifest_path) if manifest_path.exists() else None
    payload["manifest"]={"path":str(manifest_path),"sha256":manifest_sha,
                         "expected_sha256":FROZEN_MANIFEST_SHA256,
                         "match":manifest_sha==FROZEN_MANIFEST_SHA256}
    if manifest_sha != FROZEN_MANIFEST_SHA256:
        fail(f"FROZEN_MANIFEST_SHA_MISMATCH: {manifest_sha}"); manifest={}
    else:
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))

    file_checks=[]
    for rel,expected in sorted((manifest.get("modified_file_sha256") or {}).items()):
        p=H/rel; observed=sha256_file(p) if p.exists() else None
        file_checks.append(compare(f"manifest_file:{rel}",expected,observed))
    for c in file_checks:
        if not c["match"]: fail(f"HARNESS_SOURCE_DRIFT: {c['check']}")
    payload["harness_source_checks"]={"files_checked":len(file_checks),"mismatches":[c for c in file_checks if not c["match"]]}

    lock_checks=[]
    for name,expected in sorted(EXPECTED_SOURCE_LOCK.items()):
        p=SL/name; observed=sha256_file(p) if p.exists() else None
        lock_checks.append(compare(f"source_lock:{name}",expected,observed))
    for c in lock_checks:
        if not c["match"]: fail(f"SOURCE_LOCK_DRIFT: {c['check']}")
    payload["source_lock_checks"]={"files_checked":len(lock_checks),"mismatches":[c for c in lock_checks if not c["match"]]}

    ckpt=Path(args.checkpoint); ckpt_sha=sha256_file(ckpt) if ckpt.exists() else None
    expected_ckpt=(manifest.get("checkpoint_under_test") or {}).get("sha256",EXPECTED_CONTRACT["checkpoint_sha256"])
    payload["checkpoint"]={"path":str(ckpt),"sha256":ckpt_sha,"expected_sha256":expected_ckpt,"match":ckpt_sha==expected_ckpt}
    if ckpt_sha != expected_ckpt: fail(f"CHECKPOINT_DRIFT: {ckpt_sha}")

    prov_out=out/"v3_run_provenance_080.json"; child_env=dict(os.environ); child_env["CUDA_VISIBLE_DEVICES"]=""
    gen=subprocess.run([args.python,"-u",str(H/"make_run_provenance.py"),
        "--source-lock",str(SL),"--run-root",str(ckpt.parent),"--checkpoint",str(ckpt),
        "--train-ledger",str(args.train_ledger),"--validation-ledger",str(args.validation_ledger),
        "--source-dir",str(args.run_source_dir),"--out",str(prov_out)],
        cwd=str(H),capture_output=True,text=True,env=child_env)
    regenerated_sha=None
    if gen.returncode==0 and prov_out.exists():
        regenerated=json.loads(prov_out.read_text(encoding="utf-8"))
        regenerated_sha=regenerated.get("canonical_training_config_sha256")
    payload["training_config_regeneration"]={"returncode":gen.returncode,"sha256":regenerated_sha,
       "expected_sha256":EXPECTED_TRAINING_CONFIG_SHA256,"match":regenerated_sha==EXPECTED_TRAINING_CONFIG_SHA256}
    if regenerated_sha != EXPECTED_TRAINING_CONFIG_SHA256: fail(f"TRAINING_CONFIG_DRIFT: {regenerated_sha}")

    from rescue_harness import harness as Hm
    ident={}
    try:
        contract=Hm.checkpoint_identity_contract(
            teacher_model_path=Path(args.teacher_model),student_model_path=Path(args.student_model),
            source_lock=SL,training_config=Path(args.provenance_envelope),checkpoint=ckpt,
            train_ledger=Path(args.train_ledger),validation_ledger=Path(args.validation_ledger),
            run_source_dirs=run_source_dirs)
        field_checks=[compare(f"identity:{k}",v,contract.get(k)) for k,v in sorted(EXPECTED_CONTRACT.items())]
        field_checks.append(compare("identity:training_config_sha256",EXPECTED_TRAINING_CONFIG_SHA256,contract.get("training_config_sha256")))
        field_checks.append(compare("identity:identity_sha256",EXPECTED_IDENTITY_SHA256,contract.get("identity_sha256")))
        ident={"status":"PASS","identity_sha256":contract.get("identity_sha256"),"checks":field_checks,
               "mismatches":[c for c in field_checks if not c["match"]]}
        for c in field_checks:
            if not c["match"]: fail(f"IDENTITY_FIELD_DRIFT: {c['check']}")
    except Exception as exc:
        ident={"status":"FAIL","error":f"{type(exc).__name__}: {exc}"}; fail(f"IDENTITY_CONTRACT_FAILED: {exc}")
    payload["identity_contract_envelope_route"]=ident

    import torch, transformers
    runtime={"torch":torch.__version__,"transformers":transformers.__version__,
             "cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version()}
    checks=[compare(f"runtime:{k}",v,runtime.get(k)) for k,v in EXPECTED_RUNTIME.items()]
    payload["runtime"]={"observed":runtime,"checks":checks,"mismatches":[c for c in checks if not c["match"]]}
    for c in checks:
        if not c["match"]: fail(f"RUNTIME_DRIFT: {c['check']}")

    payload["verdict"]="PASS" if not payload["errors"] else "IDENTITY_DRIFT"
    payload["errors"]=sorted(set(payload["errors"]))
    Hm.atomic_write_text(out/"v3_identity_preflight.json",json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
    return 0 if payload["verdict"]=="PASS" else 1


if __name__=="__main__":
    raise SystemExit(main())
