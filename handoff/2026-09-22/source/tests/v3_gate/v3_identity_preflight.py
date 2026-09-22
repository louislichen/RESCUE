"""§1 IDENTITY PREFLIGHT — RESCUE-PRE-EXPERIMENT-CORRECTNESS-GATE-V3-001.

Read-only, CPU only.  Proves, before the GPU is touched, that the frozen V3 harness,
the frozen source-lock, the step80 checkpoint, the Teacher/Student model sides and the
training-config identity are all exactly the artefacts the independent review accepted.

Any mismatch => verdict IDENTITY_DRIFT (the gate must stop; "close enough" is not allowed).

This file is gate evidence, NOT part of the frozen manifest; it never writes into the
frozen tree.
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

#: values recorded by the accepted V3 identity contract (results/checkpoint_identity_contract.json)
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
EXPECTED_TRAINING_CONFIG_SHA256 = \
    "81dd4b1a8c2db6ffb3bd1afd6539b24d5d9a5750eebec74bc0916a1fad26413a"
EXPECTED_IDENTITY_SHA256 = \
    "5b7a4dab400a0c8782e09124fe9bdbc802b193082de12ed8f5b118049f222cd8"
EXPECTED_RUNTIME = {"torch": "2.9.1+cu128", "transformers": "4.57.1",
                    "cuda": "12.8", "cudnn": 91002}
CANONICAL_DTYPE = "bfloat16"
REJECTED_DTYPE = "float16"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_cn() -> str:
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S %z")


def compare(label: str, expected, observed) -> dict:
    return {"check": label, "expected": expected, "observed": observed,
            "match": expected == observed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-root", required=True)
    ap.add_argument("--source-lock", required=True)
    ap.add_argument("--run-source-dir", required=True,
                    help="comma-separated run source directories that produced the run")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--teacher-model", required=True)
    ap.add_argument("--student-model", required=True)
    ap.add_argument("--train-ledger", required=True)
    ap.add_argument("--validation-ledger", required=True)
    ap.add_argument("--provenance-envelope", required=True)
    ap.add_argument("--direct-training-config", required=True)
    ap.add_argument("--recorded-contract", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    H = Path(args.harness_root)
    SL = Path(args.source_lock)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(H))
    run_source_dirs = [Path(p) for p in str(args.run_source_dir).split(",") if p.strip()]

    payload: dict = {"stage": "identity_preflight", "generated_at": now_cn(),
                     "harness_root": str(H), "run_source_dirs": [str(p) for p in run_source_dirs],
                     "checks": [], "errors": []}

    def fail(msg: str) -> None:
        payload["errors"].append(msg)

    # ---- A. the frozen manifest itself -----------------------------------
    manifest_path = H / FROZEN_MANIFEST_NAME
    manifest_sha = sha256_file(manifest_path) if manifest_path.exists() else None
    payload["manifest"] = {"name": FROZEN_MANIFEST_NAME, "path": str(manifest_path),
                           "sha256": manifest_sha, "expected_sha256": FROZEN_MANIFEST_SHA256,
                           "match": manifest_sha == FROZEN_MANIFEST_SHA256}
    if manifest_sha != FROZEN_MANIFEST_SHA256:
        fail(f"FROZEN_MANIFEST_SHA_MISMATCH: {manifest_sha}")
        manifest = {}
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # ---- B. every harness source file listed in the manifest -------------
    file_checks = []
    for rel, expected in sorted((manifest.get("modified_file_sha256") or {}).items()):
        p = H / rel
        observed = sha256_file(p) if p.exists() else None
        file_checks.append(compare(f"manifest_file:{rel}", expected, observed))
    payload["harness_source_checks"] = {
        "files_checked": len(file_checks),
        "mismatches": [c for c in file_checks if not c["match"]],
    }
    for c in file_checks:
        if not c["match"]:
            fail(f"HARNESS_SOURCE_DRIFT: {c['check']}")

    # ---- C. frozen source-lock ------------------------------------------
    lock_checks = []
    for name, expected in sorted(EXPECTED_SOURCE_LOCK.items()):
        p = SL / name
        observed = sha256_file(p) if p.exists() else None
        lock_checks.append(compare(f"source_lock:{name}", expected, observed))
        manifest_expected = (manifest.get("source_lock_sha256") or {}).get(name)
        if manifest_expected is not None and manifest_expected != expected:
            fail(f"MANIFEST_RECORD_INCONSISTENT: source_lock {name}")
    payload["source_lock_checks"] = {
        "files_checked": len(lock_checks),
        "mismatches": [c for c in lock_checks if not c["match"]],
        "match_manifest_record": all(
            (manifest.get("source_lock_sha256") or {}).get(n) == e
            for n, e in EXPECTED_SOURCE_LOCK.items()),
    }
    for c in lock_checks:
        if not c["match"]:
            fail(f"SOURCE_LOCK_DRIFT: {c['check']}")

    # ---- D. checkpoint under test ---------------------------------------
    ckpt = Path(args.checkpoint)
    ckpt_sha = sha256_file(ckpt) if ckpt.exists() else None
    expected_ckpt = (manifest.get("checkpoint_under_test") or {}).get(
        "sha256", EXPECTED_CONTRACT["checkpoint_sha256"])
    payload["checkpoint"] = {"path": str(ckpt), "sha256": ckpt_sha,
                             "expected_sha256": expected_ckpt,
                             "match": ckpt_sha == expected_ckpt}
    if ckpt_sha != expected_ckpt:
        fail(f"CHECKPOINT_DRIFT: {ckpt_sha}")

    # ---- E. training-config identity: regenerate from real artefacts -----
    # NOTE: make_run_provenance.py derives the sibling "direct" config path by replacing
    # "run_provenance" with "training_config", so the name must keep that token or the
    # envelope would be overwritten by the direct config.
    prov_out = out / "v3_run_provenance_080.json"
    child_env = dict(os.environ)
    child_env["CUDA_VISIBLE_DEVICES"] = ""
    gen = subprocess.run(
        [args.python, "-u", str(H / "make_run_provenance.py"),
         "--source-lock", str(SL), "--run-root", str(Path(args.checkpoint).parent),
         "--checkpoint", str(ckpt), "--train-ledger", str(args.train_ledger),
         "--validation-ledger", str(args.validation_ledger),
         "--source-dir", str(args.run_source_dir), "--out", str(prov_out)],
        cwd=str(H), capture_output=True, text=True, env=child_env)
    payload["training_config_regeneration"] = {
        "returncode": gen.returncode,
        "stdout_tail": (gen.stdout or "").strip().splitlines()[-8:],
        "stderr_tail": (gen.stderr or "").strip().splitlines()[-8:],
    }
    regenerated_sha = None
    if gen.returncode == 0 and prov_out.exists():
        regenerated = json.loads(prov_out.read_text(encoding="utf-8"))
        regenerated_sha = regenerated.get("canonical_training_config_sha256")
    payload["training_config_regeneration"]["sha256"] = regenerated_sha
    payload["training_config_regeneration"]["expected_sha256"] = \
        EXPECTED_TRAINING_CONFIG_SHA256
    payload["training_config_regeneration"]["match"] = \
        regenerated_sha == EXPECTED_TRAINING_CONFIG_SHA256
    if regenerated_sha != EXPECTED_TRAINING_CONFIG_SHA256:
        fail(f"TRAINING_CONFIG_DRIFT: {regenerated_sha}")

    # ---- F. full identity contract, envelope route (declared == observed) -
    from rescue_harness import harness as Hm  # noqa: E402
    ident = {"route": "provenance envelope + train/validation ledgers + run source dirs"}
    try:
        contract = Hm.checkpoint_identity_contract(
            teacher_model_path=Path(args.teacher_model),
            student_model_path=Path(args.student_model),
            source_lock=SL,
            training_config=Path(args.provenance_envelope),
            checkpoint=ckpt,
            train_ledger=Path(args.train_ledger),
            validation_ledger=Path(args.validation_ledger),
            run_source_dirs=run_source_dirs,
        )
        field_checks = [compare(f"identity:{k}", v, contract.get(k))
                        for k, v in sorted(EXPECTED_CONTRACT.items())]
        field_checks.append(compare("identity:training_config_sha256",
                                    EXPECTED_TRAINING_CONFIG_SHA256,
                                    contract.get("training_config_sha256")))
        field_checks.append(compare("identity:identity_sha256",
                                    EXPECTED_IDENTITY_SHA256,
                                    contract.get("identity_sha256")))
        ident.update({
            "status": "PASS",
            "identity_sha256": contract.get("identity_sha256"),
            "provenance_verification_status":
                (contract.get("provenance_verification") or {}).get("status"),
            "provenance_records": len(
                (contract.get("provenance_verification") or {}).get("records") or []),
            "checks": field_checks,
            "mismatches": [c for c in field_checks if not c["match"]],
        })
        for c in field_checks:
            if not c["match"]:
                fail(f"IDENTITY_FIELD_DRIFT: {c['check']}")
    except Exception as exc:  # fail closed, never "close enough"
        ident.update({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
        fail(f"IDENTITY_CONTRACT_FAILED: {type(exc).__name__}: {exc}")
    payload["identity_contract_envelope_route"] = ident

    # ---- G. training-config identity, direct route (what the runner uses) -
    direct = {"route": "canonical training config file (no envelope)"}
    try:
        cfg = Hm.canonical_training_config(Path(args.direct_training_config), source_lock=SL)
        sha = hashlib.sha256(json.dumps(cfg, sort_keys=True,
                                        separators=(",", ":")).encode()).hexdigest()
        direct.update({"status": "PASS" if sha == EXPECTED_TRAINING_CONFIG_SHA256 else "FAIL",
                       "training_config_sha256": sha,
                       "expected_sha256": EXPECTED_TRAINING_CONFIG_SHA256,
                       "match": sha == EXPECTED_TRAINING_CONFIG_SHA256,
                       "is_envelope": bool(
                           __import__("rescue_harness.training_identity", fromlist=["x"])
                           .looks_like_envelope(Path(args.direct_training_config)))})
        if sha != EXPECTED_TRAINING_CONFIG_SHA256:
            fail(f"DIRECT_TRAINING_CONFIG_DRIFT: {sha}")
    except Exception as exc:
        direct.update({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
        fail(f"DIRECT_TRAINING_CONFIG_FAILED: {type(exc).__name__}: {exc}")
    payload["identity_contract_direct_route"] = direct

    # ---- H. runtime versions + dtype policy ------------------------------
    import torch  # noqa: E402
    import transformers  # noqa: E402
    from rescue_harness import training_identity as TI  # noqa: E402
    runtime = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
    }
    runtime_checks = [compare(f"runtime:{k}", v, runtime.get(k))
                      for k, v in EXPECTED_RUNTIME.items()]
    dtype = {"canonical": Hm.CANONICAL_TARGET_DTYPE,
             "canonical_expected": CANONICAL_DTYPE,
             "resolved": str(Hm.resolve_dtype(CANONICAL_DTYPE))}
    try:
        Hm.assert_target_dtype_allowed(REJECTED_DTYPE)
        dtype["float16_rejected_before_load"] = False
        fail("DTYPE_POLICY_DRIFT: float16 was accepted")
    except Hm.NumericsError:
        dtype["float16_rejected_before_load"] = True
    dtype["canonical_allowed"] = True
    try:
        Hm.assert_target_dtype_allowed(CANONICAL_DTYPE)
    except Exception as exc:
        dtype["canonical_allowed"] = False
        fail(f"DTYPE_POLICY_DRIFT: canonical dtype rejected: {exc}")
    dtype["allowed_dtypes"] = list(TI.TRAINING_DTYPES)
    payload["runtime"] = {"observed": runtime, "checks": runtime_checks,
                          "mismatches": [c for c in runtime_checks if not c["match"]]}
    payload["dtype_policy"] = dtype
    for c in runtime_checks:
        if not c["match"]:
            fail(f"RUNTIME_DRIFT: {c['check']}")

    # ---- verdict ---------------------------------------------------------
    payload["verdict"] = "PASS" if not payload["errors"] else "IDENTITY_DRIFT"
    payload["errors"] = sorted(set(payload["errors"]))
    Hm.atomic_write_text(out / "v3_identity_preflight.json",
                         json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"gate": "identity", "status": payload["verdict"],
                      "manifest_sha256": manifest_sha,
                      "checkpoint_sha256": ckpt_sha,
                      "training_config_sha256": regenerated_sha,
                      "identity_sha256": ident.get("identity_sha256"),
                      "errors": payload["errors"]}, ensure_ascii=False))
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
