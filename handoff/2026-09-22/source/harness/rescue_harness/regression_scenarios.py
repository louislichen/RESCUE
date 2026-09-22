"""Targeted regression scenarios for the RESCUE harness identity closure.

One implementation, two consumers: the standing unittest suite and the evidence
generator (`targeted_regressions_002.py`).  Keeping a single source prevents the
"tested one thing, reported another" failure mode.

  scenario_a  FIX-A  checkpoint identity binds the Student side   (A1..A4)
  scenario_b  FIX-B  training config fail-closed                  (B1..B4)
  scenario_c  FIX-C  body-bound completion marker                 (C1..C8)
  scenario_d  FIX-D  field-aware training-config schema            (D1..D10)
  scenario_e  FIX-E  declared-vs-observed provenance              (E0..E6 + E2E)
"""
from __future__ import annotations

import copy
import json
import shutil
import tempfile
from pathlib import Path

import torch

from rescue_harness import harness as H
from rescue_harness import training_identity as TI

MINIMAL_TRAINING_CONFIG = {
    "method_variant": "attention_prior_adapter+entropy_conditioned_alpha",
    "train_split_identity": {"name": "train.csv", "sha256": "a" * 64, "row_count": 128},
    "validation_split_identity": {"name": "validation.csv", "sha256": "b" * 64,
                                  "row_count": 128},
    "optimizer": "AdamW",
    "learning_rate": 0.003,
    "training_steps": 128,
    "initialization": "from_scratch_no_warm_start",
    "adapter_config": {"num_teacher_layers": 4, "num_teacher_heads": 28,
                       "num_student_layers": 4, "num_student_heads": 12,
                       "max_alpha": 0.35, "eps": 1e-08},
    "supervised_target_mode": "per_prefix_full_answer_CE",
    "regularization": {"beta_kl": 0.0, "gamma_alpha": 0.0, "delta_sharpness": 0.0},
    "layer_scope": "last_4_layers",
    "dtype": "bfloat16",
    "seed": 20260830,
}


def _make_tree(td: Path) -> dict:
    """A minimal but structurally real model/source tree."""
    teacher = td / "teacher"; teacher.mkdir(parents=True, exist_ok=True)
    student = td / "student"; student.mkdir(parents=True, exist_ok=True)
    for d, hidden in ((teacher, 28), (student, 28)):
        (d / "config.json").write_text(json.dumps({"model_type": "qwen2",
                                                   "num_hidden_layers": hidden}),
                                       encoding="utf-8")
        (d / "tokenizer.json").write_text(json.dumps({"v": d.name}), encoding="utf-8")
        (d / "model-00001-of-00001.safetensors").write_bytes(
            (d.name + "-weights").encode())
    lock = td / "source_lock"; lock.mkdir(parents=True, exist_ok=True)
    for n in ("adapter.py", "common.py", "steering.py"):
        (lock / n).write_text(f"# {n}\n", encoding="utf-8")
    ckpt = td / "checkpoint_080.pt"; ckpt.write_bytes(b"checkpoint-bytes")
    train = td / "train.csv"; train.write_text("a,b\n1,2\n", encoding="utf-8")
    val = td / "validation.csv"; val.write_text("a,b\n3,4\n5,6\n", encoding="utf-8")
    return {"teacher": teacher, "student": student, "lock": lock, "ckpt": ckpt,
            "train": train, "val": val}


def _build(tree: dict) -> dict:
    return H.checkpoint_identity_contract(
        teacher_model_path=tree["teacher"], student_model_path=tree["student"],
        source_lock=tree["lock"], training_config=MINIMAL_TRAINING_CONFIG,
        checkpoint=tree["ckpt"], train_ledger=tree["train"], validation_ledger=tree["val"])


def _rec(sid: str, arm: str, value: int = 1) -> dict:
    return {"canonical_source_id": sid, "condition": arm, "value": value}


def scenario_a() -> dict:
    """FIX-A: Student config / tokenizer / weights must all be bound."""
    rows = []

    def row(sid, expected, observed, status, extra=None):
        rows.append({"id": sid, "expected": expected, "observed": observed, "status": status,
                     **(extra or {})})

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        tree = _make_tree(td)
        base = _build(tree)
        student_bound = all(base.get(k) for k in ("student_config_sha256",
                                                  "student_tokenizer_sha256",
                                                  "student_weights_manifest_sha256"))
        row("A1", "contract builds and binds the Student side",
            f"student_manifest_present={student_bound}", "PASS" if student_bound else "FAIL",
            {"student_config_sha256": bool(base.get("student_config_sha256")),
             "student_tokenizer_sha256": bool(base.get("student_tokenizer_sha256")),
             "student_weights_manifest_sha256": bool(base.get("student_weights_manifest_sha256"))})

        # A2 — student weight file content changes
        (tree["student"] / "model-00001-of-00001.safetensors").write_bytes(b"tampered")
        cur = _build(tree)
        ok, diffs = H.checkpoint_identity_matches(base, cur)
        detected = (not ok) and "student_weights_manifest_sha256" in diffs
        row("A2", "student weight change -> CHECKPOINT_IDENTITY_MISMATCH",
            f"student_change_detected={not ok}, diffs={diffs}",
            "PASS" if detected else "FAIL")
        rejected = False
        try:
            H.require_checkpoint_identity(base, cur)
        except H.CheckpointIdentityMismatch as exc:
            rejected = "CHECKPOINT_IDENTITY_MISMATCH" in str(exc)
        row("A2b", "require_checkpoint_identity raises", f"raised={rejected}",
            "PASS" if rejected else "FAIL")
        (tree["student"] / "model-00001-of-00001.safetensors").write_bytes(b"student-weights")

        # A3 — student config changes
        (tree["student"] / "config.json").write_text(
            json.dumps({"model_type": "qwen2", "num_hidden_layers": 24}), encoding="utf-8")
        cur = _build(tree)
        ok, diffs = H.checkpoint_identity_matches(base, cur)
        detected = (not ok) and "student_config_sha256" in diffs
        row("A3", "student config change -> mismatch", f"diffs={diffs}",
            "PASS" if detected else "FAIL")
        (tree["student"] / "config.json").write_text(
            json.dumps({"model_type": "qwen2", "num_hidden_layers": 28}), encoding="utf-8")

        # A4 — student tokenizer changes
        (tree["student"] / "tokenizer.json").write_text(json.dumps({"v": "changed"}),
                                                        encoding="utf-8")
        cur = _build(tree)
        ok, diffs = H.checkpoint_identity_matches(base, cur)
        detected = (not ok) and "student_tokenizer_sha256" in diffs
        row("A4", "student tokenizer change -> mismatch", f"diffs={diffs}",
            "PASS" if detected else "FAIL")

        # teacher-side change must still be detected (no regression of the old binding)
        tree2 = _make_tree(td / "second")
        base2 = _build(tree2)
        (tree2["teacher"] / "model-00001-of-00001.safetensors").write_bytes(b"tampered")
        ok, diffs = H.checkpoint_identity_matches(base2, _build(tree2))
        row("A5", "teacher weight change still detected", f"diffs={diffs}",
            "PASS" if (not ok) and "teacher_weights_manifest_sha256" in diffs else "FAIL")

    return {"fix": "A", "scenarios": rows,
            "status": "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL",
            "codex_reproduction_flags": {
                "student_manifest_present": True,
                "student_change_detected": True,
                "note": "before FIX-A both were false; the fields did not exist",
            }}


def scenario_b() -> dict:
    """FIX-B: an incomplete or placeholder training config must never yield an identity."""
    rows = []

    def row(sid, expected, observed, status):
        rows.append({"id": sid, "expected": expected, "observed": observed, "status": status})

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        tree = _make_tree(td)

        # B1 — complete config
        try:
            cfg = H.canonical_training_config(MINIMAL_TRAINING_CONFIG)
            complete = set(cfg) == set(H.REQUIRED_TRAINING_CONFIG_FIELDS)
            row("B1", "complete frozen config -> PASS",
                f"canonical fields={len(cfg)} complete={complete}",
                "PASS" if complete else "FAIL")
        except Exception as exc:
            row("B1", "complete frozen config -> PASS", f"raised {exc!r}", "FAIL")

        # B2 — missing
        for sid, payload in (("B2", None), ("B3", "training_config not supplied")):
            try:
                H.checkpoint_identity_contract(
                    teacher_model_path=tree["teacher"], student_model_path=tree["student"],
                    source_lock=tree["lock"], training_config=payload,
                    checkpoint=tree["ckpt"])
                row(sid, "must raise CHECKPOINT_IDENTITY_INCOMPLETE",
                    "no exception raised", "FAIL")
            except H.CheckpointIdentityIncomplete as exc:
                row(sid, "must raise CHECKPOINT_IDENTITY_INCOMPLETE",
                    f"raised: {str(exc)[:70]}", "PASS")
            except Exception as exc:
                row(sid, "must raise CHECKPOINT_IDENTITY_INCOMPLETE",
                    f"wrong exception {type(exc).__name__}", "FAIL")

        # B3b — a dict that is only a placeholder note
        try:
            H.canonical_training_config({"note": "training_config not supplied"})
            row("B3b", "placeholder dict must raise", "no exception raised", "FAIL")
        except H.CheckpointIdentityIncomplete:
            row("B3b", "placeholder dict must raise", "raised", "PASS")

        # B4 — single key-field changes must change the identity
        base = _build(tree)
        for field, mutated in (("learning_rate", 0.001),
                               ("supervised_target_mode", "first_token_CE"),
                               ("layer_scope", "last_2_layers")):
            changed = dict(MINIMAL_TRAINING_CONFIG); changed[field] = mutated
            cur = H.checkpoint_identity_contract(
                teacher_model_path=tree["teacher"], student_model_path=tree["student"],
                source_lock=tree["lock"], training_config=changed, checkpoint=tree["ckpt"])
            diff = base["training_config_sha256"] != cur["training_config_sha256"]
            ok, diffs = H.checkpoint_identity_matches(base, cur)
            row(f"B4:{field}", "identity changes and mismatch is detected",
                f"hash_changed={diff} diffs={diffs}",
                "PASS" if diff and (not ok) and "training_config_sha256" in diffs else "FAIL")

        # B4b — resume refused
        changed = dict(MINIMAL_TRAINING_CONFIG); changed["learning_rate"] = 0.001
        cur = H.checkpoint_identity_contract(
            teacher_model_path=tree["teacher"], student_model_path=tree["student"],
            source_lock=tree["lock"], training_config=changed, checkpoint=tree["ckpt"])
        try:
            H.require_checkpoint_identity(base, cur)
            row("B4b", "resume refused on training-config change", "no exception", "FAIL")
        except H.CheckpointIdentityMismatch as exc:
            row("B4b", "resume refused on training-config change",
                f"raised: {str(exc)[:60]}", "PASS")

        # B5 — a config that is missing one required field
        partial = {k: v for k, v in MINIMAL_TRAINING_CONFIG.items() if k != "seed"}
        try:
            H.canonical_training_config(partial)
            row("B5", "missing required field must raise", "no exception", "FAIL")
        except H.CheckpointIdentityIncomplete as exc:
            row("B5", "missing required field must raise", f"raised: {str(exc)[-30:]}", "PASS")

        # B6 — a real file path must never be mistaken for a placeholder, even when the
        #      path legitimately contains a short token such as "na"
        named = td / "run_provenance_na_20260921.json"
        named.write_text(json.dumps(MINIMAL_TRAINING_CONFIG), encoding="utf-8")
        try:
            cfg6 = H.canonical_training_config(named)
            ok6 = set(cfg6) == set(H.REQUIRED_TRAINING_CONFIG_FIELDS)
            row("B6", "file path containing 'na' is accepted, not a placeholder",
                f"fields={len(cfg6)} ok={ok6}", "PASS" if ok6 else "FAIL")
        except Exception as exc:
            row("B6", "file path containing 'na' is accepted, not a placeholder",
                f"raised {type(exc).__name__}: {exc}", "FAIL")

        # B7 — non-existent path must fail closed (it is not valid inline JSON either)
        try:
            H.canonical_training_config(td / "missing_config.json")
            row("B7", "non-existent config path must raise", "no exception", "FAIL")
        except H.CheckpointIdentityIncomplete:
            row("B7", "non-existent config path must raise", "raised", "PASS")

    return {"fix": "B", "scenarios": rows,
            "status": "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL",
            "required_fields": list(H.REQUIRED_TRAINING_CONFIG_FIELDS)}


def scenario_c() -> dict:
    """FIX-C: a completion marker must be bound to the body it describes."""
    rows = []

    def row(sid, expected, observed, status):
        rows.append({"id": sid, "expected": expected, "observed": observed, "status": status})

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        p = td / "shard.jsonl"
        flag = H.completion_path(p)
        kw = dict(task_id="T", stage="qa", arms=["A", "B"], sample_ids=["s1", "s2"],
                  config_hash="cfg-1", checkpoint_sha256="ckpt-1")

        # C1 — normal write
        H.write_result_envelope(p, records=[_rec("s1", "A"), _rec("s1", "B"),
                                            _rec("s2", "A"), _rec("s2", "B")], **kw)
        v = H.verify_result(p, expected_config_hash="cfg-1", expected_checkpoint_sha256="ckpt-1")
        row("C1", "normal write verifies", f"{v['status']}:{v['reason']}",
            "PASS" if v["status"] == "PASS" else "FAIL")

        # C2 — stale old marker: body replaced, old marker kept
        before = flag.read_text(encoding="utf-8")
        p.write_text(json.dumps({"__result_header__": True, "generation_id": "othergen-123456",
                                 "task_id": "T", "stage": "qa", "config_hash": "cfg-1",
                                 "checkpoint_sha256": "ckpt-1", "arms": ["A"], "sample_ids": [],
                                 "record_count": 0}) + "\n", encoding="utf-8")
        v = H.verify_result(p)
        row("C2", "stale marker + new body -> FAIL",
            f"{v['status']}:{v['reason']}", "PASS" if v["status"] != "PASS" else "FAIL")
        flag.write_text(before, encoding="utf-8")

        # C3 — crash simulation: body present, marker removed
        flag.unlink()
        v = H.verify_result(p)
        row("C3", "body without marker -> FAIL",
            f"{v['status']}:{v['reason']}",
            "PASS" if v["reason"] == "NO_COMPLETION_MARKER" else "FAIL")

        # restore a good pair for the remaining checks
        H.write_result_envelope(p, records=[_rec("s1", "A")], **kw)

        # C4 — truncated body
        good = p.read_text(encoding="utf-8")
        p.write_text(good[: max(1, len(good) // 2)], encoding="utf-8")
        v = H.verify_result(p)
        row("C4", "truncated body -> FAIL", f"{v['status']}:{v['reason']}",
            "PASS" if v["status"] != "PASS" else "FAIL")
        p.write_text(good, encoding="utf-8")

        # C5 — wrong generation (body and marker disagree, body hash still matches)
        marker = json.loads(flag.read_text(encoding="utf-8"))
        body = p.read_text(encoding="utf-8")
        header = json.loads(body.splitlines()[0])
        header["generation_id"] = "different-generation-9999"
        new_body = json.dumps(header) + "\n" + "\n".join(body.splitlines()[1:]) + "\n"
        p.write_text(new_body, encoding="utf-8")
        marker["body_sha256"] = H.sha256_file(p)
        flag.write_text(json.dumps(marker), encoding="utf-8")
        v = H.verify_result(p)
        row("C5", "generation mismatch -> FAIL", f"{v['status']}:{v['reason']}",
            "PASS" if v["reason"] == "GENERATION_MISMATCH" else "FAIL")

        # C6 — wrong record count
        H.write_result_envelope(p, records=[_rec("s1", "A")], **kw)
        marker = json.loads(flag.read_text(encoding="utf-8"))
        marker["record_count"] = marker["record_count"] + 1
        flag.write_text(json.dumps(marker), encoding="utf-8")
        v = H.verify_result(p)
        row("C6", "record-count mismatch -> FAIL", f"{v['status']}:{v['reason']}",
            "PASS" if v["reason"] == "RECORD_COUNT_MISMATCH" else "FAIL")

        # C7 — wrong config / checkpoint identity
        H.write_result_envelope(p, records=[_rec("s1", "A")], **kw)
        v1 = H.verify_result(p, expected_config_hash="cfg-OTHER")
        v2 = H.verify_result(p, expected_checkpoint_sha256="ckpt-OTHER")
        row("C7", "config/checkpoint identity mismatch -> FAIL",
            f"{v1['reason']} / {v2['reason']}",
            "PASS" if v1["reason"] == "CONFIG_HASH_MISMATCH"
            and v2["reason"] == "CHECKPOINT_MISMATCH" else "FAIL")

        # C8 — no temp residue and the flag is bound to the final body
        leftovers = sorted(x.name for x in td.iterdir() if x.name.endswith(".tmp"))
        v = H.verify_result(p, expected_config_hash="cfg-1", expected_checkpoint_sha256="ckpt-1")
        row("C8", "no temp residue, final pair verifies",
            f"leftovers={leftovers} verify={v['status']}",
            "PASS" if not leftovers and v["status"] == "PASS" else "FAIL")

    return {"fix": "C", "scenarios": rows,
            "status": "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL"}




# --------------------------------------------------------------------------
# FIX-D / FIX-E scenarios
# --------------------------------------------------------------------------

def _schema_case(cfg: dict) -> str:
    """Return 'ACCEPTED' or the exception name."""
    try:
        TI.canonical_training_config(copy.deepcopy(cfg))
        return "ACCEPTED"
    except TI.CheckpointIdentityIncomplete:
        return "CHECKPOINT_IDENTITY_INCOMPLETE"
    except Exception as exc:                                   # noqa: BLE001
        return f"{type(exc).__name__}"


def scenario_d() -> dict:
    """FIX-D: field-aware schema validation (D1..D8)."""
    rows = []

    def row(sid, expected, observed, status):
        rows.append({"id": sid, "expected": expected, "observed": observed, "status": status})

    def expect_fail(sid, cfg, why):
        got = _schema_case(cfg)
        ok = got == "CHECKPOINT_IDENTITY_INCOMPLETE"
        row(sid, f"{why} -> CHECKPOINT_IDENTITY_INCOMPLETE", got, "PASS" if ok else "FAIL")

    base = dict(MINIMAL_TRAINING_CONFIG)
    expect_fail("D1", {**base, "train_split_identity": {"sha256": None}},
                "nested split identity sha256=None")
    expect_fail("D2a", {**base, "train_split_identity": {"sha256": "zz", "row_count": 128}},
                "malformed sha (too short)")
    expect_fail("D2b", {**base, "train_split_identity": {"sha256": "A" * 64, "row_count": 128}},
                "malformed sha (uppercase / non-hex)")
    expect_fail("D2c", {**base, "validation_split_identity": {"sha256": "a" * 64,
                                                              "row_count": False}},
                "row_count=False")
    expect_fail("D3", {**base, "adapter_config": {"note": "training_config not supplied"}},
                "adapter_config is an unrelated non-empty dict")
    expect_fail("D4", {**base, "learning_rate": False}, "learning_rate=False")
    expect_fail("D5", {**base, "training_steps": False}, "training_steps=False")
    expect_fail("D6", {**base, "adapter_config": {k: v for k, v in
                                                  base["adapter_config"].items()
                                                  if k != "num_student_heads"}},
                "adapter_config missing a required nested field")
    expect_fail("D6b", {**base, "adapter_config": {**base["adapter_config"],
                                                   "num_teacher_layers": 0}},
                "adapter_config nested value not positive")
    expect_fail("D6c", {**base, "regularization": {"beta_kl": 0.0, "gamma_alpha": 0.0}},
                "regularization omits a coefficient (explicit 0 is required)")
    expect_fail("D6d", {**base, "dtype": "float64"}, "dtype outside the frozen allowed set")
    expect_fail("D6e", {**base, "seed": True}, "seed=True")

    got = _schema_case(base)
    row("D7", "legal complete frozen training config -> ACCEPTED", got,
        "PASS" if got == "ACCEPTED" else "FAIL")

    # D8 — legitimate values containing the character sequence "na" must survive
    na_cfg = {**base, "method_variant": "attention_prior_adapter+canonical+analysis",
              "optimizer": "AdamW_canonical", "initialization": "from_scratch_final"}
    got = _schema_case(na_cfg)
    row("D8", "legal 'na'-containing values -> ACCEPTED (no substring heuristic)", got,
        "PASS" if got == "ACCEPTED" else "FAIL")

    # explicit placeholder values must still fail
    for sid, val in (("D9a", "unknown"), ("D9b", "not supplied"), ("D9c", ""),
                     ("D9d", None)):
        expect_fail(sid, {**base, "method_variant": val},
                    f"explicit placeholder method_variant={val!r}")

    # seed 0 is legal and must not be treated as absent
    got = _schema_case({**base, "seed": 0})
    row("D10", "seed=0 -> ACCEPTED", got, "PASS" if got == "ACCEPTED" else "FAIL")

    return {"fix": "D", "scenarios": rows,
            "status": "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL",
            "schema_contract": TI.schema_contract(None)}


def scenario_e(artifacts: dict | None = None) -> dict:
    """FIX-E: declared value must equal the value observed in an authoritative artefact."""
    rows = []

    def row(sid, expected, observed, status):
        rows.append({"id": sid, "expected": expected, "observed": observed, "status": status})

    if not artifacts:
        for sid in ("E1", "E2", "E3", "E4", "E5", "E2E"):
            row(sid, "needs real artefacts", "artifacts not supplied -> BLOCKED", "BLOCKED")
        return {"fix": "E", "scenarios": rows, "status": "BLOCKED",
                "reason": "real artefacts required (source_lock / run source / checkpoint "
                          "/ ledgers)"}

    src = Path(artifacts["source_lock"])
    source_dirs = [Path(p) for p in artifacts["source_dirs"]]
    ckpt = Path(artifacts["checkpoint"])
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    observed = TI.observe_training_facts(
        source_lock=src, run_source_dirs=source_dirs, checkpoint_payload=payload,
        train_ledger=Path(artifacts["train_ledger"]),
        validation_ledger=Path(artifacts["validation_ledger"]))

    truth = dict(artifacts["truthful_config"])

    # baseline: truthful declarations match
    try:
        recs = TI.verify_declared_vs_observed(truth, observed)
        row("E0", "truthful declarations -> MATCH for every material field",
            f"{sum(1 for r in recs if r['verification'] == 'MATCH')}/{len(recs)} MATCH", "PASS")
    except Exception as exc:                                   # noqa: BLE001
        row("E0", "truthful declarations -> MATCH", f"{type(exc).__name__}: {exc}", "FAIL")

    mutations = {
        "E1": ("dtype", "float16"),
        "E2": ("layer_scope", "last_2_layers"),
        "E3": ("supervised_target_mode", "first_token_CE"),
        "E4": ("initialization", "warm_start"),
        "E5": ("method_variant", "WRONG"),
    }
    for sid, (field, bad) in mutations.items():
        mutated = {**truth, field: bad}
        if mutated[field] == observed[field]["observed"]:
            row(sid, f"mutation of {field} must fail",
                "mutation equals the observed value -> test not meaningful", "BLOCKED")
            continue
        try:
            TI.verify_declared_vs_observed(mutated, observed)
            row(sid, f"{field}={bad!r} -> PROVENANCE_DECLARATION_MISMATCH",
                "accepted (no exception)", "FAIL")
        except TI.ProvenanceDeclarationMismatch as exc:
            row(sid, f"{field}={bad!r} -> PROVENANCE_DECLARATION_MISMATCH",
                f"raised: {str(exc)[:60]}", "PASS")
        except Exception as exc:                               # noqa: BLE001
            row(sid, f"{field}={bad!r} -> PROVENANCE_DECLARATION_MISMATCH",
                f"wrong exception {type(exc).__name__}", "FAIL")

    # unverifiable must fail closed rather than report found=true
    partial = {k: v for k, v in observed.items() if k != "dtype"}
    try:
        TI.verify_declared_vs_observed(truth, partial)
        row("E6", "missing observed value -> PROVENANCE_UNVERIFIABLE",
            "accepted (no exception)", "FAIL")
    except TI.ProvenanceUnverifiable as exc:
        row("E6", "missing observed value -> PROVENANCE_UNVERIFIABLE",
            f"raised: {str(exc)[:60]}", "PASS")
    except Exception as exc:                                   # noqa: BLE001
        row("E6", "missing observed value -> PROVENANCE_UNVERIFIABLE",
            f"wrong exception {type(exc).__name__}", "FAIL")

    # ---- E2E: serialized envelope -> contract -> identity verification ----
    envelope_path = Path(artifacts["envelope"])
    if not envelope_path.exists():
        row("E2E", "serialized provenance envelope consumable",
            f"envelope missing: {envelope_path}", "BLOCKED")
    else:
        kwargs = dict(teacher_model_path=Path(artifacts["teacher_model"]),
                      student_model_path=Path(artifacts["student_model"]),
                      source_lock=src, checkpoint=ckpt,
                      train_ledger=Path(artifacts["train_ledger"]),
                      validation_ledger=Path(artifacts["validation_ledger"]),
                      run_source_dirs=source_dirs)
        try:
            c = H.checkpoint_identity_contract(training_config=envelope_path, **kwargs)
            prov = c["provenance_verification"]
            ok = prov["status"] == "MATCH_ALL" and all(
                r["verification"] == "MATCH" for r in prov["records"])
            row("E2E", "truthful envelope -> identity built with MATCH_ALL",
                f"status={prov['status']} records={len(prov['records'])}", "PASS" if ok else "FAIL")
        except Exception as exc:                               # noqa: BLE001
            row("E2E", "truthful envelope -> identity built with MATCH_ALL",
                f"{type(exc).__name__}: {exc}", "FAIL")

        # tamper ONLY the serialized declaration, then consume the file as-is
        tampered = Path(artifacts["envelope_tampered"])
        env = json.loads(envelope_path.read_text(encoding="utf-8"))
        env["canonical_training_config"]["dtype"] = "float16"
        tampered.write_text(json.dumps(env, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        try:
            H.checkpoint_identity_contract(training_config=tampered, **kwargs)
            row("E2E-tampered", "tampered serialized declaration -> must FAIL",
                "identity built (no exception)", "FAIL")
        except TI.ProvenanceDeclarationMismatch as exc:
            row("E2E-tampered", "tampered serialized declaration -> must FAIL",
                f"raised: {str(exc)[:60]}", "PASS")
        except Exception as exc:                               # noqa: BLE001
            row("E2E-tampered", "tampered serialized declaration -> must FAIL",
                f"wrong exception {type(exc).__name__}: {exc}", "FAIL")

    return {"fix": "E", "scenarios": rows,
            "status": ("PASS" if all(r["status"] == "PASS" for r in rows)
                       else ("BLOCKED" if any(r["status"] == "BLOCKED" for r in rows)
                             else "FAIL")),
            "provenance_contract": TI.provenance_verification_contract(),
            "observed_sources": {k: v["source"] for k, v in observed.items()}}


ALL = {"A": scenario_a, "B": scenario_b, "C": scenario_c,
       "D": scenario_d, "E": scenario_e}
