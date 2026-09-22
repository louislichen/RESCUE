"""Build the canonical training config for a trained checkpoint from REAL artefacts.

FIX-E: the caller's claims are never trusted.  Every material field is observed
independently from an authoritative artefact and the result is recorded as
{field, declared, observed, source, verification}; a single mismatch aborts with
PROVENANCE_DECLARATION_MISMATCH and an unobservable field aborts with
PROVENANCE_UNVERIFIABLE.

Usage:
  make_run_provenance.py --source-lock SL --run-root RUN --checkpoint CKPT \
      --train-ledger T --validation-ledger V --source-dir DIR[,DIR] --out OUT.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rescue_harness import harness as H            # noqa: E402
from rescue_harness import training_identity as TI  # noqa: E402


def adapter_config_from_checkpoint(ckpt: Path, source_lock: Path) -> tuple[dict, dict]:
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = payload["adapter"]
    mix = tuple(int(x) for x in sd["base.mix_logits"].shape)      # S, Hs, T, Ht
    if len(mix) != 4:
        raise SystemExit(f"unexpected mix_logits shape {mix}")
    cfg = {"num_student_layers": mix[0], "num_student_heads": mix[1],
           "num_teacher_layers": mix[2], "num_teacher_heads": mix[3]}
    sys.path.insert(0, str(Path(source_lock).resolve()))
    from adapter import AttentionAdapterConfig                    # frozen dataclass
    fields = AttentionAdapterConfig.__dataclass_fields__
    cfg["max_alpha"] = fields["max_alpha"].default
    cfg["eps"] = fields["eps"].default
    return cfg, {
        "checkpoint_payload_step": payload.get("step"),
        "checkpoint_payload_supervision": payload.get("supervision"),
        "evidence": {
            "adapter_shape_fields":
                f"checkpoint_080.pt['adapter']['base.mix_logits'].shape == {mix}",
            "max_alpha": f"source_lock/adapter.py AttentionAdapterConfig.max_alpha = "
                         f"{cfg['max_alpha']}",
            "eps": f"source_lock/adapter.py AttentionAdapterConfig.eps = {cfg['eps']}",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-lock", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-ledger", required=True)
    ap.add_argument("--validation-ledger", required=True)
    ap.add_argument("--source-dir", required=True,
                    help="comma-separated source directories that produced the run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--method-variant",
                    default="attention_prior_adapter+entropy_conditioned_alpha"
                            "+per_prefix_sequence_supervision")
    ap.add_argument("--supervised-target-mode", default="per_prefix_full_answer_CE")
    ap.add_argument("--initialization", default="from_scratch_no_warm_start")
    ap.add_argument("--layer-scope", default="last_4_layers")
    ap.add_argument("--dtype", default=H.CANONICAL_TARGET_DTYPE)
    args = ap.parse_args()

    source_lock = Path(args.source_lock)
    run_root = Path(args.run_root)
    ckpt = Path(args.checkpoint)
    source_dirs = [Path(s) for s in str(args.source_dir).split(",") if s.strip()]

    adapter_cfg, ck_meta = adapter_config_from_checkpoint(ckpt, source_lock)
    training_result = json.loads((run_root / "training_result.json").read_text(encoding="utf-8"))
    declared = {
        "method_variant": args.method_variant,
        "supervised_target_mode": args.supervised_target_mode,
        "initialization": args.initialization,
        "layer_scope": args.layer_scope,
        "dtype": args.dtype,
    }

    # ---- FIX-E: observe independently, then compare -----------------------
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    observed = TI.observe_training_facts(
        source_lock=source_lock, run_source_dirs=source_dirs, checkpoint_payload=payload,
        train_ledger=Path(args.train_ledger), validation_ledger=Path(args.validation_ledger))
    for field, value in declared.items():
        if field in observed and observed[field]["observed"] != value:
            raise SystemExit(
                f"PROVENANCE_DECLARATION_MISMATCH: {field}: declared={value!r} "
                f"observed={observed[field]['observed']!r} "
                f"(source: {observed[field]['source']})")

    frozen = json.loads((source_lock / "frozen_training_config.json").read_text(encoding="utf-8"))
    contract = json.loads((source_lock / "frozen_experiment_and_statistics_contract.json")
                          .read_text(encoding="utf-8"))
    raw = {
        "method_variant": declared["method_variant"],
        "train_split_identity": H._ledger_identity(Path(args.train_ledger)),
        "validation_split_identity": H._ledger_identity(Path(args.validation_ledger)),
        "optimizer": frozen.get("optimizer"),
        "learning_rate": frozen.get("learning_rate"),
        "training_steps": frozen.get("fixed_optimizer_steps"),
        "initialization": declared["initialization"],
        "adapter_config": adapter_cfg,
        "supervised_target_mode": declared["supervised_target_mode"],
        "regularization": frozen.get("regularization"),
        "layer_scope": declared["layer_scope"],
        "dtype": declared["dtype"],
        "seed": frozen.get("seed"),
        "batch_size": frozen.get("batch_size"),
        "weight_decay": frozen.get("weight_decay"),
        "scheduler": frozen.get("scheduler"),
        "checkpoint_save_steps": frozen.get("checkpoint_save_steps"),
        "early_stopping": frozen.get("early_stopping"),
        "selection_rule": (contract.get("optimization", {}) or {}).get("selection"),
        "supervised_token_count": training_result.get("supervised_token_count"),
        "checkpoint_step": ck_meta.get("checkpoint_payload_step"),
    }
    canonical = TI.canonical_training_config(raw, source_lock=source_lock)   # FIX-D
    records = TI.verify_declared_vs_observed(canonical, observed)            # FIX-E

    payload_out = {
        "generated_at": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S %z"),
        "checkpoint": str(ckpt), "checkpoint_sha256": TI._sha256_file(ckpt),
        "checkpoint_meta": ck_meta,
        "training_result": training_result,
        "canonical_training_config": canonical,
        "canonical_training_config_sha256": hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "source_verification": records,                      # FIX-E: MATCH records
        "schema_contract": TI.schema_contract(source_lock),  # FIX-D
        "provenance_contract": TI.provenance_verification_contract(),
        "field_sources": {
            "frozen_values": f"{source_lock}/frozen_training_config.json + "
                             f"frozen_experiment_and_statistics_contract.json",
            "split_identities": "sha256 + row count of the role ledgers",
            "checkpoint_derived": ck_meta["evidence"],
            "runtime_claims": "verified declared-vs-observed, see source_verification",
        },
    }
    H.atomic_write_text(Path(args.out), json.dumps(payload_out, ensure_ascii=False, indent=2) + "\n")
    direct = Path(args.out).with_name(Path(args.out).name.replace("run_provenance",
                                                                  "training_config"))
    H.atomic_write_text(direct, json.dumps(canonical, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"provenance": args.out, "direct_training_config": str(direct),
                      "training_config_sha256": payload_out["canonical_training_config_sha256"],
                      "fields": len(canonical),
                      "verifications": {r["field"]: r["verification"] for r in records}},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
