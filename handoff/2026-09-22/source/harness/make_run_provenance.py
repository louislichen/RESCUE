"""Build the canonical training config for a trained checkpoint from REAL artefacts."""
from __future__ import annotations
import argparse, hashlib, json, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from rescue_harness import harness as H
from rescue_harness import training_identity as TI

def adapter_config_from_checkpoint(ckpt:Path,source_lock:Path):
    payload=torch.load(ckpt,map_location="cpu",weights_only=False)
    sd=payload["adapter"]; mix=tuple(int(x) for x in sd["base.mix_logits"].shape)
    if len(mix)!=4: raise SystemExit(f"unexpected mix_logits shape {mix}")
    cfg={"num_student_layers":mix[0],"num_student_heads":mix[1],
         "num_teacher_layers":mix[2],"num_teacher_heads":mix[3]}
    sys.path.insert(0,str(source_lock.resolve()))
    from adapter import AttentionAdapterConfig
    fields=AttentionAdapterConfig.__dataclass_fields__
    cfg["max_alpha"]=fields["max_alpha"].default; cfg["eps"]=fields["eps"].default
    return cfg,{"checkpoint_payload_step":payload.get("step"),
                "checkpoint_payload_supervision":payload.get("supervision")}
def main()->int:
    ap=argparse.ArgumentParser()
    for n in ("source-lock","run-root","checkpoint","train-ledger","validation-ledger","source-dir","out"):
        ap.add_argument("--"+n,required=True)
    ap.add_argument("--method-variant",default="attention_prior_adapter+entropy_conditioned_alpha+per_prefix_sequence_supervision")
    ap.add_argument("--supervised-target-mode",default="per_prefix_full_answer_CE")
    ap.add_argument("--initialization",default="from_scratch_no_warm_start")
    ap.add_argument("--layer-scope",default="last_4_layers")
    ap.add_argument("--dtype",default=H.CANONICAL_TARGET_DTYPE)
    args=ap.parse_args()
    source_lock=Path(args.source_lock); run_root=Path(args.run_root); ckpt=Path(args.checkpoint)
    source_dirs=[Path(s) for s in str(args.source_dir).split(",") if s.strip()]
    adapter_cfg,ck_meta=adapter_config_from_checkpoint(ckpt,source_lock)
    training_result=json.loads((run_root/"training_result.json").read_text(encoding="utf-8"))
    declared={"method_variant":args.method_variant,"supervised_target_mode":args.supervised_target_mode,
              "initialization":args.initialization,"layer_scope":args.layer_scope,"dtype":args.dtype}
    payload=torch.load(ckpt,map_location="cpu",weights_only=False)
    observed=TI.observe_training_facts(source_lock=source_lock,run_source_dirs=source_dirs,
        checkpoint_payload=payload,train_ledger=Path(args.train_ledger),
        validation_ledger=Path(args.validation_ledger))
    for field,value in declared.items():
        if field in observed and observed[field]["observed"]!=value:
            raise SystemExit(f"PROVENANCE_DECLARATION_MISMATCH: {field}")
    frozen=json.loads((source_lock/"frozen_training_config.json").read_text(encoding="utf-8"))
    contract=json.loads((source_lock/"frozen_experiment_and_statistics_contract.json").read_text(encoding="utf-8"))
    raw={"method_variant":declared["method_variant"],
         "train_split_identity":H._ledger_identity(Path(args.train_ledger)),
         "validation_split_identity":H._ledger_identity(Path(args.validation_ledger)),
         "optimizer":frozen.get("optimizer"),"learning_rate":frozen.get("learning_rate"),
         "training_steps":frozen.get("fixed_optimizer_steps"),"initialization":declared["initialization"],
         "adapter_config":adapter_cfg,"supervised_target_mode":declared["supervised_target_mode"],
         "regularization":frozen.get("regularization"),"layer_scope":declared["layer_scope"],
         "dtype":declared["dtype"],"seed":frozen.get("seed"),"batch_size":frozen.get("batch_size"),
         "weight_decay":frozen.get("weight_decay"),"scheduler":frozen.get("scheduler"),
         "checkpoint_save_steps":frozen.get("checkpoint_save_steps"),"early_stopping":frozen.get("early_stopping"),
         "selection_rule":(contract.get("optimization",{}) or {}).get("selection"),
         "supervised_token_count":training_result.get("supervised_token_count"),
         "checkpoint_step":ck_meta.get("checkpoint_payload_step")}
    canonical=TI.canonical_training_config(raw,source_lock=source_lock)
    records=TI.verify_declared_vs_observed(canonical,observed)
    payload_out={"generated_at":datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S %z"),
      "checkpoint":str(ckpt),"checkpoint_sha256":TI._sha256_file(ckpt),"checkpoint_meta":ck_meta,
      "training_result":training_result,"canonical_training_config":canonical,
      "canonical_training_config_sha256":hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
      "source_verification":records,"schema_contract":TI.schema_contract(source_lock),
      "provenance_contract":TI.provenance_verification_contract()}
    H.atomic_write_text(Path(args.out),json.dumps(payload_out,ensure_ascii=False,indent=2)+"\n")
    direct=Path(args.out).with_name(Path(args.out).name.replace("run_provenance","training_config"))
    H.atomic_write_text(direct,json.dumps(canonical,ensure_ascii=False,indent=2)+"\n")
    return 0
if __name__=="__main__": raise SystemExit(main())
