"""RESCUE 002: numerically characterise the locked 7B target under both dtypes.

The frozen teacher prior worker loaded the locked Qwen2-7B with
``dtype=torch.float16``.  In the current environment that produced non-finite
logits; the checkpoint is stored bfloat16.  This probe records the measurement
instead of asserting it, and reports per-layer finiteness so the failure can be
localised.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def probe(model, tokenizer, text, lengths, label):
    rec = {"label": label, "dtype": str(next(model.parameters()).dtype), "cases": []}
    ids_all = tokenizer(text, add_special_tokens=False)["input_ids"]
    dev = next(model.parameters()).device
    for n in lengths:
        ids = torch.tensor([ids_all[:n]], device=dev, dtype=torch.long)
        with torch.no_grad():
            o = model(input_ids=ids, use_cache=False, output_hidden_states=True)
        lg = o.logits[0, -1, :].float()
        hs = o.hidden_states
        per_layer = [bool(torch.isfinite(h.float()).all()) for h in hs]
        rec["cases"].append({
            "length": int(n),
            "logits_finite": bool(torch.isfinite(lg).all()),
            "logits_nan_count": int(torch.isnan(lg).sum()),
            "logits_argmax": int(torch.argmax(lg)) if torch.isfinite(lg).any() else None,
            "hidden_states_finite": bool(all(per_layer)),
            "first_non_finite_layer": next((i for i, f in enumerate(per_layer) if not f), None),
            "num_layers": len(per_layer),
        })
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-model", required=True)
    p.add_argument("--student-model", required=True)
    p.add_argument("--source-dir", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--validation-ledger", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[16, 300, 1582])
    args = p.parse_args()

    sys.path.insert(0, str(Path(args.source_dir).resolve()))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sequence_loss_runner import load_rows

    row = load_rows(Path(args.validation_ledger), Path(args.dataset_path))[0]
    text = row["prompt"]
    out = {"probe": "locked_7B_target_numerics", "checkpoint_dtype_on_disk": "bfloat16",
           "note": "no quantization, no context truncation, no model substitution",
           "prompt_source": str(args.validation_ledger) + " row 0",
           "runs": []}
    for dt_name, dt in (("float16", torch.float16), ("bfloat16", torch.bfloat16)):
        m = AutoModelForCausalLM.from_pretrained(
            args.teacher_model, dtype=dt, local_files_only=True,
            attn_implementation="eager", device_map="balanced",
            max_memory={i: "8GiB" for i in range(torch.cuda.device_count())},
            low_cpu_mem_usage=True).eval()
        tok = AutoTokenizer.from_pretrained(args.student_model, local_files_only=True)
        rec = probe(m, tok, text, args.lengths, dt_name)
        rec["device_map"] = dict(getattr(m, "hf_device_map", {}))
        out["runs"].append(rec)
        del m
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "OK",
                      "float16_finite": [c["logits_finite"] for c in out["runs"][0]["cases"]],
                      "bfloat16_finite": [c["logits_finite"] for c in out["runs"][1]["cases"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
