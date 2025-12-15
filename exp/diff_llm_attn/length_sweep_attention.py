import argparse
import json
from typing import List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..tools.corrs import *
from ..tools.utils import *
from ..tools.hooks import *


def parse_lengths(s: str) -> List[int]:
    parts = [p.strip() for p in s.replace(" ", ",").split(",") if p.strip()]
    out = []
    for p in parts:
        v = int(p)
        if v > 1:
            out.append(v)
    return sorted(list(dict.fromkeys(out)))


def load_model(model_path: str, dtype: torch.dtype,) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)

    max_memory = {
        0: "18GiB",
        1: "18GiB",
        2: "18GiB",
        3: "18GiB",
        4: "18GiB",
        5: "18GiB",
        6: "18GiB",
        7: "5GiB",
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        max_memory=max_memory,
        local_files_only=True,
    )

    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation("eager")
        except Exception as e:
            print(f"!! Could not set eager attention for {model_path}: {e}")

    model.eval()
    return tok, model


def avoid_accelerate_output_gather(model):
    skip = {
        "logits",
        "past_key_values",
        "hidden_states",
        "attentions",
        "last_hidden_state",
    }
    for m in model.modules():
        hk = getattr(m, "_hf_hook", None)
        if hk is not None:
            hk.skip_keys = skip


def parse_args():
    p = argparse.ArgumentParser("Length sweep attention similarity (Route A + Multi-GPU + JSONL + Plot)")

    p.add_argument("--model_path", type=str, required=True)

    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--split", type=str, default="train[:80%]")
    p.add_argument("--max_samples", type=int, default=300)

    p.add_argument("--sweep_step", type=int, default=10, help="If >0, ignore --lengths and sweep to --max_L.")
    p.add_argument("--max_L", type=int, required=True, help="Upper bound when --sweep_step>0.")

    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attention_precision", type=int, default=6)

    p.add_argument("--output_path", type=str, required=True)

    return p.parse_args()


def main():
    args = parse_args()

    L_list = build_sweep_lengths(int(args.sweep_step), int(args.max_L))
    print(f">>> step={args.sweep_step}, max_L={args.max_L}")

    required_L = args.max_L

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    dtype = str2dtype(args.dtype)
    tokenizer, model = load_model(args.model_path, dtype)
    dev = next(model.parameters()).device

    print(f">>> Loading dataset...")
    ds = load_ds(args.dataset_path, args.split)
    n = len(ds) if hasattr(ds, "__len__") else None

    total = 0
    i = 0

    pbar = tqdm(total=args.max_samples, desc="Groups")

    while total < args.max_samples:
        if n is not None and i >= n:
            break

        texts: List[str] = []
        token_len = 0

        while token_len < required_L:
            if n is not None and i >= n:
                break
            ex = ds[i]
            i += 1
            t = ex.get("text")
            if not t or not t.strip():
                continue
            texts.append(t.strip())
            enc_tmp = tokenizer("\n\n".join(texts), return_tensors="pt", truncation=False)
            token_len = int(enc_tmp["input_ids"].shape[-1])

        if token_len < required_L:
            print("Cannot find proper input...")
            break

        group_text = "\n\n".join(texts)
        enc = tokenizer(group_text, return_tensors="pt", truncation=False)
        input_ids_full = enc["input_ids"][0]
        attn_mask_full = enc["attention_mask"][0] if "attention_mask" in enc else None
        T = int(input_ids_full.shape[0])

        input_ids_max = input_ids_full[:required_L].unsqueeze(0)
        attn_mask_max = attn_mask_full[:required_L].unsqueeze(0) if attn_mask_full is not None else None

        input_ids = input_ids_max.to(dev)
        mask = attn_mask_max.to(dev) if attn_mask_max is not None else None

        vecs = sampled_attention_rows_last_layer(model, input_ids, mask, L_list, args.attention_precision)
        vecs = {str(k): to_jsonable(v) for k, v in vecs.items()}  
        record = {
            "i": total,              
            "vecs": vecs             
        }
        with open(args.output_path, "a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

        del input_ids, mask, input_ids_max, attn_mask_max
        torch.cuda.empty_cache()

        total += 1
        pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    main()