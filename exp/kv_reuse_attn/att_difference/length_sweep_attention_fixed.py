#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""KV-cache reuse attention deviation experiment (HotpotQA).

We compare the *same* model under two settings:
  1) recompute: prefill the whole prompt (docs + query) normally.
  2) naive_reuse: prefill each document independently, concatenate (fuse) their KV caches,
     then append the query.

Control mode:
  --control_mode none:
      compute recompute vs naive_reuse (default behavior)
  --control_mode recompute_self:
      compute recompute vs recompute (run twice) as a sanity-control baseline

To avoid materializing O(L^2) attentions, we only extract the last-layer attention row
for the last prompt token (q_len=1) and average across heads.

Filtering:
- We build the prompt using either all docs in `context` (doc_mode=all) or only gold docs
  referenced by `supporting_facts` (doc_mode=gold_only).
- If total token length (docs + query) exceeds --max_L (default 2050), we drop the sample.

Outputs:
- JSONL per kept sample with metrics (JSD / Pearson / Entropy).
- PNG histograms:
    - JSD: one distribution (because it is a pairwise metric)
    - Pearson: one distribution (pairwise metric)
    - Entropy: two distributions (one for each condition/run)

Notes:
- HotpotQA itself does NOT provide any KV-cache interface; KV reuse is implemented here.
- Intended for large models sharded across multiple GPUs via device_map/max_memory.
"""

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import matplotlib.pyplot as plt
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# -------------------------
# Small helpers
# -------------------------

def str2bool(val):
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {val}")


def str2dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def dtype_name(dt: torch.dtype) -> str:
    if dt == torch.float16:
        return "float16"
    if dt == torch.bfloat16:
        return "bfloat16"
    if dt == torch.float32:
        return "float32"
    return str(dt)


def parse_max_memory(s: Optional[str]) -> Optional[Dict[int, str]]:
    if s is None:
        return None
    s = str(s).strip()
    if not s or s.lower() in {"none", "null"}:
        return None
    out: Dict[int, str] = {}
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Bad max_memory item '{item}', expected 'gpu:GiB' like '0:10GiB'")
        k, v = item.split(":", 1)
        out[int(k.strip())] = v.strip()
    return out


def parse_gib(s: str) -> float:
    st = str(s).strip()
    m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]+)\s*$", st)
    if not m:
        raise ValueError(f"Bad memory string: '{s}' (expected like 10GiB)")
    val = float(m.group(1))
    unit = m.group(2).lower()
    if unit in {"gib", "gb", "g"}:
        return val
    if unit in {"mib", "mb", "m"}:
        return val / 1024.0
    raise ValueError(f"Unsupported unit in memory string: '{s}'")


def fmt_gib(x: float) -> str:
    if abs(x - round(x)) < 1e-6:
        return f"{int(round(x))}GiB"
    return f"{x:.2f}GiB"


def ensure_max_memory_uses_all_visible_gpus(
    max_mem: Optional[Dict[int, str]],
    reserve_gpu0_gib: float = 0.0,
) -> Optional[Dict[int, str]]:
    if not torch.cuda.is_available() or max_mem is None:
        return max_mem
    n = torch.cuda.device_count()
    if n <= 0:
        return max_mem

    out = dict(max_mem)
    template_key = next((k for k in sorted(out.keys()) if k != 0), None)
    template = out[template_key] if template_key is not None else out.get(0, "10GiB")

    for gid in range(n):
        out.setdefault(gid, template)

    if reserve_gpu0_gib > 0.0 and 0 in out:
        try:
            cur = parse_gib(out[0])
            out[0] = fmt_gib(max(0.5, cur - reserve_gpu0_gib))
        except Exception:
            pass
    return out


def _looks_like_bf16_unsupported(errmsg: str) -> bool:
    s = (errmsg or "").lower()
    keywords = [
        "bfloat16", "bf16",
        "not supported", "unsupported",
        "not implemented",
        "requires", "only supports",
        "amp",
    ]
    return any(k in s for k in keywords)


# -------------------------
# Cache utils (multi-version compatible)
# -------------------------

def _is_cache_obj(pkv: Any) -> bool:
    if pkv is None:
        return False
    if hasattr(pkv, "get_seq_length"):
        return True
    tname = type(pkv).__name__.lower()
    return "cache" in tname and not isinstance(pkv, (tuple, list))


def cache_to_legacy(pkv: Any):
    if pkv is None:
        return None
    if isinstance(pkv, (tuple, list)):
        return pkv
    if hasattr(pkv, "to_legacy_cache"):
        return pkv.to_legacy_cache()
    return pkv


def legacy_to_cache(legacy_pkv: Any):
    if legacy_pkv is None:
        return None
    if _is_cache_obj(legacy_pkv):
        return legacy_pkv

    try:
        from transformers.cache_utils import DynamicCache
        if hasattr(DynamicCache, "from_legacy_cache"):
            return DynamicCache.from_legacy_cache(legacy_pkv)
    except Exception:
        pass

    try:
        from transformers.cache_utils import Cache
        if hasattr(Cache, "from_legacy_cache"):
            return Cache.from_legacy_cache(legacy_pkv)
    except Exception:
        pass

    return legacy_pkv


def set_attn_impl_eager(model) -> None:
    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation("eager")
            return
        except Exception:
            pass
    try:
        if hasattr(model, "config") and hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "eager"
    except Exception:
        pass
    if hasattr(model, "model") and hasattr(model.model, "set_attn_implementation"):
        try:
            model.model.set_attn_implementation("eager")
        except Exception:
            pass


# -------------------------
# Metrics
# -------------------------

def pearson_corr(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    n = min(len(a), len(b))
    if n <= 1:
        return None
    a = a[:n]
    b = b[:n]
    ma = sum(a) / n
    mb = sum(b) / n
    cov = 0.0
    va = 0.0
    vb = 0.0
    for x, y in zip(a, b):
        dx = x - ma
        dy = y - mb
        cov += dx * dy
        va += dx * dx
        vb += dy * dy
    denom = math.sqrt(va * vb)
    if denom == 0.0:
        return None
    r = cov / denom
    if not math.isfinite(r):
        return None
    return float(r)


def _normalize_prob(x: Sequence[float], eps: float = 1e-12) -> Optional[List[float]]:
    if not x:
        return None
    y = [0.0 if (v is None or not math.isfinite(float(v)) or float(v) < 0.0) else float(v) for v in x]
    s = sum(y)
    if s <= 0.0:
        return None
    return [(v / s) + eps for v in y]


def entropy(p: Sequence[float]) -> Optional[float]:
    pn = _normalize_prob(p)
    if pn is None:
        return None
    s = sum(pn)
    pn = [v / s for v in pn]
    h = 0.0
    for v in pn:
        if v > 0:
            h -= v * math.log(v)
    return float(h)


def js_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-12) -> Optional[float]:
    pn = _normalize_prob(p, eps=eps)
    qn = _normalize_prob(q, eps=eps)
    if pn is None or qn is None:
        return None
    n = min(len(pn), len(qn))
    pn = pn[:n]
    qn = qn[:n]
    sp = sum(pn)
    sq = sum(qn)
    pn = [v / sp for v in pn]
    qn = [v / sq for v in qn]
    m = [(a + b) / 2.0 for a, b in zip(pn, qn)]

    def kl(a, b):
        s = 0.0
        for x, y in zip(a, b):
            s += x * math.log(x / y)
        return s

    js = 0.5 * kl(pn, m) + 0.5 * kl(qn, m)
    if not math.isfinite(js):
        return None
    return float(js)


# -------------------------
# HotpotQA parsing (robust)
# -------------------------

def _extract_docs_from_context(context_obj) -> List[Tuple[str, List[str]]]:
    docs: List[Tuple[str, List[str]]] = []

    if isinstance(context_obj, dict) and "title" in context_obj and "sentences" in context_obj:
        titles = context_obj["title"]
        sents = context_obj["sentences"]
        if isinstance(titles, list) and isinstance(sents, list):
            for t, ss in zip(titles, sents):
                if ss is None:
                    ss = []
                if isinstance(ss, list):
                    docs.append((str(t), [str(x) for x in ss]))
        return docs

    if isinstance(context_obj, list):
        for item in context_obj:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                t = str(item[0])
                ss = item[1] if item[1] is not None else []
                if isinstance(ss, list):
                    docs.append((t, [str(x) for x in ss]))
            elif isinstance(item, dict) and "title" in item and ("sentences" in item or "sentence" in item):
                t = str(item["title"])
                ss = item.get("sentences", item.get("sentence", []))
                if ss is None:
                    ss = []
                if isinstance(ss, list):
                    docs.append((t, [str(x) for x in ss]))
        return docs

    return docs


def _extract_gold_titles(sample: dict) -> List[str]:
    sf = sample.get("supporting_facts")
    titles: List[str] = []
    if sf is None:
        return titles

    if isinstance(sf, dict) and "title" in sf:
        ts = sf["title"]
        if isinstance(ts, list):
            titles.extend([str(x) for x in ts])
        return list(dict.fromkeys(titles))

    if isinstance(sf, list):
        for item in sf:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                titles.append(str(item[0]))
            elif isinstance(item, dict) and "title" in item:
                titles.append(str(item["title"]))
        return list(dict.fromkeys(titles))

    return list(dict.fromkeys(titles))


def build_docs_and_query(sample: dict, doc_mode: str) -> Tuple[List[str], str]:
    docs = _extract_docs_from_context(sample.get("context"))
    if doc_mode == "gold_only":
        gold_titles = set(_extract_gold_titles(sample))
        docs = [d for d in docs if d[0] in gold_titles]

    doc_texts: List[str] = []
    for title, sents in docs:
        doc_text = f"[Title] {title}\n" + " ".join(sents)
        doc_texts.append(doc_text)

    q = str(sample.get("question", "")).strip()
    query_text = f"Question: {q}\nAnswer:"
    return doc_texts, query_text


# -------------------------
# KV-cache helpers
# -------------------------

@torch.no_grad()
def prefill_cache(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_values=None,
    dtype_autocast: Optional[torch.dtype] = None,
):
    ctx = (
        torch.autocast(device_type="cuda", dtype=dtype_autocast)
        if (dtype_autocast is not None and torch.cuda.is_available())
        else torch.no_grad()
    )
    with ctx:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
    return out.past_key_values


def fuse_past_key_values(pkv_list: List[Any]):
    if not pkv_list:
        raise ValueError("pkv_list is empty")

    legacy_list = [cache_to_legacy(p) for p in pkv_list]
    num_layers = len(legacy_list[0])
    fused = []
    for l in range(num_layers):
        Ks = [pkv[l][0] for pkv in legacy_list]
        Vs = [pkv[l][1] for pkv in legacy_list]
        fused_K = torch.cat(Ks, dim=2)
        fused_V = torch.cat(Vs, dim=2)
        fused.append((fused_K, fused_V))
    fused_legacy = tuple(fused)
    return legacy_to_cache(fused_legacy)


@torch.no_grad()
def last_token_attn_row(
    model,
    full_ids: torch.Tensor,
    dtype_autocast: Optional[torch.dtype] = None,
) -> List[float]:
    assert full_ids.ndim == 2 and full_ids.shape[0] == 1
    L = full_ids.shape[1]
    if L < 2:
        return [1.0]

    device = full_ids.device
    prefix = full_ids[:, :-1]
    last = full_ids[:, -1:]

    attn_mask_prefix = torch.ones((1, L - 1), device=device, dtype=torch.long)
    pkv = prefill_cache(model, prefix, attention_mask=attn_mask_prefix, dtype_autocast=dtype_autocast)

    attn_mask_full = torch.ones((1, L), device=device, dtype=torch.long)
    ctx = (
        torch.autocast(device_type="cuda", dtype=dtype_autocast)
        if (dtype_autocast is not None and torch.cuda.is_available())
        else torch.no_grad()
    )
    with ctx:
        out = model(
            input_ids=last,
            attention_mask=attn_mask_full,
            past_key_values=pkv,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )

    if out.attentions is None:
        raise RuntimeError(
            "Model did not return attentions. Ensure eager attention is enabled "
            "(model.set_attn_implementation('eager')) and that Transformers version supports it."
        )

    attn = out.attentions[-1]  # [B, H, 1, L]
    if attn.ndim != 4:
        raise RuntimeError(f"Unexpected attention shape: {tuple(attn.shape)}")
    attn_mean = attn[0].mean(dim=0)[0]  # [L]
    return attn_mean.detach().float().cpu().tolist()


@torch.no_grad()
def last_token_attn_row_naive_reuse(
    model,
    doc_ids_list: List[torch.Tensor],
    query_ids: torch.Tensor,
    dtype_autocast: Optional[torch.dtype] = None,
) -> List[float]:
    assert query_ids.ndim == 2 and query_ids.shape[0] == 1
    device = query_ids.device

    pkv_docs = []
    doc_lens = []
    for doc_ids in doc_ids_list:
        assert doc_ids.ndim == 2 and doc_ids.shape[0] == 1
        dlen = doc_ids.shape[1]
        if dlen == 0:
            continue
        attn_mask = torch.ones((1, dlen), device=device, dtype=torch.long)
        pkv = prefill_cache(model, doc_ids, attention_mask=attn_mask, dtype_autocast=dtype_autocast)
        pkv_docs.append(pkv)
        doc_lens.append(dlen)

    docs_len = int(sum(doc_lens))
    if docs_len == 0:
        return last_token_attn_row(model, query_ids, dtype_autocast=dtype_autocast)

    pkv_fused = fuse_past_key_values(pkv_docs)

    qlen = query_ids.shape[1]
    if qlen < 2:
        full_len = docs_len + qlen
        attn_mask_full = torch.ones((1, full_len), device=device, dtype=torch.long)
        pos_last = torch.tensor([[docs_len]], device=device, dtype=torch.long)
        ctx = (
            torch.autocast(device_type="cuda", dtype=dtype_autocast)
            if (dtype_autocast is not None and torch.cuda.is_available())
            else torch.no_grad()
        )
        with ctx:
            out = model(
                input_ids=query_ids,
                attention_mask=attn_mask_full,
                position_ids=pos_last,
                past_key_values=pkv_fused,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
        if out.attentions is None:
            raise RuntimeError("Model did not return attentions in naive_reuse path. Ensure eager attention is enabled.")
        attn = out.attentions[-1]
        attn_mean = attn[0].mean(dim=0)[0]
        return attn_mean.detach().float().cpu().tolist()

    q_prefix = query_ids[:, :-1]
    q_last = query_ids[:, -1:]
    q_prefix_len = q_prefix.shape[1]

    attn_mask_prefix = torch.ones((1, docs_len + q_prefix_len), device=device, dtype=torch.long)
    pos_prefix = torch.arange(docs_len, docs_len + q_prefix_len, device=device, dtype=torch.long).unsqueeze(0)

    pkv2 = prefill_cache(
        model,
        q_prefix,
        attention_mask=attn_mask_prefix,
        position_ids=pos_prefix,
        past_key_values=pkv_fused,
        dtype_autocast=dtype_autocast,
    )

    full_len = docs_len + qlen
    attn_mask_full = torch.ones((1, full_len), device=device, dtype=torch.long)
    pos_last = torch.tensor([[docs_len + q_prefix_len]], device=device, dtype=torch.long)

    ctx = (
        torch.autocast(device_type="cuda", dtype=dtype_autocast)
        if (dtype_autocast is not None and torch.cuda.is_available())
        else torch.no_grad()
    )
    with ctx:
        out = model(
            input_ids=q_last,
            attention_mask=attn_mask_full,
            position_ids=pos_last,
            past_key_values=pkv2,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )

    if out.attentions is None:
        raise RuntimeError("Model did not return attentions in naive_reuse final-step. Ensure eager attention is enabled.")

    attn = out.attentions[-1]
    attn_mean = attn[0].mean(dim=0)[0]
    return attn_mean.detach().float().cpu().tolist()


# -------------------------
# Plotting
# -------------------------

def save_hist(vals: List[float], title: str, xlabel: str, out_path: Path, dpi: int = 300, bins: int = 50):
    plt.figure()
    plt.hist(vals, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def save_hist2(a: List[float], b: List[float], labels: Tuple[str, str], title: str, xlabel: str, out_path: Path, dpi: int = 300, bins: int = 50):
    plt.figure()
    plt.hist(a, bins=bins, alpha=0.7, label=labels[0])
    plt.hist(b, bins=bins, alpha=0.7, label=labels[1])
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


# -------------------------
# Dataset loading
# -------------------------

def _parse_split_slice(split: str):
    split = split.strip()
    if "[" not in split:
        return split, None, None, False

    base = split.split("[", 1)[0].strip()
    inside = split.split("[", 1)[1].rsplit("]", 1)[0].strip()

    if ":" not in inside:
        inside = inside.strip()
        if inside.endswith("%"):
            frac = float(inside[:-1]) / 100.0
            return base, frac, frac, True
        idx = int(inside)
        return base, idx, idx + 1, False

    a, b = inside.split(":", 1)
    a = a.strip()
    b = b.strip()

    def parse_endpoint(x: str):
        if x == "":
            return None, False
        if x.endswith("%"):
            return float(x[:-1]) / 100.0, True
        return int(x), False

    start, sp = parse_endpoint(a)
    stop, ep = parse_endpoint(b)

    is_percent = sp or ep
    if is_percent:
        if start is None:
            start = 0.0
        if stop is None:
            stop = 1.0
        return base, float(start), float(stop), True

    return base, start, stop, False


def _apply_slice(ds: Dataset, split: str) -> Dataset:
    base, start, stop, is_percent = _parse_split_slice(split)
    if base != split and hasattr(ds, "select"):
        n = len(ds)
        if is_percent:
            s = int(round(n * float(start))) if start is not None else 0
            e = int(round(n * float(stop))) if stop is not None else n
        else:
            s = int(start) if start is not None else 0
            e = int(stop) if stop is not None else n
        s = max(0, min(n, s))
        e = max(0, min(n, e))
        if e < s:
            e = s
        return ds.select(list(range(s, e)))
    return ds


def load_hf_dataset(name: str, config: Optional[str], path: Optional[str], split: str) -> Dataset:
    if path:
        loaded = load_from_disk(path)
        if isinstance(loaded, DatasetDict):
            base, _, _, _ = _parse_split_slice(split)
            if base not in loaded:
                raise KeyError(f"Split '{base}' not found in dataset on disk. Available: {list(loaded.keys())}")
            return _apply_slice(loaded[base], split)
        if isinstance(loaded, Dataset):
            return _apply_slice(loaded, split)
        raise ValueError(f"Unsupported dataset object from disk: {type(loaded)}")

    ds = load_dataset(name, config, split=split)
    if isinstance(ds, DatasetDict):
        base, _, _, _ = _parse_split_slice(split)
        return ds[base]
    return ds


# -------------------------
# Main
# -------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--model_a_path", type=str, required=True, help="Path or HF id for the (single) model")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])

    p.add_argument("--dataset_name", type=str, default="hotpotqa")
    p.add_argument("--dataset_config", type=str, default="distractor")
    p.add_argument("--dataset_path", type=str, default=None)
    p.add_argument("--split", type=str, default="train[:1%]")
    p.add_argument("--max_samples", type=int, default=100, help="Number of *kept* samples to collect")

    p.add_argument("--doc_mode", type=str, default="all", choices=["all", "gold_only"],
                   help="Use all docs in context (incl. distractors) or only gold docs from supporting_facts")

    # NEW: control mode
    p.add_argument("--control_mode", type=str, default="none", choices=["none", "recompute_self"],
                   help="none: recompute vs naive_reuse; recompute_self: recompute vs recompute (run twice)")

    p.add_argument("--max_L", type=int, default=2050, help="Drop sample if total token length > max_L")

    p.add_argument("--dump_path", type=str2bool, default=True)
    p.add_argument("--output_jsonl", type=str, default=None)
    p.add_argument("--output_plot", type=str, default="kvreuse_metrics.png", help="Prefix for output plots")
    p.add_argument("--plot_dpi", type=int, default=300)

    p.add_argument("--device_map_a", type=str, default="auto")
    p.add_argument("--max_memory_a", type=str, default=None)
    p.add_argument("--use_all_visible_gpus_a", type=str2bool, default=True)
    p.add_argument("--reserve_gpu0_a", type=float, default=2.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--empty_cache_each_sample", type=str2bool, default=True)

    return p


def load_model_with_fallback(
    model_path: str,
    requested_dtype: torch.dtype,
    device_map: str,
    max_memory: Optional[Dict[int, str]],
):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=requested_dtype,
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=True,
        )
        return model, requested_dtype
    except Exception as e:
        msg = str(e)
        if requested_dtype == torch.bfloat16 and _looks_like_bf16_unsupported(msg):
            print("[WARN] Requested dtype=bfloat16 but model load indicates bf16 unsupported on this setup.")
            print("[WARN] Falling back to float16 and continuing.")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=device_map,
                max_memory=max_memory,
                trust_remote_code=True,
            )
            return model, torch.float16
        raise


def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)

    requested_dtype = str2dtype(args.dtype)

    ds = load_hf_dataset(args.dataset_name, args.dataset_config, args.dataset_path, args.split)

    tok = AutoTokenizer.from_pretrained(args.model_a_path, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    max_mem = parse_max_memory(args.max_memory_a)
    if args.use_all_visible_gpus_a:
        max_mem = ensure_max_memory_uses_all_visible_gpus(max_mem, reserve_gpu0_gib=float(args.reserve_gpu0_a))

    model, effective_dtype = load_model_with_fallback(
        args.model_a_path,
        requested_dtype=requested_dtype,
        device_map=args.device_map_a,
        max_memory=max_mem,
    )
    model.eval()
    set_attn_impl_eager(model)

    print(f"[INFO] Requested dtype: {dtype_name(requested_dtype)} | Effective dtype: {dtype_name(effective_dtype)}")
    print(f"[INFO] doc_mode={args.doc_mode} | control_mode={args.control_mode}")

    out_jsonl = args.output_jsonl
    if out_jsonl is None:
        out_jsonl = f"kvreuse_{args.dataset_name}_{args.dataset_config}_{args.doc_mode}_{args.control_mode}.jsonl"
    out_jsonl = Path(out_jsonl)

    plot_prefix = Path(args.output_plot)
    if plot_prefix.suffix.lower() == ".png":
        plot_prefix = plot_prefix.with_suffix("")
    # keep plots separate across control modes
    plot_prefix = plot_prefix.parent / f"{plot_prefix.name}_{args.doc_mode}_{args.control_mode}"

    kept = 0
    dropped = 0
    tried = 0

    # Pairwise metrics (always single-list)
    jsds: List[float] = []
    pears: List[float] = []

    # Entropy is per-condition (two lists)
    ent_a: List[float] = []
    ent_b: List[float] = []

    out_f = out_jsonl.open("w", encoding="utf-8") if args.dump_path else None
    pbar = tqdm(total=args.max_samples, desc=f"Collecting kept samples ({args.doc_mode} | {args.control_mode})")

    dtype_autocast = effective_dtype if effective_dtype != torch.float32 else None

    for sample_index, sample in enumerate(ds):
        if kept >= args.max_samples:
            break

        tried += 1

        doc_texts, query_text = build_docs_and_query(sample, doc_mode=args.doc_mode)

        doc_ids_list: List[torch.Tensor] = []
        total_len = 0
        for dt in doc_texts:
            ids = tok(dt, add_special_tokens=False, return_tensors="pt")["input_ids"]
            if ids.numel() == 0:
                continue
            doc_ids_list.append(ids)
            total_len += ids.shape[1]

        q_ids = tok(query_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        total_len += q_ids.shape[1]

        if total_len > args.max_L:
            dropped += 1
            continue

        full_ids_cpu = torch.cat([*doc_ids_list, q_ids], dim=1) if doc_ids_list else q_ids

        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        full_ids = full_ids_cpu.to(device)
        q_ids_dev = q_ids.to(device)
        doc_ids_list_dev = [d.to(device) for d in doc_ids_list]

        try:
            if args.control_mode == "recompute_self":
                # Control: recompute vs recompute (run twice)
                a1 = last_token_attn_row(model, full_ids, dtype_autocast=dtype_autocast)
                a2 = last_token_attn_row(model, full_ids, dtype_autocast=dtype_autocast)

                jsd = js_divergence(a1, a2)
                pr = pearson_corr(a1, a2)
                h1 = entropy(a1)
                h2 = entropy(a2)

                rec = {
                    "sample_index": int(sample_index),
                    "doc_mode": args.doc_mode,
                    "control_mode": args.control_mode,
                    "total_tokens": int(total_len),
                    "num_docs": int(len(doc_texts)),
                    "jsd": jsd,
                    "pearson": pr,
                    "entropy_run1": h1,
                    "entropy_run2": h2,
                    "requested_dtype": dtype_name(requested_dtype),
                    "effective_dtype": dtype_name(effective_dtype),
                }

                if out_f is not None:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if jsd is not None:
                    jsds.append(jsd)
                if pr is not None:
                    pears.append(pr)
                if h1 is not None:
                    ent_a.append(h1)
                if h2 is not None:
                    ent_b.append(h2)

            else:
                # Default: recompute vs naive_reuse
                a = last_token_attn_row(model, full_ids, dtype_autocast=dtype_autocast)
                b = last_token_attn_row_naive_reuse(model, doc_ids_list_dev, q_ids_dev, dtype_autocast=dtype_autocast)

                jsd = js_divergence(a, b)
                pr = pearson_corr(a, b)
                ha = entropy(a)
                hb = entropy(b)

                rec = {
                    "sample_index": int(sample_index),
                    "doc_mode": args.doc_mode,
                    "control_mode": args.control_mode,
                    "total_tokens": int(total_len),
                    "num_docs": int(len(doc_texts)),
                    "jsd": jsd,
                    "pearson": pr,
                    "entropy_recompute": ha,
                    "entropy_naive_reuse": hb,
                    "requested_dtype": dtype_name(requested_dtype),
                    "effective_dtype": dtype_name(effective_dtype),
                }

                if out_f is not None:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if jsd is not None:
                    jsds.append(jsd)
                if pr is not None:
                    pears.append(pr)
                if ha is not None:
                    ent_a.append(ha)
                if hb is not None:
                    ent_b.append(hb)

            kept += 1
            pbar.update(1)

        except RuntimeError as e:
            msg = str(e)

            # bf16 forward fallback (best-effort)
            if effective_dtype == torch.bfloat16 and _looks_like_bf16_unsupported(msg):
                print("[WARN] bf16 seems unsupported during forward. Falling back to float16 and retrying this sample.")
                try:
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    model, effective_dtype = load_model_with_fallback(
                        args.model_a_path,
                        requested_dtype=torch.float16,
                        device_map=args.device_map_a,
                        max_memory=max_mem,
                    )
                    model.eval()
                    set_attn_impl_eager(model)
                    dtype_autocast = torch.float16

                    # retry current sample
                    if args.control_mode == "recompute_self":
                        a1 = last_token_attn_row(model, full_ids, dtype_autocast=dtype_autocast)
                        a2 = last_token_attn_row(model, full_ids, dtype_autocast=dtype_autocast)
                        jsd = js_divergence(a1, a2)
                        pr = pearson_corr(a1, a2)
                        h1 = entropy(a1)
                        h2 = entropy(a2)
                        rec = {
                            "sample_index": int(sample_index),
                            "doc_mode": args.doc_mode,
                            "control_mode": args.control_mode,
                            "total_tokens": int(total_len),
                            "num_docs": int(len(doc_texts)),
                            "jsd": jsd,
                            "pearson": pr,
                            "entropy_run1": h1,
                            "entropy_run2": h2,
                            "requested_dtype": dtype_name(requested_dtype),
                            "effective_dtype": dtype_name(effective_dtype),
                            "note": "bf16_forward_failed_fallback_to_fp16",
                        }
                        if out_f is not None:
                            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        if jsd is not None:
                            jsds.append(jsd)
                        if pr is not None:
                            pears.append(pr)
                        if h1 is not None:
                            ent_a.append(h1)
                        if h2 is not None:
                            ent_b.append(h2)
                    else:
                        a = last_token_attn_row(model, full_ids, dtype_autocast=dtype_autocast)
                        b = last_token_attn_row_naive_reuse(model, doc_ids_list_dev, q_ids_dev, dtype_autocast=dtype_autocast)
                        jsd = js_divergence(a, b)
                        pr = pearson_corr(a, b)
                        ha = entropy(a)
                        hb = entropy(b)
                        rec = {
                            "sample_index": int(sample_index),
                            "doc_mode": args.doc_mode,
                            "control_mode": args.control_mode,
                            "total_tokens": int(total_len),
                            "num_docs": int(len(doc_texts)),
                            "jsd": jsd,
                            "pearson": pr,
                            "entropy_recompute": ha,
                            "entropy_naive_reuse": hb,
                            "requested_dtype": dtype_name(requested_dtype),
                            "effective_dtype": dtype_name(effective_dtype),
                            "note": "bf16_forward_failed_fallback_to_fp16",
                        }
                        if out_f is not None:
                            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        if jsd is not None:
                            jsds.append(jsd)
                        if pr is not None:
                            pears.append(pr)
                        if ha is not None:
                            ent_a.append(ha)
                        if hb is not None:
                            ent_b.append(hb)

                    kept += 1
                    pbar.update(1)
                    continue
                except Exception:
                    pass

            low = msg.lower()
            if "out of memory" in low or ("cuda" in low and "memory" in low):
                dropped += 1
            else:
                dropped += 1

            if args.empty_cache_each_sample and torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        finally:
            if args.empty_cache_each_sample and torch.cuda.is_available():
                torch.cuda.empty_cache()

    pbar.close()
    if out_f is not None:
        out_f.close()

    print("\n=== Summary ===")
    print(f"Tried samples: {tried}")
    print(f"Kept samples: {kept}")
    print(f"Dropped samples (len>{args.max_L} or OOM/err): {dropped}")
    print(f"Output JSONL: {out_jsonl if args.dump_path else '(disabled)'}")
    print(f"Final effective dtype: {dtype_name(effective_dtype)}")

    # Plot titles/labels depend on control mode
    if jsds:
        if args.control_mode == "recompute_self":
            save_hist(jsds, f"JSD (recompute vs recompute) | {args.doc_mode}", "JSD",
                      plot_prefix.parent / f"{plot_prefix.name}_jsd.png", dpi=args.plot_dpi)
        else:
            save_hist(jsds, f"JSD (recompute vs naive reuse) | {args.doc_mode}", "JSD",
                      plot_prefix.parent / f"{plot_prefix.name}_jsd.png", dpi=args.plot_dpi)

    if pears:
        if args.control_mode == "recompute_self":
            save_hist(pears, f"Pearson (recompute vs recompute) | {args.doc_mode}", "Pearson r",
                      plot_prefix.parent / f"{plot_prefix.name}_pearson.png", dpi=args.plot_dpi)
        else:
            save_hist(pears, f"Pearson (recompute vs naive reuse) | {args.doc_mode}", "Pearson r",
                      plot_prefix.parent / f"{plot_prefix.name}_pearson.png", dpi=args.plot_dpi)

    if ent_a and ent_b:
        if args.control_mode == "recompute_self":
            save_hist2(ent_a, ent_b, ("recompute_run1", "recompute_run2"),
                       f"Entropy | {args.doc_mode}", "Entropy",
                       plot_prefix.parent / f"{plot_prefix.name}_entropy.png", dpi=args.plot_dpi)
        else:
            save_hist2(ent_a, ent_b, ("recompute", "naive_reuse"),
                       f"Entropy | {args.doc_mode}", "Entropy",
                       plot_prefix.parent / f"{plot_prefix.name}_entropy.png", dpi=args.plot_dpi)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    main()
