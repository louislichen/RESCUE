#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Length sweep: compare attention similarity between two models as context length grows.

Route A (strict): build each group until token length >= required_L,
so every group contributes to every sampled L.

Multi-GPU: shard model weights across multiple GPUs using device_map/max_memory.

JSONL format (per group, per L):
{
  "sample_index": int,
  "L": int,
  "prompt": str,
  "generated": "",
  "generated_tokens": [],
  "last_layer_attention_a": [[...]] or null,
  "last_layer_attention_b": [[...]] or null,
  "pearson": float or null,
  "spearman": float or null,
  "jsd": float or null,
  "group_total_tokens": int,
  "group_num_segments": int
}
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------
# Utils: args / dtype / bool
# ----------------------------
def str2bool(val):
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {val}")


def str2dtype(name: str):
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def parse_lengths(s: str) -> List[int]:
    parts = [p.strip() for p in s.replace(" ", ",").split(",") if p.strip()]
    out = []
    for p in parts:
        v = int(p)
        if v > 1:
            out.append(v)
    return sorted(list(dict.fromkeys(out)))


def build_sweep_lengths(step: int, max_L: int) -> List[int]:
    if step <= 0:
        raise ValueError("sweep_step must be > 0")
    if max_L <= 1:
        raise ValueError("max_L must be > 1")
    Ls = list(range(step, max_L + 1, step))
    if Ls[-1] != max_L:
        Ls.append(max_L)
    return sorted(list(dict.fromkeys(Ls)))


def parse_max_memory(s: str) -> Optional[Dict[int, str]]:
    if s is None:
        return None
    s = s.strip()
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
    if unit in {"kib", "kb", "k"}:
        return val / (1024.0 * 1024.0)
    raise ValueError(f"Unsupported unit in memory string: '{s}'")


def fmt_gib(x: float) -> str:
    if abs(x - round(x)) < 1e-6:
        return f"{int(round(x))}GiB"
    return f"{x:.2f}GiB"


def ensure_max_memory_uses_all_visible_gpus(
    max_mem: Optional[Dict[int, str]],
    reserve_gpu0_gib: float = 0.0,
    only_if_cuda: bool = True,
) -> Optional[Dict[int, str]]:
    if only_if_cuda and (not torch.cuda.is_available()):
        return max_mem

    n = torch.cuda.device_count()
    if n <= 0:
        return max_mem

    if max_mem is None:
        return None

    out = dict(max_mem)

    candidates = [k for k in sorted(out.keys()) if k != 0]
    base_key = candidates[0] if candidates else (0 if 0 in out else None)
    base_budget = out[base_key] if base_key is not None else "10GiB"

    for gid in range(n):
        if gid not in out:
            out[gid] = base_budget

    if reserve_gpu0_gib > 0.0 and 0 in out:
        try:
            cur = parse_gib(out[0])
            new = max(0.5, cur - reserve_gpu0_gib)
            out[0] = fmt_gib(new)
        except Exception:
            pass

    return out


def is_finite_list(xs: List[float]) -> bool:
    for x in xs:
        if not math.isfinite(float(x)):
            return False
    return True


# ----------------------------
# Stats: Pearson & Spearman
# ----------------------------
def pearson_corr(a: List[float], b: List[float]) -> Optional[float]:
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
    return cov / denom


def rankdata_average_ties(x: List[float]) -> List[float]:
    n = len(x)
    order = sorted(range(n), key=lambda i: x[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(a: List[float], b: List[float]) -> Optional[float]:
    n = min(len(a), len(b))
    if n <= 1:
        return None
    a = a[:n]
    b = b[:n]
    ra = rankdata_average_ties(a)
    rb = rankdata_average_ties(b)
    return pearson_corr(ra, rb)


# ----------------------------
# Divergence: Jensen-Shannon (distribution distance)
# ----------------------------
def _normalize_prob(x: List[float], eps: float = 1e-12) -> Optional[List[float]]:
    if not x:
        return None
    y = [0.0 if (v is None or not math.isfinite(float(v)) or float(v) < 0.0) else float(v) for v in x]
    s = sum(y)
    if s <= 0.0:
        return None
    y = [v / s for v in y]
    y = [max(eps, v) for v in y]
    s2 = sum(y)
    if s2 <= 0.0:
        return None
    return [v / s2 for v in y]


def js_divergence(p: List[float], q: List[float], eps: float = 1e-12, log_base: float = 2.0) -> Optional[float]:
    n = min(len(p), len(q))
    if n <= 1:
        return None
    p = _normalize_prob(p[:n], eps=eps)
    q = _normalize_prob(q[:n], eps=eps)
    if p is None or q is None:
        return None

    m = [(pi + qi) * 0.5 for pi, qi in zip(p, q)]

    def kl(a, b):
        out = 0.0
        for ai, bi in zip(a, b):
            out += ai * (math.log(ai / bi) / math.log(log_base))
        return out

    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    if not math.isfinite(jsd):
        return None
    return max(0.0, float(jsd))


def mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    v = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return m, math.sqrt(v)


# ----------------------------
# Dataset helpers
# ----------------------------
def resolve_split(ds_container: DatasetDict, split_expr: str):
    if "[" not in split_expr:
        return ds_container[split_expr]

    base, bracket = split_expr.split("[", 1)
    bracket = bracket.rstrip("]")
    if base not in ds_container:
        raise ValueError(f"Split base '{base}' not found in local dataset.")
    ds = ds_container[base]
    total = len(ds)

    def parse_bound(tok: str):
        tok = tok.strip()
        if tok == "":
            return None
        if tok.endswith("%"):
            return int(float(tok[:-1]) / 100.0 * total)
        return int(tok)

    start_s, end_s = bracket.split(":", 1)
    start = parse_bound(start_s)
    end = parse_bound(end_s)
    if start is None:
        start = 0
    if end is None or end > total:
        end = total
    start = max(0, start)
    end = max(start, min(end, total))
    return ds.select(range(start, end))


def load_ds(dataset_name: str, dataset_config: str, split: str, dataset_path: Optional[str]) -> Dataset:
    if dataset_path:
        p = Path(dataset_path).expanduser()
        if p.exists():
            try:
                loaded = load_from_disk(p)
                if isinstance(loaded, Dataset):
                    ds_container = DatasetDict({"train": loaded})
                elif isinstance(loaded, DatasetDict):
                    ds_container = loaded
                else:
                    raise ValueError(f"Unsupported dataset object from disk: {type(loaded)}")
                return resolve_split(ds_container, split)
            except Exception as e:
                print(f"!! Warning: failed to load dataset from '{p}': {e}")
                print("!! Fallback to HuggingFace load_dataset(...)")
        else:
            print(f"!! Warning: dataset_path '{p}' does not exist.")
            print("!! Fallback to HuggingFace load_dataset(...)")

    return load_dataset(dataset_name, dataset_config, split=split)


# ----------------------------
# Model loading (multi-GPU sharding)
# ----------------------------
def load_model_sharded(
    model_path: str,
    dtype: torch.dtype,
    device_map: str,
    max_memory: Optional[Dict[int, str]],
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
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


def summarize_device_map(model, name: str):
    dm = getattr(model, "hf_device_map", None)
    if not dm:
        print(f">>> {name}: no hf_device_map found (maybe single GPU/CPU).")
        return
    counts: Dict[str, int] = {}
    for _, dev in dm.items():
        counts[str(dev)] = counts.get(str(dev), 0) + 1
    pretty = ", ".join([f"{k}:{v}" for k, v in sorted(counts.items())])
    print(f">>> {name} hf_device_map module counts: {pretty}")


def get_input_device(model: AutoModelForCausalLM) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


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


# ----------------------------
# last-layer attention via self_attn hook
# sample multiple rows in one forward (each row moved to CPU)
# ----------------------------
def _find_decoder_layers_list(model: AutoModelForCausalLM):
    candidates = [
        (lambda m: m.model.layers),
        (lambda m: m.model.model.layers),
        (lambda m: m.transformer.h),
        (lambda m: m.gpt_neox.layers),
        (lambda m: m.model.decoder.layers),
    ]
    for getter in candidates:
        try:
            layers = getter(model)
            if isinstance(layers, (list, torch.nn.ModuleList)) and len(layers) > 0:
                return layers
        except Exception:
            continue
    raise RuntimeError("Could not locate decoder layers list in this model.")


@torch.no_grad()
def sampled_attention_rows_last_layer(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,                       # [1, Lmax]
    attention_mask: Optional[torch.Tensor],         # [1, Lmax] or None
    sample_Ls: List[int],                           # e.g. [10,20,...,2050]
    attention_precision: int,
) -> Dict[int, Optional[List[float]]]:
    layers = _find_decoder_layers_list(model)
    last_layer = layers[-1]
    if not hasattr(last_layer, "self_attn"):
        return {L: None for L in sample_Ls}

    Lmax = int(input_ids.shape[-1])
    sample_Ls = [int(L) for L in sample_Ls if 1 < int(L) <= Lmax]
    sample_Ls = sorted(list(dict.fromkeys(sample_Ls)))

    row_indices = [L - 1 for L in sample_Ls]
    row_to_L = {r: (r + 1) for r in row_indices}

    cache: Dict[int, Optional[List[float]]] = {L: None for L in sample_Ls}

    def hook_fn(module, inputs, output):
        attn_weights = None
        if isinstance(output, tuple) and len(output) >= 2 and torch.is_tensor(output[1]):
            attn_weights = output[1]
        else:
            return

        if attn_weights.dim() != 4:
            return

        for r in row_indices:
            L = row_to_L[r]
            v = attn_weights[0, :, r, :L].float().mean(dim=0).detach().cpu()
            if not torch.isfinite(v).all():
                cache[L] = None
                continue
            cache[L] = [round(float(x), attention_precision) for x in v.tolist()]

    h = last_layer.self_attn.register_forward_hook(hook_fn)
    try:
        _ = model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=False,
            use_cache=False,
        )
    finally:
        h.remove()

    return cache


def auto_dump_name(model_a_path: str, model_b_path: str, base_dir: Path) -> Path:
    a = Path(model_a_path).name or "A"
    b = Path(model_b_path).name or "B"

    def safe(s: str) -> str:
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in s)

    stem = f"length_sweep_{safe(a)}__vs__{safe(b)}.jsonl"
    out = base_dir / stem
    k = 1
    while out.exists():
        out = base_dir / stem.replace(".jsonl", f"_{k}.jsonl")
        k += 1
    return out


# ----------------------------
# Plot helpers (NEW)
# ----------------------------
def _nice_ylim(data_mean: List[float], data_std: List[float], clamp: Optional[Tuple[float, float]] = None):
    ys = []
    for m, s in zip(data_mean, data_std):
        ys.append(m - s)
        ys.append(m + s)
    ys = [y for y in ys if math.isfinite(float(y))]
    if not ys:
        return None

    lo = min(ys)
    hi = max(ys)
    if lo == hi:
        lo -= 0.05
        hi += 0.05

    # add margin
    span = hi - lo
    lo -= 0.06 * span
    hi += 0.06 * span

    if clamp is not None:
        lo = max(clamp[0], lo)
        hi = min(clamp[1], hi)
        # if clamped to a point, expand a bit
        if lo >= hi:
            mid = 0.5 * (clamp[0] + clamp[1])
            lo, hi = mid - 0.1, mid + 0.1

    return lo, hi


def save_single_metric_plot(
    xs: List[int],
    means: List[float],
    stds: List[float],
    title: str,
    ylabel: str,
    color: str,
    out_path: Path,
    ylim: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
):
    # Professional-ish defaults (no seaborn)
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    # Mean line
    ax.plot(xs, means, marker="o", markersize=3.5, linewidth=2.0, color=color, label="Mean")

    # Std band (more readable than dense error bars)
    lower = [m - s for m, s in zip(means, stds)]
    upper = [m + s for m, s in zip(means, stds)]
    ax.fill_between(xs, lower, upper, color=color, alpha=0.18, linewidth=0.0, label="±1 std")

    ax.set_title(title)
    ax.set_xlabel("Context length L (tokens)")
    ax.set_ylabel(ylabel)

    # Grid: major + minor
    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", alpha=0.18)

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    # cleaner legend
    ax.legend(loc="best", frameon=True, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _derive_plot_paths(output_plot: str) -> Tuple[Path, Path, Path]:
    """
    If output_plot="length_sweep_corr.png",
    save:
      length_sweep_corr_pearson.png
      length_sweep_corr_spearman.png
      length_sweep_corr_jsd.png
    """
    p = Path(output_plot)
    stem = p.stem
    suffix = p.suffix if p.suffix else ".png"
    base_dir = p.parent if str(p.parent) != "" else Path(".")
    pear = base_dir / f"{stem}_pearson{suffix}"
    spear = base_dir / f"{stem}_spearman{suffix}"
    jsd = base_dir / f"{stem}_jsd{suffix}"
    return pear, spear, jsd


# ----------------------------
# Main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser("Length sweep attention similarity (Route A + Multi-GPU + JSONL + Plot)")

    p.add_argument("--model_a_path", type=str, required=True)
    p.add_argument("--model_b_path", type=str, required=True)

    p.add_argument("--dataset_name", type=str, default="wikitext")
    p.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--dataset_path", type=str, default=None)
    p.add_argument("--split", type=str, default="train[:1%]")
    p.add_argument("--max_samples", type=int, default=100)

    p.add_argument("--lengths", type=str, default="32,64,128,256,384,512")
    p.add_argument("--sweep_step", type=int, default=0, help="If >0, ignore --lengths and sweep to --max_L.")
    p.add_argument("--max_L", type=int, default=2050, help="Upper bound when --sweep_step>0.")

    p.add_argument("--concat_k", type=int, default=0, help="(Ignored in Route A) kept for compatibility.")

    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attention_precision", type=int, default=6)
    p.add_argument("--max_prompt_chars", type=int, default=2000)

    p.add_argument("--min_group_tokens", type=int, default=None)

    p.add_argument("--dump_path", type=str2bool, default=True)
    p.add_argument("--output_jsonl", type=str, default=None)

    # still keep one param, but now it becomes the "base name"
    p.add_argument("--output_plot", type=str, default="length_sweep_corr.png")

    # NEW: nicer plot control
    p.add_argument("--plot_dpi", type=int, default=300, help="PNG DPI (paper-like default: 300)")

    p.add_argument("--device_map_a", type=str, default="auto")
    p.add_argument("--device_map_b", type=str, default="auto")

    p.add_argument("--max_memory_a", type=str, default="0:10GiB,1:10GiB,2:10GiB,3:10GiB")
    p.add_argument("--max_memory_b", type=str, default="0:10GiB,1:10GiB,2:10GiB,3:10GiB")

    p.add_argument("--use_all_visible_gpus_a", type=str2bool, default=True)
    p.add_argument("--reserve_gpu0_a", type=float, default=2.0)

    p.add_argument("--empty_cache_each_group", type=str2bool, default=True)

    return p.parse_args()


def main():
    args = parse_args()

    if args.sweep_step and int(args.sweep_step) > 0:
        L_list = build_sweep_lengths(int(args.sweep_step), int(args.max_L))
        print(f">>> Dense sweep enabled: step={args.sweep_step}, max_L={args.max_L}, points={len(L_list)}")
    else:
        L_list = parse_lengths(args.lengths)
        if not L_list:
            raise ValueError("Empty --lengths")
        print(f">>> Discrete lengths enabled: points={len(L_list)} -> {L_list[:10]}{'...' if len(L_list)>10 else ''}")

    required_L = max(L_list)
    if args.min_group_tokens is not None:
        required_L = max(required_L, int(args.min_group_tokens))

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    dtype = str2dtype(args.dtype)

    max_mem_a = parse_max_memory(args.max_memory_a)
    max_mem_b = parse_max_memory(args.max_memory_b)

    if args.use_all_visible_gpus_a:
        max_mem_a = ensure_max_memory_uses_all_visible_gpus(
            max_mem_a,
            reserve_gpu0_gib=float(args.reserve_gpu0_a),
            only_if_cuda=True,
        )

    print(">>> Loading model A (sharded):", args.model_a_path)
    if max_mem_a is not None:
        print(f">>> model A max_memory (after auto-fix): {max_mem_a}")
    tok_a, model_a = load_model_sharded(args.model_a_path, dtype, args.device_map_a, max_mem_a)
    avoid_accelerate_output_gather(model_a)
    summarize_device_map(model_a, "model A")

    print(">>> Loading model B (sharded):", args.model_b_path)
    if max_mem_b is not None:
        print(f">>> model B max_memory: {max_mem_b}")
    tok_b, model_b = load_model_sharded(args.model_b_path, dtype, args.device_map_b, max_mem_b)
    avoid_accelerate_output_gather(model_b)
    summarize_device_map(model_b, "model B")

    tokenizer = tok_a

    print(f">>> Loading dataset {args.dataset_name}/{args.dataset_config}/{args.split}")
    if args.dataset_path:
        print(f">>> dataset_path requested: {args.dataset_path}")
    ds = load_ds(args.dataset_name, args.dataset_config, args.split, args.dataset_path)
    n = len(ds) if hasattr(ds, "__len__") else None

    dump_fh = None
    out_jsonl = None
    if args.dump_path:
        out_jsonl = Path(args.output_jsonl) if args.output_jsonl else auto_dump_name(args.model_a_path, args.model_b_path, Path.cwd())
        print(f">>> Writing JSONL to: {out_jsonl}")
        dump_fh = out_jsonl.open("w", encoding="utf-8")

    pearson_by_L: Dict[int, List[float]] = {L: [] for L in L_list}
    spearman_by_L: Dict[int, List[float]] = {L: [] for L in L_list}
    jsd_by_L: Dict[int, List[float]] = {L: [] for L in L_list}

    print(f">>> Route A enabled: each group token length >= {required_L} (required).")

    total = 0
    group_index = 0
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
            break

        group_text = "\n\n".join(texts)
        enc = tokenizer(group_text, return_tensors="pt", truncation=False)
        input_ids_full = enc["input_ids"][0]
        attn_mask_full = enc["attention_mask"][0] if "attention_mask" in enc else None
        T = int(input_ids_full.shape[0])

        if T < required_L:
            continue

        input_ids_max = input_ids_full[:required_L].unsqueeze(0)
        attn_mask_max = attn_mask_full[:required_L].unsqueeze(0) if attn_mask_full is not None else None

        dev_a = get_input_device(model_a)
        dev_b = get_input_device(model_b)

        input_ids_a = input_ids_max.to(dev_a)
        mask_a = attn_mask_max.to(dev_a) if attn_mask_max is not None else None

        input_ids_b = input_ids_max.to(dev_b)
        mask_b = attn_mask_max.to(dev_b) if attn_mask_max is not None else None

        vecs_a = sampled_attention_rows_last_layer(model_a, input_ids_a, mask_a, L_list, args.attention_precision)
        vecs_b = sampled_attention_rows_last_layer(model_b, input_ids_b, mask_b, L_list, args.attention_precision)

        for L in L_list:
            vec_a = vecs_a.get(L, None)
            vec_b = vecs_b.get(L, None)

            pear = None
            spear = None
            jsd = None

            if vec_a is not None and vec_b is not None and is_finite_list(vec_a) and is_finite_list(vec_b):
                pear = pearson_corr(vec_a, vec_b)
                spear = spearman_corr(vec_a, vec_b)
                jsd = js_divergence(vec_a, vec_b)

                if pear is not None and math.isfinite(pear):
                    pearson_by_L[L].append(float(pear))
                if spear is not None and math.isfinite(spear):
                    spearman_by_L[L].append(float(spear))
                if jsd is not None and math.isfinite(jsd):
                    jsd_by_L[L].append(float(jsd))
            else:
                vec_a = None
                vec_b = None

            prefix_text = tokenizer.decode(input_ids_max[0, :L], skip_special_tokens=False)
            if args.max_prompt_chars and len(prefix_text) > args.max_prompt_chars:
                prefix_text = prefix_text[: args.max_prompt_chars] + " ...[TRUNCATED]"

            if dump_fh is not None:
                obj = {
                    "sample_index": group_index,
                    "L": L,
                    "prompt": prefix_text,
                    "generated": "",
                    "generated_tokens": [],
                    "last_layer_attention_a": [vec_a] if vec_a is not None else None,
                    "last_layer_attention_b": [vec_b] if vec_b is not None else None,
                    "pearson": None if pear is None or (isinstance(pear, float) and not math.isfinite(pear)) else round(float(pear), 6),
                    "spearman": None if spear is None or (isinstance(spear, float) and not math.isfinite(spear)) else round(float(spear), 6),
                    "jsd": None if jsd is None or (isinstance(jsd, float) and not math.isfinite(jsd)) else round(float(jsd), 6),
                    "group_total_tokens": T,
                    "group_num_segments": len(texts),
                }
                dump_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

        del input_ids_a, mask_a, input_ids_b, mask_b, input_ids_max, attn_mask_max
        if args.empty_cache_each_group and torch.cuda.is_available():
            torch.cuda.empty_cache()

        group_index += 1
        total += 1
        pbar.update(1)

    pbar.close()
    if dump_fh is not None:
        dump_fh.close()

    # summarize arrays for plotting
    xs: List[int] = []
    pear_means: List[float] = []
    pear_stds: List[float] = []
    spear_means: List[float] = []
    spear_stds: List[float] = []
    jsd_means: List[float] = []
    jsd_stds: List[float] = []

    for L in L_list:
        mp, sp = mean_std(pearson_by_L[L])
        ms, ss = mean_std(spearman_by_L[L])
        mj, sj = mean_std(jsd_by_L[L])

        if mp is None and ms is None and mj is None:
            continue

        xs.append(L)
        pear_means.append(0.0 if mp is None else float(mp))
        pear_stds.append(0.0 if sp is None else float(sp))
        spear_means.append(0.0 if ms is None else float(ms))
        spear_stds.append(0.0 if ss is None else float(ss))
        jsd_means.append(0.0 if mj is None else float(mj))
        jsd_stds.append(0.0 if sj is None else float(sj))

    # ----------------------------
    # NEW: save 3 separate figures
    # ----------------------------
    if xs:
        out_pear, out_spear, out_jsd = _derive_plot_paths(args.output_plot)

        # choose y-limits
        pear_ylim = _nice_ylim(pear_means, pear_stds, clamp=(-1.0, 1.0))
        spear_ylim = _nice_ylim(spear_means, spear_stds, clamp=(-1.0, 1.0))
        jsd_ylim = _nice_ylim(jsd_means, jsd_stds, clamp=(0.0, 1.0))

        # titles (slightly more informative)
        a_name = Path(args.model_a_path).name
        b_name = Path(args.model_b_path).name
        common_prefix = f"{a_name} vs {b_name} | last-layer attention | sampled rows"

        save_single_metric_plot(
            xs, pear_means, pear_stds,
            title=f"Pearson correlation vs context length\n{common_prefix}",
            ylabel="Pearson correlation",
            color="tab:blue",
            out_path=out_pear,
            ylim=pear_ylim,
            dpi=int(args.plot_dpi),
        )
        print(f">>> Saved plot: {out_pear}")

        save_single_metric_plot(
            xs, spear_means, spear_stds,
            title=f"Spearman correlation vs context length\n{common_prefix}",
            ylabel="Spearman correlation",
            color="tab:orange",
            out_path=out_spear,
            ylim=spear_ylim,
            dpi=int(args.plot_dpi),
        )
        print(f">>> Saved plot: {out_spear}")

        save_single_metric_plot(
            xs, jsd_means, jsd_stds,
            title=f"Jensen–Shannon Divergence vs context length\n{common_prefix}",
            ylabel="JSD (base 2)",
            color="tab:green",
            out_path=out_jsd,
            ylim=jsd_ylim,
            dpi=int(args.plot_dpi),
        )
        print(f">>> Saved plot: {out_jsd}")

    print("\n================ SUMMARY ================")
    print(f"Groups evaluated : {total}")
    for L in L_list:
        mp, sp = mean_std(pearson_by_L[L])
        ms, ss = mean_std(spearman_by_L[L])
        mj, sj = mean_std(jsd_by_L[L])
        cntp = len(pearson_by_L[L])
        cnts = len(spearman_by_L[L])
        cntj = len(jsd_by_L[L])
        if cntp == 0 and cnts == 0 and cntj == 0:
            continue
        print(
            f"L={L:4d} | Pearson n={cntp:3d} mean={mp if mp is not None else 'NA'} std={sp if sp is not None else 'NA'}"
            f" | Spearman n={cnts:3d} mean={ms if ms is not None else 'NA'} std={ss if ss is not None else 'NA'}"
            f" | JSD(base2) n={cntj:3d} mean={mj if mj is not None else 'NA'} std={sj if sj is not None else 'NA'}"
        )
    if out_jsonl:
        print(f"JSONL saved     : {out_jsonl}")
    print("========================================")


if __name__ == "__main__":
    main()
