#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG KV-cache experiment (HotpotQA):

Compare two KV construction routes and evaluate downstream decode accuracy.

Route A (one-shot / recompute):
  - Concatenate all retrieved chunks + query into one prompt
  - One forward (prefill) to build past_key_values
  - Greedy decode to generate answer

Route B (KV reuse simulation / naive fuse):
  - Prefill each chunk independently (position starts from 0 for every chunk)
  - Naively fuse chunk past_key_values by concatenating along seq_len dimension
  - Prefill query with fused past (continuous position_ids)
  - Greedy decode to generate answer

Length control:
  - We DO NOT truncate docs/query with max_doc_tokens/max_query_tokens.
  - Instead, we pre-tokenize and compute lengths for each sample.
  - If max(Route A prompt length, Route B prompt length) > --max_L, we drop the sample.
  - We report seen / evaluated / dropped counts.

Fix in this version (IMPORTANT):
  - Robust HotpotQA context parsing:
    HotpotQA can store sample["context"] as either:
      (1) list of [title, [sent...]] pairs (legacy)
      (2) dict: {"title": [...], "sentences": [[...], ...]} (HF common)

Additional change (CONTROL VARIABLES):
  - Route A and Route B now use the SAME formatting / separators for docs:
      segment_i = "[Doc i]\\n" + doc_text
    and separators are identical (default sep="\\n\\n").
  - In Route B, each doc is prefetched independently using exactly the segment
    as it appears in Route A, PLUS a trailing separator sep (including last doc),
    so the fused "docs part" tokenization matches Route A's docs block before query.
  - Only differences remaining between A/B:
      A: docs are prefetched jointly (docs can attend across docs)
      B: docs are prefetched independently (no cross-doc attention), positions reset per doc

Debug prompts dump (NEW):
  - Keep everything else unchanged.
  - If --debug_print_first n > 0:
      write a new JSONL file with 2n lines (A + B for each of first n evaluated samples),
      storing the fully concatenated prompt strings (not printed to console).
"""

import os
import re
import json
import argparse
import string
from typing import List, Tuple, Optional, Dict, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ----------------------------
# HotpotQA helpers (ROBUST)
# ----------------------------

def _extract_gold_titles(sample: Dict[str, Any]) -> set:
    gold_titles = set()
    for t, _ in sample.get("supporting_facts", []) or []:
        if t is not None:
            gold_titles.add(str(t))
    return gold_titles

def _extract_docs_from_context(ctx: Any) -> List[Tuple[str, List[str]]]:
    """
    Return list of (title, [sentences...]) from HotpotQA context that may be:
      1) list of [title, [sentences...]] pairs (legacy)
      2) dict: {"title": [...], "sentences": [[...], ...]} (HF common)
    """
    docs: List[Tuple[str, List[str]]] = []

    # Case 1: dict form
    if isinstance(ctx, dict):
        titles = ctx.get("title", [])
        sents = ctx.get("sentences", [])
        if isinstance(titles, list) and isinstance(sents, list):
            n = min(len(titles), len(sents))
            for i in range(n):
                title = titles[i]
                ss = sents[i]
                if not isinstance(ss, list):
                    continue
                docs.append((str(title), [str(x) for x in ss]))
        return docs

    # Case 2: list of pairs form
    if isinstance(ctx, list):
        for item in ctx:
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                continue
            title, ss = item
            if not isinstance(ss, list):
                continue
            docs.append((str(title), [str(x) for x in ss]))
        return docs

    return docs

def build_docs_and_query(sample: Dict[str, Any], doc_mode: str = "all") -> Tuple[List[str], str]:
    """
    Build doc_texts (list[str]) and query string.

    doc_mode:
      - all: use all docs in context
      - gold_only: use only docs whose title appears in supporting_facts
    """
    ctx = sample.get("context", None)
    docs = _extract_docs_from_context(ctx)

    gold_titles = _extract_gold_titles(sample) if doc_mode == "gold_only" else set()

    doc_texts: List[str] = []
    for title, ss in docs:
        if doc_mode == "gold_only" and gold_titles and title not in gold_titles:
            continue
        text = (title.strip() + "\n" + " ".join([x for x in ss])).strip()
        if text:
            doc_texts.append(text)

    q = str(sample.get("question", "")).strip()
    query = f"Question: {q}\nAnswer:"
    return doc_texts, query


# ----------------------------
# Normalization + EM/F1
# ----------------------------

_ARTICLES = {"a", "an", "the"}

def _remove_articles(text: str) -> str:
    return " ".join([w for w in text.split() if w not in _ARTICLES])

def _white_space_fix(text: str) -> str:
    return " ".join(text.split())

def _remove_punc(text: str) -> str:
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

def _lower(text: str) -> str:
    return text.lower()

def normalize_answer(s: str) -> str:
    s = str(s)
    return _white_space_fix(_remove_articles(_remove_punc(_lower(s))))

def exact_match_score(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in gold_tokens:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ----------------------------
# Lightweight answer postprocess
# ----------------------------

_NUM_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+|\b\d+\b")

def _looks_like_yesno(gold: str) -> bool:
    g = normalize_answer(gold)
    return g in {"yes", "no"}

def _looks_like_number_answer(gold: str) -> bool:
    g = gold.strip()
    return any(ch.isdigit() for ch in g)

def extract_final_answer(gen_text: str, gold_answer: str) -> str:
    """
    Lightweight extraction:
      - cut off new QA blocks / new docs
      - prefer trailing "Answer:" segment
      - take first line/sentence
      - special-case yes/no and numeric answers
    """
    if gen_text is None:
        return ""

    t = str(gen_text).strip()

    for marker in ["\nQuestion:", "\nQ:", "\n[Doc", "\nDocument "]:
        if marker in t:
            t = t.split(marker, 1)[0].rstrip()

    if "Answer:" in t:
        t = t.split("Answer:")[-1].strip()

    t = t.splitlines()[0].strip()

    for end in [".", "?", "!"]:
        if end in t:
            t = t.split(end, 1)[0].strip()

    if _looks_like_yesno(gold_answer):
        m = re.search(r"\b(yes|no)\b", t, flags=re.IGNORECASE)
        if m:
            return m.group(1).lower()
        m2 = re.search(r"^\s*(yes|no)\b", t, flags=re.IGNORECASE)
        if m2:
            return m2.group(1).lower()

    if _looks_like_number_answer(gold_answer):
        m = _NUM_RE.search(t)
        if m:
            return m.group(0)

    return t


# ----------------------------
# Tokenization + Prompt building (CONTROLLED)
# ----------------------------

def make_doc_segments(doc_texts: List[str]) -> List[str]:
    """
    Each doc segment is exactly how it appears in Route A.
    """
    return [f"[Doc {i+1}]\n{d}" for i, d in enumerate(doc_texts)]

def make_full_prompt_from_segments(doc_segments: List[str], query: str, sep: str = "\n\n") -> str:
    """
    Route A prompt:
      docs_block + sep + query
    """
    docs_block = sep.join(doc_segments)
    return (docs_block + sep + query).strip() if docs_block else query.strip()

def make_docs_part_text_for_b(doc_segments: List[str], sep: str = "\n\n") -> str:
    """
    The text that Route B intends to represent in fused KV BEFORE query.
    We include a trailing sep after the last doc, so that:
      docs_part_text + query  == Route A prompt (same tokenization)
    """
    if not doc_segments:
        return ""
    return sep.join(doc_segments) + sep

def tokenize_text(tokenizer, text: str) -> torch.Tensor:
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,
    )
    return enc["input_ids"]

def pretokenize_for_routes(tokenizer, doc_texts: List[str], query: str, sep: str = "\n\n"):
    """
    Returns:
      full_ids: [1, L_A]
      doc_ids_list: list of [1, Li] (each corresponds to doc_segment + sep)  <-- controlled
      query_ids: [1, Lq]
      L_A, L_B, full_prompt_str, b_prompt_str (docs_part+query)
    """
    doc_segments = make_doc_segments(doc_texts)

    # Route A prompt
    full_prompt = make_full_prompt_from_segments(doc_segments, query, sep=sep)
    full_ids = tokenize_text(tokenizer, full_prompt)
    L_A = int(full_ids.shape[1])

    # Route B doc inputs: each doc segment PLUS a trailing separator sep
    doc_inputs_b = [seg + sep for seg in doc_segments]  # include sep after last doc too
    doc_ids_list = [tokenize_text(tokenizer, s) for s in doc_inputs_b]

    query_ids = tokenize_text(tokenizer, query)
    docs_part_text = make_docs_part_text_for_b(doc_segments, sep=sep)
    b_prompt = (docs_part_text + query).strip() if docs_part_text else query.strip()

    L_B = int(sum(x.shape[1] for x in doc_ids_list) + query_ids.shape[1])
    return full_ids, doc_ids_list, query_ids, L_A, L_B, full_prompt, b_prompt


# ----------------------------
# KV cache utils
# ----------------------------

def _is_cache_obj(past_key_values) -> bool:
    return hasattr(past_key_values, "get_seq_length") or hasattr(past_key_values, "to_legacy_cache")

def _to_legacy_cache(past_key_values):
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values

def _from_legacy_cache(legacy_cache):
    try:
        from transformers.cache_utils import DynamicCache  # type: ignore
        return DynamicCache.from_legacy_cache(legacy_cache)
    except Exception:
        return legacy_cache

def past_seq_len(past_key_values) -> int:
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    legacy = _to_legacy_cache(past_key_values)
    k0 = legacy[0][0]
    return int(k0.shape[2])

def fuse_past_key_values(pasts):
    """Naively fuse past caches by concatenating along seq_len dim=2."""
    assert len(pasts) > 0, "No past_key_values to fuse."

    any_cache_obj = any(_is_cache_obj(p) for p in pasts)
    legacy_pasts = [_to_legacy_cache(p) for p in pasts]

    n_layers = len(legacy_pasts[0])
    fused = []
    for layer in range(n_layers):
        ks, vs = [], []
        for p in legacy_pasts:
            k, v = p[layer]
            ks.append(k)
            vs.append(v)
        fused_k = torch.cat(ks, dim=2)
        fused_v = torch.cat(vs, dim=2)
        fused.append((fused_k, fused_v))
    fused_legacy = tuple(fused)
    return _from_legacy_cache(fused_legacy) if any_cache_obj else fused_legacy


# ----------------------------
# Prefill builders (Route A / Route B)
# ----------------------------

@torch.no_grad()
def prefill_route_a(model, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    return out.past_key_values, out.logits[:, -1, :]

@torch.no_grad()
def prefill_route_b_from_ids(
    model,
    doc_ids_list: List[torch.Tensor],
    query_ids: torch.Tensor,
    device_map_auto: bool,
):
    device = next(model.parameters()).device

    def maybe_to_device(x: torch.Tensor) -> torch.Tensor:
        return x if device_map_auto else x.to(device)

    doc_pasts = []
    for doc_ids in doc_ids_list:
        doc_ids = maybe_to_device(doc_ids)
        out = model(
            input_ids=doc_ids,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        doc_pasts.append(out.past_key_values)

    if len(doc_pasts) == 0:
        fused = None
        fused_len = 0
    else:
        fused = fuse_past_key_values(doc_pasts)
        fused_len = past_seq_len(fused)

    query_ids = maybe_to_device(query_ids)
    q_len = int(query_ids.shape[1])

    if fused is None:
        out = model(
            input_ids=query_ids,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        return out.past_key_values, out.logits[:, -1, :]

    q_dev = query_ids.device
    position_ids = torch.arange(fused_len, fused_len + q_len, device=q_dev).unsqueeze(0)
    attention_mask = torch.ones((1, fused_len + q_len), device=q_dev, dtype=torch.long)

    out = model(
        input_ids=query_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=fused,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    return out.past_key_values, out.logits[:, -1, :]


# ----------------------------
# Greedy decode from past
# ----------------------------

@torch.no_grad()
def greedy_decode_from_past(
    model,
    tokenizer,
    past_key_values,
    logits_last: torch.Tensor,
    max_new_tokens: int = 32,
    stop_on_eos: bool = True,
    stop_on_question: bool = False,
):
    """
    Greedy (argmax) decoding.
    If stop_on_question=True, stops early if decoded text contains "\\nQuestion:" or "\\nQ:".
    """
    device = logits_last.device
    eos_id = tokenizer.eos_token_id

    generated: List[int] = []
    past = past_key_values
    cur_len = past_seq_len(past)

    for _ in range(max_new_tokens):
        next_token = int(torch.argmax(logits_last, dim=-1).item())
        generated.append(next_token)

        if stop_on_eos and eos_id is not None and next_token == eos_id:
            break

        if stop_on_question:
            txt = tokenizer.decode(generated, skip_special_tokens=True)
            if "\nQuestion:" in txt or "\nQ:" in txt:
                break

        input_ids = torch.tensor([[next_token]], device=device, dtype=torch.long)
        position_ids = torch.tensor([[cur_len]], device=device, dtype=torch.long)
        attention_mask = torch.ones((1, cur_len + 1), device=device, dtype=torch.long)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        past = out.past_key_values
        logits_last = out.logits[:, -1, :]
        cur_len += 1

    text = tokenizer.decode(generated, skip_special_tokens=True)
    if stop_on_question and "\nQuestion:" in text:
        text = text.split("\nQuestion:", 1)[0].rstrip()
    return generated, text


# ----------------------------
# Plot helpers
# ----------------------------

def save_hist(values: List[float], title: str, xlabel: str, out_path: str, bins: int = 30, dpi: int = 600):
    if len(values) == 0:
        return
    plt.figure()
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def save_em_bar(em_values: List[int], title: str, out_path: str, dpi: int = 600):
    """Plot EM (0/1) as a 2-bar count chart.

    Using a standard histogram can look "continuous" (and even show negative x ticks)
    because matplotlib auto-binning + auto-limits are tuned for continuous values.
    EM is binary, so a fixed 0/1 bar chart is the clearest.
    """
    if len(em_values) == 0:
        return
    c0 = sum(1 for x in em_values if int(x) == 0)
    c1 = sum(1 for x in em_values if int(x) == 1)

    plt.figure()
    plt.bar([0, 1], [c0, c1])
    plt.xticks([0, 1], ["0", "1"])
    plt.xlim(-0.5, 1.5)
    plt.xlabel("EM (0/1)")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


# ----------------------------
# Args + dataset loader
# ----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True, help="Local path or HF repo id.")
    ap.add_argument("--dataset_name", type=str, default="hotpotqa")
    ap.add_argument("--dataset_config", type=str, default="distractor", choices=["distractor", "fullwiki"])
    ap.add_argument("--dataset_split", type=str, default="validation")
    ap.add_argument("--dataset_path", type=str, default=None, help="Optional local dataset dir (load_from_disk preferred).")
    ap.add_argument("--no_fallback_online", action="store_true",
                    help="If set, when dataset_path fails, DO NOT fallback to online load_dataset; raise error instead.")
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--doc_mode", type=str, default="all", choices=["all", "gold_only"])

    ap.add_argument("--max_L", type=int, default=2050, help="Drop samples whose prompt length exceeds this threshold.")
    ap.add_argument("--max_new_tokens", type=int, default=32)

    ap.add_argument("--stop_on_question", action="store_true",
                    help="Stop generation early if it starts a new QA block (e.g., '\\nQuestion:').")

    ap.add_argument("--enable_postprocess", action="store_true",
                    help="If set, compute EM/F1 on cleaned answers as well (recommended).")

    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", type=str, default="auto", help='e.g. "auto" or "cuda:0"')
    ap.add_argument("--output_dir", type=str, default="rag_kv_decode_eval_out")
    ap.add_argument("--jsonl_name", type=str, default="decode_eval.jsonl")

    # NEW: debug prompt dump
    ap.add_argument("--debug_print_first", type=int, default=0,
                    help="If >0, dump first N evaluated samples' prompts into a new JSONL (2 lines per sample: A/B).")
    ap.add_argument("--debug_prompt_jsonl_name", type=str, default="debug_prompts.jsonl",
                    help="Filename for prompt-dump JSONL (written under output_dir).")

    ap.add_argument("--plot_dpi", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--write_dropped", action="store_true",
                    help="If set, write dropped samples to decode_eval.jsonl with reason/lengths.")
    return ap.parse_args()

def load_hotpot_dataset(
    dataset_name: str,
    dataset_config: str,
    dataset_split: str,
    dataset_path: Optional[str] = None,
    no_fallback_online: bool = False,
):
    from datasets import load_dataset, load_from_disk

    if dataset_path is not None:
        try:
            print(f"[Dataset] Trying to load from local disk: {dataset_path}")
            ds = load_from_disk(dataset_path)
            if dataset_split not in ds:
                raise KeyError(f"Split '{dataset_split}' not found in {list(ds.keys())}")
            print("[Dataset] Loaded from local disk successfully.")
            return ds[dataset_split]
        except Exception as e:
            print(f"[Dataset] Local load failed: {e}")
            if no_fallback_online:
                raise RuntimeError(f"Local dataset load failed and --no_fallback_online is set. ({e})")
            print("[Dataset] Falling back to online load_dataset(...)")

    print("[Dataset] Loading via load_dataset (may require internet)")
    return load_dataset(dataset_name, dataset_config)[dataset_split]


# ----------------------------
# Main
# ----------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out_jsonl = os.path.join(args.output_dir, args.jsonl_name)

    # NEW: prompt dump file (optional)
    prompt_dump_path = os.path.join(args.output_dir, args.debug_prompt_jsonl_name)
    dump_prompts = (args.debug_print_first is not None and args.debug_print_first > 0)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    model_dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=model_dtype,
            device_map=args.device_map,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=model_dtype,
            device_map=args.device_map,
        )
    model.eval()

    dataset = load_hotpot_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        dataset_path=args.dataset_path,
        no_fallback_online=args.no_fallback_online,
    )

    device_map_auto = (args.device_map == "auto")

    seen = 0
    evaluated = 0
    dropped = 0

    # raw metrics
    em_a_list, f1_a_list = [], []
    em_b_list, f1_b_list = [], []

    # clean metrics
    em_a_clean_list, f1_a_clean_list = [], []
    em_b_clean_list, f1_b_clean_list = [], []

    dumped_eval = 0

    prompt_f = open(prompt_dump_path, "w", encoding="utf-8") if dump_prompts else None

    try:
        with open(out_jsonl, "w", encoding="utf-8") as f:
            pbar = tqdm(total=min(args.max_samples, len(dataset)), desc="Eval", dynamic_ncols=True)
            for sample in dataset:
                if seen >= args.max_samples:
                    break
                i = seen
                seen += 1
                pbar.update(1)

                doc_texts, query = build_docs_and_query(sample, doc_mode=args.doc_mode)
                gold = str(sample.get("answer", "")).strip()

                full_ids, doc_ids_list, query_ids, L_A, L_B, full_prompt, b_prompt = pretokenize_for_routes(
                    tokenizer, doc_texts, query, sep="\n\n"
                )
                L = max(L_A, L_B)

                if L > args.max_L:
                    dropped += 1
                    if args.write_dropped:
                        rec = {
                            "sample_index": i,
                            "dropped": True,
                            "drop_reason": f"max_prompt_len>{args.max_L}",
                            "L_A": L_A,
                            "L_B": L_B,
                            "L": L,
                            "num_docs": len(doc_texts),
                            "doc_mode": args.doc_mode,
                            "question": sample.get("question", ""),
                            "gold_answer": gold,
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    continue

                # NEW: dump prompts for first N evaluated samples (2 lines per sample: A & B)
                if prompt_f is not None and dumped_eval < args.debug_print_first:
                    rec_a = {
                        "sample_index": i,
                        "route": "A",
                        "num_docs": len(doc_texts),
                        "L": L_A,
                        "question": sample.get("question", ""),
                        "gold_answer": gold,
                        "prompt": full_prompt,
                    }
                    rec_b = {
                        "sample_index": i,
                        "route": "B",
                        "num_docs": len(doc_texts),
                        "L": L_B,
                        "question": sample.get("question", ""),
                        "gold_answer": gold,
                        "prompt": b_prompt,
                    }
                    prompt_f.write(json.dumps(rec_a, ensure_ascii=False) + "\n")
                    prompt_f.write(json.dumps(rec_b, ensure_ascii=False) + "\n")
                    dumped_eval += 1

                # Route A attention mask
                if device_map_auto:
                    full_attn = torch.ones_like(full_ids, dtype=torch.long)
                    full_ids_in = full_ids
                else:
                    device = next(model.parameters()).device
                    full_ids_in = full_ids.to(device)
                    full_attn = torch.ones_like(full_ids_in, device=device, dtype=torch.long)

                # Route A
                past_a, logits_last_a = prefill_route_a(model, full_ids_in, attention_mask=full_attn)
                gen_ids_a, gen_text_a = greedy_decode_from_past(
                    model, tokenizer, past_a, logits_last_a,
                    max_new_tokens=args.max_new_tokens,
                    stop_on_question=args.stop_on_question,
                )

                # Route B
                past_b, logits_last_b = prefill_route_b_from_ids(
                    model,
                    doc_ids_list=doc_ids_list,
                    query_ids=query_ids,
                    device_map_auto=device_map_auto,
                )
                gen_ids_b, gen_text_b = greedy_decode_from_past(
                    model, tokenizer, past_b, logits_last_b,
                    max_new_tokens=args.max_new_tokens,
                    stop_on_question=args.stop_on_question,
                )

                # raw metrics
                em_a = exact_match_score(gen_text_a, gold)
                f1_a = f1_score(gen_text_a, gold)
                em_b = exact_match_score(gen_text_b, gold)
                f1_b = f1_score(gen_text_b, gold)

                em_a_list.append(em_a); f1_a_list.append(f1_a)
                em_b_list.append(em_b); f1_b_list.append(f1_b)

                # clean metrics
                gen_text_a_clean = extract_final_answer(gen_text_a, gold)
                gen_text_b_clean = extract_final_answer(gen_text_b, gold)

                em_a_clean = exact_match_score(gen_text_a_clean, gold)
                f1_a_clean = f1_score(gen_text_a_clean, gold)
                em_b_clean = exact_match_score(gen_text_b_clean, gold)
                f1_b_clean = f1_score(gen_text_b_clean, gold)

                if args.enable_postprocess:
                    em_a_clean_list.append(em_a_clean); f1_a_clean_list.append(f1_a_clean)
                    em_b_clean_list.append(em_b_clean); f1_b_clean_list.append(f1_b_clean)

                evaluated += 1

                rec = {
                    "sample_index": i,
                    "dropped": False,
                    "L_A": L_A,
                    "L_B": L_B,
                    "L": L,
                    "num_docs": len(doc_texts),
                    "doc_mode": args.doc_mode,
                    "question": sample.get("question", ""),
                    "gold_answer": gold,

                    "gen_route_a_raw": gen_text_a,
                    "gen_route_b_raw": gen_text_b,
                    "gen_route_a_clean": gen_text_a_clean,
                    "gen_route_b_clean": gen_text_b_clean,

                    "gen_tokens_route_a": gen_ids_a,
                    "gen_tokens_route_b": gen_ids_b,

                    "em_route_a_raw": em_a,
                    "f1_route_a_raw": f1_a,
                    "em_route_b_raw": em_b,
                    "f1_route_b_raw": f1_b,

                    "em_route_a_clean": em_a_clean,
                    "f1_route_a_clean": f1_a_clean,
                    "em_route_b_clean": em_b_clean,
                    "f1_route_b_clean": f1_b_clean,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            pbar.close()

    finally:
        if prompt_f is not None:
            prompt_f.close()

    def mean(x):
        return float(np.mean(x)) if len(x) else 0.0

    em_a_mean, f1_a_mean = mean(em_a_list), mean(f1_a_list)
    em_b_mean, f1_b_mean = mean(em_b_list), mean(f1_b_list)

    print("\n=== Dataset Length Gate ===")
    print(f"Seen      = {seen}")
    print(f"Evaluated = {evaluated}")
    print(f"Dropped   = {dropped}  (max_L={args.max_L})")

    print("\n=== Final Results (RAW, on evaluated samples) ===")
    print("Route A (one-shot prefill):")
    print(f"  EM = {em_a_mean:.4f}")
    print(f"  F1 = {f1_a_mean:.4f}")
    print("Route B (chunk-wise KV reuse / naive fuse):")
    print(f"  EM = {em_b_mean:.4f}")
    print(f"  F1 = {f1_b_mean:.4f}")
    print("Δ(Route A − Route B):")
    print(f"  ΔEM = {(em_a_mean - em_b_mean):+.4f}")
    print(f"  ΔF1 = {(f1_a_mean - f1_b_mean):+.4f}")

    if args.enable_postprocess:
        em_a_c_mean, f1_a_c_mean = mean(em_a_clean_list), mean(f1_a_clean_list)
        em_b_c_mean, f1_b_c_mean = mean(em_b_clean_list), mean(f1_b_clean_list)
        print("\n=== Final Results (CLEAN, after lightweight postprocess) ===")
        print("Route A (one-shot prefill):")
        print(f"  EM = {em_a_c_mean:.4f}")
        print(f"  F1 = {f1_a_c_mean:.4f}")
        print("Route B (chunk-wise KV reuse / naive fuse):")
        print(f"  EM = {em_b_c_mean:.4f}")
        print(f"  F1 = {f1_b_c_mean:.4f}")
        print("Δ(Route A − Route B):")
        print(f"  ΔEM = {(em_a_c_mean - em_b_c_mean):+.4f}")
        print(f"  ΔF1 = {(f1_a_c_mean - f1_b_c_mean):+.4f}")

    print(f"\nJSONL saved to: {out_jsonl}")
    if dump_prompts:
        print(f"Debug prompts JSONL saved to: {prompt_dump_path}")

    # Plots (RAW)
    save_hist(f1_a_list, "Route A F1 distribution (RAW)", "F1", os.path.join(args.output_dir, "f1_route_a_hist.png"), dpi=args.plot_dpi)
    save_hist(f1_b_list, "Route B F1 distribution (RAW)", "F1", os.path.join(args.output_dir, "f1_route_b_hist.png"), dpi=args.plot_dpi)
    # EM is binary (0/1): draw as a two-bar count plot with fixed x ticks.
    save_em_bar(em_a_list, "Route A EM distribution (RAW)", os.path.join(args.output_dir, "em_route_a_hist.png"), dpi=args.plot_dpi)
    save_em_bar(em_b_list, "Route B EM distribution (RAW)", os.path.join(args.output_dir, "em_route_b_hist.png"), dpi=args.plot_dpi)

    # Optional plots (CLEAN)
    if args.enable_postprocess:
        save_hist(f1_a_clean_list, "Route A F1 distribution (CLEAN)", "F1", os.path.join(args.output_dir, "f1_route_a_clean_hist.png"), dpi=args.plot_dpi)
        save_hist(f1_b_clean_list, "Route B F1 distribution (CLEAN)", "F1", os.path.join(args.output_dir, "f1_route_b_clean_hist.png"), dpi=args.plot_dpi)
        save_em_bar(em_a_clean_list, "Route A EM distribution (CLEAN)", os.path.join(args.output_dir, "em_route_a_clean_hist.png"), dpi=args.plot_dpi)
        save_em_bar(em_b_clean_list, "Route B EM distribution (CLEAN)", os.path.join(args.output_dir, "em_route_b_clean_hist.png"), dpi=args.plot_dpi)

if __name__ == "__main__":
    main()
