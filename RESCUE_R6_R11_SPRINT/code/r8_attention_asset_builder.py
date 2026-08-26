#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R8 attention asset builder: per-sample attention over ALL candidate spans.

For each sample in the frozen R6 window, runs ONE forward pass of a Qwen2
model (teacher 7B or student 1.5B) over the EXACT Route-A full prompt, takes
the last token position as the query position, averages attention over the
last `last_layers` layers across all heads, and emits per-candidate-span
summed attention for EVERY candidate span (all context sentences, not only
gold), so R8 selectors (StudentAttentionTopK / TeacherAttentionTopK /
TeacherStudentGapTopK) can rank any repair location.

Candidate spans = each doc's supporting-sentence token spans (same extraction
as R7 gold_token_spans) PLUS, for non-gold docs' sentences, the same
sentence-level token spans derived from the context sentences.  Coordinates
are doc-local in the fused docs-block layout (matching R7 repair coords).

Output JSONL (one line per sample):
  {"sample_index": i,
   "cand_spans": [[di, ts, te], ...],      # doc-local token coords
   "attn_sum":   [float, ...],             # same order as cand_spans
   "gold_mask":  [0/1, ...],               # 1 if that candidate is a gold support span
   "span_text":  [str, ...]}               # token-joined text for lexical checks
"""
from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import torch

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))
import length_sweep_attention_decode_eval_maxL_fixed_v2 as r6  # noqa: E402
import r7_pure_logic as P  # noqa: E402

FROZEN_SEP = "\n\n"
LAST_LAYERS = 4


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Qwen2 model path")
    ap.add_argument("--tag", type=str, required=True, help="teacher|student")
    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--dataset_name", type=str, default="hotpotqa")
    ap.add_argument("--dataset_config", type=str, default="distractor")
    ap.add_argument("--dataset_split", type=str, default="validation")
    ap.add_argument("--no_fallback_online", action="store_true")
    ap.add_argument("--max_samples", type=int, default=50)
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--last_layers", type=int, default=LAST_LAYERS)
    return ap.parse_args()


def set_eager_attention(model) -> None:
    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation("eager")
            return
        except Exception:
            pass
    try:
        model.config._attn_implementation = "eager"
    except Exception:
        pass


def _doc_base(doc_texts: Sequence[str], tokenizer: Any, di: int) -> int:
    doc_inputs_b = [seg + FROZEN_SEP for seg in r6.make_doc_segments(doc_texts)]
    total = 0
    for seg in doc_inputs_b[:di]:
        ids = tokenizer(seg, add_special_tokens=False, truncation=False)["input_ids"]
        total += len(ids)
    return total


def all_candidate_sentence_spans(sample: Any, doc_texts: Sequence[str],
                                 tokenizer: Any,
                                 doc_inputs_b: Sequence[str],
                                 ) -> Tuple[List[Tuple[int, int, int]],
                                            List[bool]]:
    """Every context sentence's token span (doc-local fused coords) + gold mask.

    A context sentence is any sentence in doc di (context.sentences[di][j]).
    R7 gold spans only cover supporting sentences; R8 candidates cover ALL
    sentences so non-gold locations can be ranked. Sentence char offsets are
    located with the same ' '.join cursor scan used by
    supporting_sentence_spans_in_doc_text, then each char span is mapped to
    token coords via the shared char_spans_to_token_spans (exact R7 code
    path), keeping the gold flag aligned per sentence.
    """
    from length_sweep_attention_decode_eval_maxL_fixed_v2 import (
        _extract_docs_from_context, _extract_gold_titles)
    docs = _extract_docs_from_context(sample.get("context", None))
    gold_titles = _extract_gold_titles(sample)
    sf = sample.get("supporting_facts", []) or []
    if isinstance(sf, dict):
        titles = [str(t) for t in (sf.get("title", []) or [])]
        sids = [int(s) for s in (sf.get("sent_id", []) or [])]
        pairs = set(zip(titles, sids))
    else:
        pairs = set((str(t), int(s)) for t, s in sf)

    char_spans: List[Tuple[int, int, int]] = []
    gold_flags: List[bool] = []
    for di, (title, sents) in enumerate(docs):
        title_s = title.strip()
        body_start = len(title_s) + 1
        cursor = body_start
        for sid, sent in enumerate(sents):
            s = str(sent)
            char_spans.append((di, cursor, cursor + len(s)))
            gold_flags.append((str(title) in gold_titles)
                              and (str(title), sid) in pairs)
            cursor += len(s) + 1  # single space from " ".join

    out_spans: List[Tuple[int, int, int]] = []
    out_flags: List[bool] = []
    for csp, g in zip(char_spans, gold_flags):
        mapped = P.char_spans_to_token_spans(
            tokenizer, doc_inputs_b, [csp])
        if mapped:
            out_spans.append(mapped[0])
            out_flags.append(g)
    return out_spans, out_flags


def span_attn_sums(model, tokenizer, doc_texts, query,
                   cand_spans: Sequence[Tuple[int, int, int]],
                   last_layers: int) -> List[float]:
    """One forward over the Route-A full prompt; summed attn per candidate span."""
    full_ids, _, _, _L_A, _L_B, _fp, _bp = r6.pretokenize_for_routes(
        tokenizer, doc_texts, query, sep=FROZEN_SEP)
    layers = model.model.layers
    n_layers = len(layers)
    keep_from = max(0, n_layers - last_layers) if last_layers > 0 else 0
    dev0 = model.model.embed_tokens.weight.device
    input_ids = full_ids.to(dev0)
    T = int(input_ids.shape[1])
    hidden = model.model.embed_tokens(input_ids)
    pos_ids = torch.arange(T, device=dev0).unsqueeze(0).long()
    dtype = hidden.dtype
    neg_inf = torch.finfo(dtype).min
    causal = torch.zeros(1, 1, T, T, device=dev0, dtype=dtype)
    causal = causal.masked_fill(
        torch.triu(torch.ones(1, 1, T, T, device=dev0), diagonal=1).bool(),
        neg_inf)
    position_embeddings = model.model.rotary_emb(hidden, pos_ids)
    acc = None
    count = 0
    with torch.no_grad():
        for li in range(n_layers):
            layer = layers[li]
            layer_dev = next(layer.parameters()).device
            hidden = hidden.to(layer_dev)
            causal_d = causal.to(layer_dev)
            pos_d = pos_ids.to(layer_dev)
            pe_d = (position_embeddings[0].to(layer_dev),
                    position_embeddings[1].to(layer_dev))
            residual = hidden
            ln = layer.input_layernorm(hidden)
            attn_out, aw = layer.self_attn(
                hidden_states=ln, attention_mask=causal_d, position_ids=pos_d,
                use_cache=False, position_embeddings=pe_d)
            hidden = residual + attn_out
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden
            if li >= keep_from and aw is not None:
                step = aw[0, :, -1, :].mean(dim=0).detach().float().cpu()
                acc = step if acc is None else acc + step
                count += 1
    if acc is None or count == 0:
        raise RuntimeError("attention loop captured nothing")
    probs = acc / count  # [T]
    sums: List[float] = []
    for di, ts, te in cand_spans:
        base = _doc_base(doc_texts, tokenizer, di)
        s = float(probs[base + ts: base + te].sum().item())
        sums.append(s)
    return sums


def main():
    args = parse_args()
    torch.manual_seed(1234)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    model_dtype = dtype_map[args.dtype]

    tokenizer = r6.AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = r6.AutoModelForCausalLM.from_pretrained(
            args.model, dtype=model_dtype, device_map=args.device_map)
    except TypeError:
        model = r6.AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=model_dtype, device_map=args.device_map)
    model.eval()
    set_eager_attention(model)

    dataset = r6.load_hotpot_dataset(
        dataset_name=args.dataset_name, dataset_config=args.dataset_config,
        dataset_split=args.dataset_split, dataset_path=args.dataset_path,
        no_fallback_online=args.no_fallback_online)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        seen = 0
        for sample in dataset:
            if seen >= args.max_samples:
                break
            i = seen
            seen += 1
            doc_texts, query = r6.build_docs_and_query(sample, doc_mode="all")
            doc_inputs_b = [seg + FROZEN_SEP for seg in r6.make_doc_segments(doc_texts)]
            cands, gold_mask = all_candidate_sentence_spans(
                sample, doc_texts, tokenizer, doc_inputs_b)
            if not cands:
                print(f"  sample {i}: no candidate spans, skip", flush=True)
                continue
            sums = span_attn_sums(model, tokenizer, doc_texts, query, cands,
                                  args.last_layers)
            spans_text = []
            for di, ts, te in cands:
                ids = tokenizer(doc_inputs_b[di], add_special_tokens=False,
                                truncation=False)["input_ids"]
                toks = ids[ts:te]
                try:
                    spans_text.append(tokenizer.decode(toks))
                except Exception:
                    spans_text.append("")
            f.write(json.dumps({
                "sample_index": i,
                "cand_spans": cands,
                "attn_sum": sums,
                "gold_mask": [int(g) for g in gold_mask],
                "span_text": spans_text,
                "query": query,
            }, ensure_ascii=False) + "\n")
            ng = sum(gold_mask)
            top = sorted(range(len(sums)), key=lambda j: -sums[j])[:3]
            print(f"  sample {i}: cands={len(cands)} gold={ng} top={top} "
                  f"top_gold={[j for j in top if gold_mask[j]]}", flush=True)
    print(f"R8 {args.tag} attention asset written to: {args.output}")


if __name__ == "__main__":
    main()