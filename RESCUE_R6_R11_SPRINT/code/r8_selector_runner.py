#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R8: important-token selector comparison on the R6/R7 window.

Five frozen selectors, all selecting repair spans from the SAME candidate
pool (all context sentences, gold and non-gold), with the SAME repair
operator (R7 span-level KV write-back) and the SAME single budget:

  1. RandomMatched          — non-gold, per-span length-matched (R7 logic)
  2. QueryOverlap           — fixed lexical overlap (query vs span text),
                              no training, character-offset projection
  3. StudentAttentionTopK   — student (Qwen2-1.5B) attention asset ranking
  4. TeacherAttentionTopK   — teacher (Qwen2-7B) attention asset ranking
  5. TeacherStudentGapTopK  — max(teacher_prior - student_attention, 0)

All selectors run on the SAME sample window as R7 (n=44 paired), same
Route A/B baselines (per-row fields), and the SAME budget picked from R7
(frozen: 1% — the most discriminative <=10% budget).

Teacher/student assets come from r8_attention_asset_builder.py
(cand_spans + attn_sum + gold_mask + span_text per sample).

Output JSONL: one row per (selector, sample, budget), containing the
selector ranking (top-k index order), selected spans, written/processed
tokens, per-sample gold-serving recall / teacher-student agreement, and the
paired CleanF1 for repair / route A / route B.

Cost axis identical to R7: processed_token_ratio = processed_tokens / L_A.
"""
from __future__ import annotations
import argparse, json, math, os, random, sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))
import length_sweep_attention_decode_eval_maxL_fixed_v2 as r6  # noqa: E402
from r7_pure_logic import (  # noqa: E402
    supporting_sentence_spans_in_doc_text,
    char_spans_to_token_spans,
    truncate_spans_by_budget,
    random_length_matched_spans,
    non_gold_segments,
    doc_prefix_starts,
    abs_span,
)
from r7_span_repair_runner import (  # noqa: E402
    prefill_route_b_with_span_repair,
    repair_span_kv,
    gc_collect,
)

FROZEN_SEP = "\n\n"
FROZEN_REPAIR_SEED = 20260827

SELECTORS = ["random_matched", "query_overlap", "student_attn_topk",
             "teacher_attn_topk", "teacher_student_gap_topk"]


def parse_args():
    ap = argparse.ArgumentParser(description="R8 important-token selector comparison")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--dataset_name", type=str, default="hotpotqa")
    ap.add_argument("--dataset_config", type=str, default="distractor")
    ap.add_argument("--dataset_split", type=str, default="validation")
    ap.add_argument("--no_fallback_online", action="store_true")
    ap.add_argument("--max_samples", type=int, default=50)
    ap.add_argument("--max_L", type=int, default=2050)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--dtype", type=str, default="float16")
    ap.add_argument("--device_map", type=str, default="cuda:0")
    ap.add_argument("--output_dir", type=str, default="r8_selector_out")
    ap.add_argument("--jsonl_name", type=str, default="decode_eval.jsonl")
    ap.add_argument("--budget_pct", type=float, default=1.0,
                    help="frozen single budget from R7 (default 1%)")
    ap.add_argument("--teacher_asset", type=str, required=True)
    ap.add_argument("--student_asset", type=str, required=True)
    ap.add_argument("--seed", type=int, default=1234)
    return ap.parse_args()


def load_asset(path: str) -> Dict[int, Dict[str, Any]]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["sample_index"]] = rec
    return out


def query_overlap_order(query_text: str, span_texts: Sequence[str],
                        tokenizer) -> List[float]:
    """Lexical overlap score per span: fraction of span tokens (as strings)
    that also occur in the query token string set. Character-offset based:
    tokens are decoded strings compared at the string level, so the rule
    transfers across tokenizers without an alignment network."""
    q_set = set(tokenizer.tokenize(query_text))
    scores = []
    for st in span_texts:
        stoks = tokenizer.tokenize(st)
        if not stoks:
            scores.append(0.0)
            continue
        hit = sum(1 for t in stoks if t in q_set)
        scores.append(hit / len(stoks))
    return scores


def gap_order(teacher_sum: Sequence[float], student_sum: Sequence[float]
              ) -> List[float]:
    """max(teacher_prior - student_attention, 0) per candidate."""
    return [max(float(t) - float(s), 0.0) for t, s in zip(teacher_sum, student_sum)]


def rank_spans(asset: Dict[str, Any], selector: str, tokenizer,
               teacher_sum: Sequence[float], student_sum: Sequence[float],
               ) -> Tuple[List[int], List[float]]:
    """Return (order_indices, per-candidate score) for a selector over the
    full candidate pool of one sample."""
    cand_spans = asset["cand_spans"]
    n = len(cand_spans)
    if selector == "random_matched":
        oracle_lens = [4] * min(n, 1)  # placeholder, replaced by budget below
        return _random_order(cand_spans, n), [0.0] * n
    if selector == "query_overlap":
        scores = query_overlap_order(asset.get("query", ""),
                                     asset.get("span_text", []), tokenizer)
    elif selector == "student_attn_topk":
        scores = [float(x) for x in student_sum]
    elif selector == "teacher_attn_topk":
        scores = [float(x) for x in teacher_sum]
    elif selector == "teacher_student_gap_topk":
        scores = gap_order(teacher_sum, student_sum)
    else:
        raise ValueError(f"unknown selector {selector}")
    order = sorted(range(n), key=lambda j: (-scores[j], j))
    return order, scores


def _random_order(cand_spans, n):
    rng = random.Random(20260827)
    order = list(range(n))
    rng.shuffle(order)
    return order


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out_jsonl = os.path.join(args.output_dir, args.jsonl_name)

    torch.manual_seed(args.seed)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    model_dtype = dtype_map[args.dtype]

    tokenizer = r6.AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = r6.AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=model_dtype, device_map=args.device_map)
    except TypeError:
        model = r6.AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=model_dtype, device_map=args.device_map)
    model.eval()

    teacher_asset = load_asset(args.teacher_asset)
    student_asset = load_asset(args.student_asset)

    dataset = r6.load_hotpot_dataset(
        dataset_name=args.dataset_name, dataset_config=args.dataset_config,
        dataset_split=args.dataset_split, dataset_path=args.dataset_path,
        no_fallback_online=args.no_fallback_online)

    gold_spans_ctx = {}  # sample -> gold token spans (R7 coords)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        dropped = 0
        excluded = 0
        evaluated = 0
        for i, sample in enumerate(dataset):
            if i >= args.max_samples:
                break
            doc_texts, query = r6.build_docs_and_query(sample, doc_mode="all")
            doc_inputs_b = [seg + FROZEN_SEP for seg in r6.make_doc_segments(doc_texts)]
            full_ids, doc_ids_list, query_ids, L_A, L_B, full_prompt, b_prompt = \
                r6.pretokenize_for_routes(tokenizer, doc_texts, query, sep=FROZEN_SEP)
            L = max(L_A, L_B)
            if L > args.max_L:
                dropped += 1
                print(f"[drop] {i} L={L}>{args.max_L}", flush=True)
                continue

            gold = sample.get("answer", "").strip()
            char_spans = supporting_sentence_spans_in_doc_text(sample, doc_texts)
            tok_spans = sorted(char_spans_to_token_spans(
                tokenizer, doc_inputs_b, char_spans))

            # assets must be present for this sample (frozen window: assets
            # built for the same 50 samples)
            tasset = teacher_asset.get(i)
            sasset = student_asset.get(i)
            if tasset is None or sasset is None:
                raise RuntimeError(
                    f"R8 ATTENTION_ASSET_UNAVAILABLE: sample {i} missing "
                    f"teacher/student asset (teacher={tasset is not None}, "
                    f"student={sasset is not None})")

            teacher_sum = tasset["attn_sum"]
            student_sum = sasset["attn_sum"]
            cands = [tuple(c) for c in tasset["cand_spans"]]
            if len(cands) != len(teacher_sum) or len(cands) != len(student_sum):
                raise RuntimeError(
                    f"R8 asset shape mismatch sample {i}: cands={len(cands)} "
                    f"teacher={len(teacher_sum)} student={len(student_sum)}")

            # Route A (full) — decode for paired baseline
            device = model.get_input_embeddings().weight.device
            full_ids_in = full_ids.to(device)
            full_attn = torch.ones_like(full_ids_in, device=device, dtype=torch.long)
            past_a, logits_last_a = r6.prefill_route_a(
                model, full_ids_in, attention_mask=full_attn)
            gen_a, txt_a = r6.greedy_decode_from_past(
                model, tokenizer, past_a, logits_last_a,
                max_new_tokens=args.max_new_tokens)
            txt_a_clean = r6.extract_final_answer(txt_a, gold)
            del past_a, logits_last_a, full_ids_in, full_attn
            gc_collect()

            # Route B (naive) — fused KV, decode for paired baseline
            past_b, logits_b = r6.prefill_route_b_from_ids(
                model, doc_ids_list=doc_ids_list, query_ids=query_ids,
                device_map_auto=(args.device_map == "auto"), repair_mode="none",
                repair_k=0)
            gen_b, txt_b = r6.greedy_decode_from_past(
                model, tokenizer, past_b, logits_b,
                max_new_tokens=args.max_new_tokens)
            txt_b_clean = r6.extract_final_answer(txt_b, gold)

            budget_tokens = int(args.budget_pct / 100.0 * L_A)
            non_gold_segs = non_gold_segments(
                [int(d.shape[1]) for d in doc_ids_list], tok_spans)

            for sel in SELECTORS:
                order, scores = rank_spans(
                    tasset, sel, tokenizer, teacher_sum, student_sum)
                if sel == "random_matched":
                    # same-k + length matched to oracle selection at this budget
                    oracle_sel = truncate_spans_by_budget(
                        tok_spans, L_A, args.budget_pct)
                    min_span_tokens = sum(te - ts for _, ts, te in oracle_sel)
                    non_gold_total = sum(te - ts for _, ts, te in non_gold_segs)
                    if non_gold_total < min_span_tokens:
                        excluded += 1
                        rec2 = {
                            "sample_index": i, "selector": sel,
                            "dropped": True,
                            "drop_reason": "RANDOM_LENGTH_MATCHED_UNCONSTRUCTIBLE",
                            "budget_pct": args.budget_pct, "excluded_count": 1,
                        }
                        f.write(json.dumps(rec2, ensure_ascii=False) + "\n")
                        continue
                    sel_spans = random_length_matched_spans(
                        non_gold_segs, oracle_sel, FROZEN_REPAIR_SEED + i)
                    attn_used = False
                    topk_used = None
                    table = None
                else:
                    # top-k by score, respecting the budget (span = min unit)
                    sel_spans = truncate_spans_by_budget(
                        cands, L_A, args.budget_pct, attn_order=order)
                    attn_used = True
                    topk_used = order
                    table = scores

                if not sel_spans:
                    past_r, logits_r, acct = past_b, logits_b, {
                        "written_kv_tokens": 0, "processed_tokens": 0,
                        "attended_prefix_tokens": 0}
                    txt_r, txt_r_clean = txt_b, txt_b_clean
                else:
                    past_r, logits_r, acct = prefill_route_b_with_span_repair(
                        model, doc_ids_list, query_ids, sel_spans)
                    gen_r, txt_r = r6.greedy_decode_from_past(
                        model, tokenizer, past_r, logits_r,
                        max_new_tokens=args.max_new_tokens)
                    txt_r_clean = r6.extract_final_answer(txt_r, gold)

                # diagnostics
                gold_set = set(tok_spans)
                gold_token_recall = 0.0
                hit_gold = 0
                tot_sel_tok = 0
                for d, ts, te in sel_spans:
                    ntok = te - ts
                    tot_sel_tok += ntok
                    for gd, gs, ge in tok_spans:
                        if d == gd:
                            ov = min(te, ge) - max(ts, gs)
                            if ov > 0:
                                gold_token_recall += ov
                                hit_gold += 1
                gold_token_recall = gold_token_recall / max(1, sum(te - ts for _, ts, te in tok_spans))
                gold_hit_rate = hit_gold / max(1, len(sel_spans))

                r = {
                    "sample_index": i,
                    "selector": sel,
                    "condition": f"r8_{sel}",
                    "budget_pct": args.budget_pct,
                    "dropped": False,
                    "L_A": L_A, "L_B": L_B, "L": L,
                    "num_docs": len(doc_ids_list),
                    "gold_token_spans": tok_spans,
                    "selected_spans": sorted(sel_spans),
                    "written_kv_tokens": acct["written_kv_tokens"],
                    "processed_tokens": acct["processed_tokens"],
                    "attended_prefix_tokens": acct["attended_prefix_tokens"],
                    "processed_token_ratio": (acct["processed_tokens"] / L_A) if L_A else 0,
                    "written_token_ratio": (acct["written_kv_tokens"] / L_A) if L_A else 0,
                    "gen_route_a_raw": txt_a, "gen_route_a_clean": txt_a_clean,
                    "em_route_a_clean": r6.exact_match_score(txt_a_clean, gold),
                    "f1_route_a_clean": r6.f1_score(txt_a_clean, gold),
                    "gen_route_b_raw": txt_b, "gen_route_b_clean": txt_b_clean,
                    "em_route_b_clean": r6.exact_match_score(txt_b_clean, gold),
                    "f1_route_b_clean": r6.f1_score(txt_b_clean, gold),
                    "repair_gen_raw": txt_r, "repair_gen_clean": txt_r_clean,
                    "em_repair_clean": r6.exact_match_score(txt_r_clean, gold),
                    "f1_repair_clean": r6.f1_score(txt_r_clean, gold),
                    "gold_token_recall": gold_token_recall,
                    "gold_span_hit_rate": gold_hit_rate,
                    "num_candidates": len(cands),
                    "num_gold_spans": len(tok_spans),
                    "topk_used_indices": topk_used,
                    "selector_scores_sample": table,
                    "teacher_attention_used": attn_used,
                    "repair_route_semantics": "R7 SPAN_LEVEL_WRITE_BACK (prefix via past KV)",
                    "route_a_semantics": "FULL_CONTINUOUS_PREFILL",
                    "route_b_semantics": "NAIVE_DOCUMENT_KV_REUSE",
                }
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                try:
                    del past_r, logits_r
                except UnboundLocalError:
                    pass
                gc_collect()
            evaluated += 1
            del past_b, logits_b, doc_ids_list, query_ids, full_ids
            gc_collect()
            print(f"[ok] {i} {len(SELECTORS)} selectors done",
                  flush=True)

    print(f"R8 selector comparison saved to {out_jsonl}")
    print(f"Seen={i+1 if 'i' in dir() else 0} Evaluated={evaluated} "
          f"Dropped={dropped} Excluded={excluded}")


if __name__ == "__main__":
    main()