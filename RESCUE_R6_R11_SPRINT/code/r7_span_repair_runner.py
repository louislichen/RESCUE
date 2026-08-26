#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R7: selective token/span repair runner (full pipeline, standalone).

Reuses the R6 runner module (r6_runner_latest.py) for prompt formatting,
dataset loading, tokenization, Route-A continuous prefill, Route-B per-doc
prefill + KV concat, greedy decode and evaluation, so the paired population,
prompt text, document order and evaluator are bit-identical to R6.

Adds ONLY:
  1. supporting-sentence -> absolute token span location inside the fused KV
     (R6 Route-B docs block), via tokenizer offset_mapping;
  2. span-level KV write-back: re-prefill `prefix + selected span` (prefix
     passed as past_key_values; only the span tokens are newly computed) and
     replace ONLY the selected KV interval, asserting total seq len unchanged;
  3. three R7 conditions:
       oracle_span            : all gold supporting sentence spans
                                (budget_pct is ignored / must be 100)
       oracle_budget          : gold spans ordered by teacher attention,
                                budgets {1,2,5,10,20}% of total prompt tokens
                                (requires --teacher_attention_jsonl; missing
                                asset -> hard failure
                                TEACHER_ATTENTION_ASSET_UNAVAILABLE)
       random_matched_budget  : non-gold random spans with same written-token
                                count and same span count as the Oracle budget
                                selection, deterministic seed 20260827+idx
  4. cost accounting per evaluated record:
       written_kv_tokens      = sum of replaced KV positions
       processed_tokens       = sum of tokens actually computed in repair
                                forwards (span tokens; prefix read via past KV)
       attended_prefix_tokens = sum of prefix KV positions read across repairs
  5. per-record provenance (condition, selected spans, route semantics).

Contract points (frozen R7 section):
  - never recompute unselected downstream docs implicitly
  - never hide processed cost behind written tokens
  - minimum unit = contiguous span (single-token write-back not fabricatable)
  - teacher-attention ordering never forged: missing asset -> explicit
    TEACHER_ATTENTION_ASSET_UNAVAILABLE failure.
"""
from __future__ import annotations
import argparse, itertools, json, os, random, sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))
_ROOT = _THIS.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# R6 module re-used as-is (must sit next to this file in the repo).
import length_sweep_attention_decode_eval_maxL_fixed_v2 as r6  # noqa: E402
# Torch-free shared logic (same code path as the unit tests).
from r7_pure_logic import (  # noqa: E402
    _doc_seg_prefix_len,
    supporting_sentence_spans_in_doc_text,
    char_spans_to_token_spans,
    truncate_spans_by_budget,
    random_length_matched_spans,
    all_candidate_spans,
    doc_prefix_starts,
    abs_span,
)

FROZEN_REPAIR_SEED = 20260827
FROZEN_SEP = "\n\n"


# ---------------------------------------------------------------------------
# Span location: supporting sentence -> absolute fused-KV token interval
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Span-level KV write-back
# ---------------------------------------------------------------------------

@torch.no_grad()
def repair_span_kv(model, fused_past, doc_inputs_ids: Sequence[torch.Tensor],
                   spans: Sequence[Tuple[int, int, int]],
                   ) -> Tuple[Any, Dict[str, int]]:
    """Replace ONLY the selected token intervals of the fused KV.

    For each span (di, ts, te): absolute interval
      [base_di + ts, base_di + te) where base_di = sum(doc_lengths[:di]).
    Forward: input_ids = span tokens; past_key_values = prefix KV (positions
    [0, abs_start)); position_ids = arange(abs_start, abs_start + span_len)
    so RoPE/positions stay absolute; attention_mask covers prefix + span.
    Then splice `new` KV over [abs_start, abs_end) and assert total length.

    Returns (repaired_fused_or_dynamic, accounting).
    """
    had_cache_obj = r6._is_cache_obj(fused_past)
    legacy = r6._to_legacy_cache(fused_past)
    device = model.get_input_embeddings().weight.device
    doc_lengths = [int(ids.shape[1]) for ids in doc_inputs_ids]
    total_len = int(legacy[0][0].shape[2])

    acct = {"written_kv_tokens": 0, "processed_tokens": 0, "attended_prefix_tokens": 0}
    repaired = legacy

    for di, ts, te in spans:
        slen = te - ts
        if slen <= 0:
            continue
        abs_start, abs_end = abs_span(total_len, doc_lengths, (di, ts, te))
        # prefix past = KV positions [0, abs_start)
        prefix = tuple((k[:, :, :abs_start, :], v[:, :, :abs_start, :]) for k, v in repaired)
        span_ids = doc_inputs_ids[di][:, ts:te].to(device)
        position_ids = torch.arange(abs_start, abs_start + slen, device=device).unsqueeze(0)
        attention_mask = torch.ones((1, abs_start + slen), device=device, dtype=torch.long)
        out = model(
            input_ids=span_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=r6._from_legacy_cache(prefix) if had_cache_obj else prefix,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        out_legacy = r6._to_legacy_cache(out.past_key_values)
        updated = []
        for (old_k, old_v), (new_k, new_v) in zip(repaired, out_legacy):
            rep_k, rep_v = new_k[:, :, -slen:, :], new_v[:, :, -slen:, :]
            merged_k = torch.cat((old_k[:, :, :abs_start, :], rep_k, old_k[:, :, abs_end:, :]), dim=2)
            merged_v = torch.cat((old_v[:, :, :abs_start, :], rep_v, old_v[:, :, abs_end:, :]), dim=2)
            if merged_k.shape[2] != total_len or merged_v.shape[2] != total_len:
                raise RuntimeError("R7 repair changed total KV sequence length")
            updated.append((merged_k, merged_v))
        repaired = tuple(updated)
        acct["written_kv_tokens"] += slen
        acct["processed_tokens"] += slen          # only span tokens computed
        acct["attended_prefix_tokens"] += abs_start  # prefix KV read during repair

    repaired = r6._from_legacy_cache(repaired) if had_cache_obj else repaired
    return repaired, acct


# ---------------------------------------------------------------------------
# Route-B prefill with R7 span repair
# ---------------------------------------------------------------------------

@torch.no_grad()
def prefill_route_b_with_span_repair(
    model, doc_ids_list: List[torch.Tensor], query_ids: torch.Tensor,
    spans: Sequence[Tuple[int, int, int]],
) -> Tuple[Any, torch.Tensor, Dict[str, int]]:
    """R6 Route-B construction, then fused-KV span repair, then query append.

    Memory discipline: doc_pasts (each doc's full KV) are freed right after
    fusing; the fused KV is the only large resident tensor during repair.
    no_grad: prefill/decode must not keep autograd graphs (R6 builds all its
    prefills under @torch.no_grad() too).
    """
    device = model.get_input_embeddings().weight.device

    doc_pasts = []
    for doc_ids in doc_ids_list:
        doc_ids = doc_ids.to(device)
        out = model(
            input_ids=doc_ids,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        doc_pasts.append(out.past_key_values)
    fused = r6.fuse_past_key_values(doc_pasts)
    del doc_pasts
    gc_collect()
    fused_len = r6.past_seq_len(fused)
    acct = {"written_kv_tokens": 0, "processed_tokens": 0, "attended_prefix_tokens": 0}
    if spans:
        fused, acct = repair_span_kv(model, fused, doc_ids_list, spans)
        fused_len = r6.past_seq_len(fused)

    query_ids = query_ids.to(device)
    q_len = int(query_ids.shape[1])
    position_ids = torch.arange(fused_len, fused_len + q_len, device=device).unsqueeze(0)
    attention_mask = torch.ones((1, fused_len + q_len), device=device, dtype=torch.long)
    out = model(
        input_ids=query_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=fused,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    return out.past_key_values, out.logits[:, -1, :], acct


def gc_collect() -> None:
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Args / main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="R7 selective token/span repair runner")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--dataset_name", type=str, default="hotpotqa")
    ap.add_argument("--dataset_config", type=str, default="distractor",
                    choices=["distractor", "fullwiki"])
    ap.add_argument("--dataset_split", type=str, default="validation")
    ap.add_argument("--dataset_path", type=str, default=None)
    ap.add_argument("--no_fallback_online", action="store_true")
    ap.add_argument("--max_samples", type=int, default=50)
    ap.add_argument("--doc_mode", type=str, default="all", choices=["all", "gold_only"])
    ap.add_argument("--max_L", type=int, default=2050)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--stop_on_question", action="store_true")
    ap.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", type=str, default="cuda:0")
    ap.add_argument("--output_dir", type=str, default="r7_span_repair_out")
    ap.add_argument("--jsonl_name", type=str, default="decode_eval.jsonl")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--write_dropped", action="store_true")

    ap.add_argument("--r7_condition", required=True,
                    choices=["oracle_span", "oracle_budget", "random_matched_budget"])
    ap.add_argument("--budget_pct", type=float, default=None,
                    help="single written-token budget percent; oracle_span forces 100.")
    ap.add_argument("--budgets", type=str, default=None,
                    help="comma-separated budget percents, e.g. '1,2,5,10,20'; "
                         "overrides --budget_pct and runs all in ONE process "
                         "(Route A/B prefills shared).")
    ap.add_argument("--teacher_attention_jsonl", type=str, default=None,
                    help="gold-span teacher-attention ordering asset (oracle_budget only).")
    ap.add_argument("--repair_seed", type=int, default=FROZEN_REPAIR_SEED)
    return ap.parse_args()


def _all_candidate_spans(doc_ids_list: Sequence[torch.Tensor],
                         block_tokens: int = 4) -> List[Tuple[int, int, int]]:
    """Non-gold candidates: contiguous token blocks (of block_tokens) inside each
    doc input. Delegates to the torch-free shared pool builder."""
    return all_candidate_spans([int(ids.shape[1]) for ids in doc_ids_list],
                               block_tokens=block_tokens)


def main():
    args = parse_args()

    if args.r7_condition == "oracle_span":
        budgets = [100.0]
        if args.budgets is not None:
            print("NOTE: oracle_span fixed at 100%; --budgets ignored.")
        elif args.budget_pct not in (None, 100.0):
            raise ValueError("oracle_span requires budget_pct=100 (all gold spans)")
    else:
        if args.budgets is not None:
            budgets = [float(x) for x in args.budgets.split(",") if x.strip()]
        elif args.budget_pct is not None:
            budgets = [args.budget_pct]
        else:
            raise ValueError(f"--budget_pct or --budgets required for {args.r7_condition}")
        if not budgets:
            raise ValueError("empty budget list")

    if args.r7_condition == "oracle_budget" and args.teacher_attention_jsonl is None:
        raise SystemExit(
            "TEACHER_ATTENTION_ASSET_UNAVAILABLE: oracle_budget requires "
            "--teacher_attention_jsonl (missing asset -> explicit failure, "
            "no forged ordering)."
        )

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

    dataset = r6.load_hotpot_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        dataset_path=args.dataset_path,
        no_fallback_online=args.no_fallback_online,
    )

    # teacher attention asset (for oracle_budget): {sample_index: [idx order]}
    teacher_order: Dict[int, List[int]] = {}
    if args.teacher_attention_jsonl:
        for line in open(args.teacher_attention_jsonl, encoding="utf-8"):
            o = json.loads(line)
            teacher_order[int(o["sample_index"])] = [int(x) for x in o["span_order"]]

    seen = evaluated = dropped = excluded = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for sample in dataset:
            if seen >= args.max_samples:
                break
            i = seen
            seen += 1

            doc_texts, query = r6.build_docs_and_query(sample, doc_mode=args.doc_mode)
            gold = str(sample.get("answer", "")).strip()

            full_ids, doc_ids_list, query_ids, L_A, L_B, full_prompt, b_prompt = \
                r6.pretokenize_for_routes(tokenizer, doc_texts, query, sep=FROZEN_SEP)
            L = max(L_A, L_B)

            if L > args.max_L:
                dropped += 1
                if args.write_dropped:
                    f.write(json.dumps({
                        "sample_index": i, "dropped": True,
                        "drop_reason": f"max_prompt_len>{args.max_L}",
                        "L_A": L_A, "L_B": L_B, "L": L,
                        "num_docs": len(doc_texts),
                        "question": sample.get("question", ""),
                        "gold_answer": gold,
                    }, ensure_ascii=False) + "\n")
                continue

            # ---- span location ----
            char_spans = supporting_sentence_spans_in_doc_text(sample, doc_texts)
            doc_inputs_b = [seg + FROZEN_SEP for seg in r6.make_doc_segments(doc_texts)]
            # Sorted like the teacher-asset builder: span_order indices refer to
            # this exact sorted list (guaranteed identical ordering both sides).
            tok_spans = sorted(char_spans_to_token_spans(tokenizer, doc_inputs_b, char_spans))

            rec = {
                "sample_index": i, "dropped": False,
                "condition": args.r7_condition, "L_A": L_A, "L_B": L_B, "L": L,
                "num_docs": len(doc_texts), "doc_mode": args.doc_mode,
                "gold_sentence_spans": sorted((di, ts, te) for di, ts, te in char_spans),
                "gold_token_spans": sorted(tok_spans),
                "question": sample.get("question", ""),
                "gold_answer": gold,
            }

            # ---- Route A (full, one-shot) ----
            device = model.get_input_embeddings().weight.device
            full_ids_in = full_ids.to(device)
            full_attn = torch.ones_like(full_ids_in, device=device, dtype=torch.long)
            past_a, logits_last_a = r6.prefill_route_a(model, full_ids_in, attention_mask=full_attn)
            gen_a, txt_a = r6.greedy_decode_from_past(
                model, tokenizer, past_a, logits_last_a,
                max_new_tokens=args.max_new_tokens, stop_on_question=args.stop_on_question)
            txt_a_clean = r6.extract_final_answer(txt_a, gold)
            rec.update({
                "gen_route_a_raw": txt_a, "gen_route_a_clean": txt_a_clean,
                "em_route_a_clean": r6.exact_match_score(txt_a_clean, gold),
                "f1_route_a_clean": r6.f1_score(txt_a_clean, gold),
                "route_a_semantics": "FULL_CONTINUOUS_PREFILL",
            })
            # Route-A KV is no longer needed after decode; release it before
            # Route-B so the two big caches never coexist (each ~3-4GB on this
            # 1.5B model; both would exceed 11GB).
            del past_a, logits_last_a, full_ids_in, full_attn
            gc_collect()

            # ---- Route B (naive) ----
            past_b, logits_b = r6.prefill_route_b_from_ids(
                model, doc_ids_list=doc_ids_list, query_ids=query_ids,
                device_map_auto=(args.device_map == "auto"), repair_mode="none", repair_k=0)
            gen_b, txt_b = r6.greedy_decode_from_past(
                model, tokenizer, past_b, logits_b,
                max_new_tokens=args.max_new_tokens, stop_on_question=args.stop_on_question)
            txt_b_clean = r6.extract_final_answer(txt_b, gold)
            rec.update({
                "gen_route_b_raw": txt_b, "gen_route_b_clean": txt_b_clean,
                "em_route_b_clean": r6.exact_match_score(txt_b_clean, gold),
                "f1_route_b_clean": r6.f1_score(txt_b_clean, gold),
                "route_b_semantics": "NAIVE_DOCUMENT_KV_REUSE",
            })
            # NOTE: past_b/logits_b intentionally kept alive here — the
            # zero-budget repair branch aliases them; freed after the repair
            # loop (end of sample).

            # ---- R7 repair conditions on the shared fused KV ----
            for bp in budgets:
                if args.r7_condition == "oracle_span":
                    sel_spans = list(tok_spans)
                    attn_used = False
                elif args.r7_condition == "oracle_budget":
                    order = teacher_order.get(i)
                    if order is None:
                        raise RuntimeError(
                            f"sample {i}: missing teacher attention order in asset")
                    sel_spans = truncate_spans_by_budget(tok_spans, L_A, bp, order)
                    attn_used = True
                else:  # random_matched_budget
                    oracle_sel = truncate_spans_by_budget(tok_spans, L_A, bp)
                    gold_pos = set((di, ts, te) for di, ts, te in tok_spans)
                    non_gold = [s for s in _all_candidate_spans(doc_ids_list)
                                if s not in gold_pos]
                    if len(non_gold) < len(oracle_sel):
                        # frozen mechanical exclusion
                        excluded += 1
                        rec2 = dict(rec)
                        rec2.update({
                            "dropped": True,
                            "drop_reason": "RANDOM_LENGTH_MATCHED_UNCONSTRUCTIBLE",
                            "budget_pct": bp,
                            "excluded_count": 1,
                        })
                        f.write(json.dumps(rec2, ensure_ascii=False) + "\n")
                        continue
                    sel_spans = random_length_matched_spans(
                        non_gold, oracle_sel, args.repair_seed + i)
                    attn_used = True

                if not sel_spans:
                    # no repair possible -> route B result stands as repair result
                    past_r, logits_r, acct = past_b, logits_b, {
                        "written_kv_tokens": 0, "processed_tokens": 0,
                        "attended_prefix_tokens": 0}
                    txt_r, txt_r_clean = txt_b, txt_b_clean
                else:
                    past_r, logits_r, acct = prefill_route_b_with_span_repair(
                        model, doc_ids_list, query_ids, sel_spans)
                    gen_r, txt_r = r6.greedy_decode_from_past(
                        model, tokenizer, past_r, logits_r,
                        max_new_tokens=args.max_new_tokens,
                        stop_on_question=args.stop_on_question)
                    txt_r_clean = r6.extract_final_answer(txt_r, gold)

                r = dict(rec)
                r.update({
                    "budget_pct": bp,
                    "selected_spans": sorted(sel_spans),
                    "written_kv_tokens": acct["written_kv_tokens"],
                    "processed_tokens": acct["processed_tokens"],
                    "attended_prefix_tokens": acct["attended_prefix_tokens"],
                    "processed_token_ratio": (acct["processed_tokens"] / L_A) if L_A else 0,
                    "written_token_ratio": (acct["written_kv_tokens"] / L_A) if L_A else 0,
                    "repair_gen_raw": txt_r,
                    "repair_gen_clean": txt_r_clean,
                    "em_repair_clean": r6.exact_match_score(txt_r_clean, gold),
                    "f1_repair_clean": r6.f1_score(txt_r_clean, gold),
                    "teacher_attention_used": attn_used,
                    "repair_route_semantics": (
                        "ROUTE_B_NAIVE_FUSED_KV + SPAN_LEVEL_WRITE_BACK "
                        "(prefix via past KV, only span tokens computed)"),
                })
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                try:
                    del past_r, logits_r
                except UnboundLocalError:
                    pass
                gc_collect()
            evaluated += 1
            # Route-B KV may be aliased by the last repair branch (past_r);
            # drop it here once all budgets for this sample are done.
            del past_b, logits_b
            del doc_ids_list, query_ids, full_ids
            gc_collect()

    print(f"Seen={seen} Evaluated={evaluated} Dropped={dropped} Excluded={excluded}")
    print(f"JSONL saved to: {out_jsonl}")


if __name__ == "__main__":
    main()