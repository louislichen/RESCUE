#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R7 pure-logic helpers (no torch dependency), used by BOTH the runner and
the deterministic unit tests, so the tests exercise the exact code paths that
the GPU runner calls.

Kept torch-free by design: span location, budget truncation and
RandomTokenLengthMatched selection must be verifiable without a GPU.
"""
from __future__ import annotations
import itertools, random
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _doc_seg_prefix_len(doc_index: int) -> int:
    """Length of the '[Doc N]\\n' prefix inside a Route-A doc segment."""
    return len(f"[Doc {doc_index + 1}]\n")


def supporting_sentence_spans_in_doc_text(sample: Dict[str, Any],
                                           doc_texts: Sequence[str],
                                           ) -> List[Tuple[int, int, int]]:
    """Return [(doc_index, char_start, char_end)] of every gold supporting
    sentence, in doc_texts[i] coordinates (title + '\\n' + ' '.join(sents)).

    doc_texts uses the exact R6 build_docs_and_query format, so we recompute
    sentence char spans from the original context sentence lists with the same
    join rules (' '.join, single space), never by whitespace-splitting text.
    """
    from length_sweep_attention_decode_eval_maxL_fixed_v2 import (
        _extract_docs_from_context, _extract_gold_titles)
    docs = _extract_docs_from_context(sample.get("context", None))
    gold_titles = _extract_gold_titles(sample)

    if len(docs) != len(doc_texts):
        raise RuntimeError("R7: context/docs alignment mismatch vs R6 doc_texts")

    results: List[Tuple[int, int, int]] = []

    sf = sample.get("supporting_facts", []) or []
    if isinstance(sf, dict):
        titles = [str(t) for t in (sf.get("title", []) or [])]
        sids = [int(s) for s in (sf.get("sent_id", []) or [])]
        pairs = list(zip(titles, sids))
    else:
        pairs = [(str(t), int(s)) for t, s in sf]

    for di, (title, sents) in enumerate(docs):
        if title not in gold_titles:
            continue
        title_s = title.strip()
        body_start = len(title_s) + 1
        cursor = body_start
        for sid, sent in enumerate(sents):
            s = str(sent)
            sel = [p for p in pairs if p[0] == title and p[1] == sid]
            if sel:
                results.append((di, cursor, cursor + len(s)))
            cursor += len(s) + 1  # single space from " ".join
    return results


def doc_prefix_starts(doc_lengths: Sequence[int]) -> List[int]:
    """Absolute fused-KV start position of each doc segment (cumulative sum)."""
    starts: List[int] = []
    acc = 0
    for l in doc_lengths:
        starts.append(acc)
        acc += l
    return starts


def abs_span(fused_len: int, doc_lengths: Sequence[int], span: Tuple[int, int, int],
             ) -> Tuple[int, int]:
    """Absolute [astart, aend) of (di, ts, te) inside the fused KV, with bounds check."""
    di, ts, te = span
    starts = doc_prefix_starts(doc_lengths)
    base = starts[di]
    astart, aend = base + ts, base + te
    if aend > fused_len or astart < 0:
        raise ValueError(
            f"span [{astart},{aend}) outside fused len {fused_len} "
            f"(doc {di}, base {base}, ts/te {ts}/{te})"
        )
    return astart, aend


def char_spans_to_token_spans(tokenizer, doc_inputs_b: Sequence[str],
                              spans: Sequence[Tuple[int, int, int]],
                              ) -> List[Tuple[int, int, int]]:
    """Map (doc_index, char_start, char_end) in doc_text coords to
    (doc_index, tok_start, tok_end) inside the R6 Route-B doc inputs."""
    out: List[Tuple[int, int, int]] = []
    for di, cs, ce in spans:
        seg = doc_inputs_b[di]
        base = _doc_seg_prefix_len(di)
        gs, ge = base + cs, base + ce
        if gs < 0:
            gs = 0
        if ge > len(seg):
            ge = len(seg)
        if gs >= ge:
            continue
        enc = tokenizer(
            seg,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
        )
        raw_om = enc["offset_mapping"]
        # Fast tokenizers return list[tuple] for a single string, or
        # list[list[tuple]] when batched; normalize to a single-sequence list.
        if raw_om and isinstance(raw_om[0], (list, tuple)) and len(raw_om[0]) == 2 \
                and isinstance(raw_om[0][0], int):
            om = raw_om
        elif raw_om:
            om = raw_om[0]
        else:
            om = []
        ts: Optional[int] = None
        te: Optional[int] = None
        for tok_i, (a, b) in enumerate(om):
            if a is None or b is None:
                continue
            if ts is None and b > gs:
                ts = tok_i
            if a < ge:
                te = tok_i + 1
            else:
                break
        if ts is not None and te is not None:
            out.append((di, int(ts), int(te)))
    return out


def truncate_spans_by_budget(spans: Sequence[Tuple[int, int, int]],
                             total_prompt_tokens: int,
                             budget_pct: float,
                             attn_order: Optional[Sequence[int]] = None,
                             ) -> List[Tuple[int, int, int]]:
    """Select a prefix of spans whose total written tokens <= budget tokens.

    budget_tokens = budget_pct * total_prompt_tokens (frozen semantic).
    If attn_order is given, spans are processed in that order (highest attention
    first); otherwise original order. A single over-budget span is kept whole
    (minimum unit = contiguous span).
    """
    budget_tokens = int(budget_pct / 100.0 * total_prompt_tokens)
    order = list(range(len(spans))) if attn_order is None else list(attn_order)
    sel: List[Tuple[int, int, int]] = []
    used = 0
    for idx in order:
        sp = spans[idx]
        slen = sp[2] - sp[1]
        if slen <= 0:
            continue
        if used + slen > budget_tokens:
            if not sel and slen > budget_tokens:
                sel = [sp]
            break
        sel.append(sp)
        used += slen
    return sel


def random_length_matched_spans(non_gold_spans: Sequence[Tuple[int, int, int]],
                                oracle_spans: Sequence[Tuple[int, int, int]],
                                seed: int,
                                ) -> List[Tuple[int, int, int]]:
    """Select non-gold spans with the same span count and the closest total
    written-token count to the oracle selection, deterministic tie-break with
    random.Random(seed)."""
    k = len(oracle_spans)
    if k == 0:
        return []
    oracle_tokens = sum(s[2] - s[1] for s in oracle_spans)
    if len(non_gold_spans) < k:
        raise ValueError(
            f"R7 RANDOM_LENGTH_MATCHED_UNCONSTRUCTIBLE: eligible non-gold "
            f"spans {len(non_gold_spans)} < k={k}"
        )
    best_gap: Optional[int] = None
    best: List[List[Tuple[int, int, int]]] = []
    for combo in itertools.combinations(non_gold_spans, k):
        tot = sum(s[2] - s[1] for s in combo)
        gap = abs(tot - oracle_tokens)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best = [list(combo)]
        elif gap == best_gap:
            best.append(list(combo))
    return list(sorted(random.Random(seed).choice(best)))


def all_candidate_spans(doc_token_lengths: Sequence[int], block_tokens: int = 4,
                        ) -> List[Tuple[int, int, int]]:
    """Non-gold candidates: contiguous token blocks inside each doc input."""
    cand: List[Tuple[int, int, int]] = []
    for di, Ldi in enumerate(doc_token_lengths):
        t = 0
        while t + block_tokens <= Ldi:
            cand.append((di, t, t + block_tokens))
            t += block_tokens
    return cand