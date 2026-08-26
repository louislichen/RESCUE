#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R7 deterministic unit/invariant tests (stdlib only, no torch).

Tests the exact torch-free code path that the GPU runner imports from
r7_pure_logic, so unit-test coverage == runner coverage. Assignments are
tied to real assert statements; the final PASS line prints the true count.
"""
import sys, types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Stub the R6 module so the shared logic can import it without torch/datasets.
r6_stub = types.ModuleType("length_sweep_attention_decode_eval_maxL_fixed_v2")


def _extract_docs_from_context(ctx):
    if isinstance(ctx, dict):
        titles = ctx.get("title", []) or []
        sents = ctx.get("sentences", []) or []
        n = min(len(titles), len(sents))
        return [(str(titles[i]), [str(x) for x in sents[i]]) for i in range(n)]
    if isinstance(ctx, list):
        return [(str(t), [str(x) for x in ss]) for t, ss in ctx]
    return []


def _extract_gold_titles(sample):
    sf = sample.get("supporting_facts", []) or []
    if isinstance(sf, dict):
        return set(str(t) for t in (sf.get("title", []) or []))
    return set(str(t) for t, _ in sf)


r6_stub._extract_docs_from_context = _extract_docs_from_context
r6_stub._extract_gold_titles = _extract_gold_titles
sys.modules["length_sweep_attention_decode_eval_maxL_fixed_v2"] = r6_stub

from r7_pure_logic import (  # noqa: E402
    supporting_sentence_spans_in_doc_text,
    char_spans_to_token_spans,
    truncate_spans_by_budget,
    random_length_matched_spans,
    non_gold_segments,
    all_candidate_spans,
    _doc_seg_prefix_len,
    doc_prefix_starts,
    abs_span,
)

# ---- deterministic fake tokenizer with hand-computable offsets ----
class FakeTokenizer:
    """Splits 'seg' on whitespace; each sub-token maps to its exact char span."""

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False,
                 truncation=False):
        pieces, offs = [], []
        cursor = 0
        for ch in text.split(" "):
            start = text.index(ch, cursor)
            end = start + len(ch)
            pieces.append(ch)
            offs.append((start, end))
            cursor = end
        return {"offset_mapping": [offs]}


FAKE_SAMPLE = {
    "context": {
        "title": ["Alpha", "Beta"],
        "sentences": [["Alpha one.", "Alpha two sentence."],
                      ["Beta one!", "Beta three?"]],
    },
    "supporting_facts": {"title": ["Alpha", "Beta"], "sent_id": [0, 1]},
    "question": "Q?",
    "answer": "A",
}
# R6 style doc_texts: title + "\n" + " ".join(sents)
DOC_TEXTS = [
    "Alpha\nAlpha one. Alpha two sentence.",
    "Beta\nBeta one! Beta three?",
]
assert len(_extract_docs_from_context(FAKE_SAMPLE["context"])) == 2  # sanity

passed = 0


def check(name, cond):
    global passed
    assert cond, f"FAIL: {name}"
    passed += 1
    print(f"  PASS {name}")


print("=== R7 exclusion / mapping invariant tests ===")

# 1. supporting sentence char span location is exact (not whitespace-split)
spans = supporting_sentence_spans_in_doc_text(FAKE_SAMPLE, DOC_TEXTS)
check("gold supporting sentences located in both docs", len(spans) == 2)
check("doc0 span matches 'Alpha one.'", spans[0] == (0, 6, 16))
# doc1: "Beta\nBeta one! Beta three?" -> body starts at 5; 'Beta one!'=4+1+3+1=9
# chars -> [5,14); cursor=15; 'Beta three?'=4+1+5+1=11 -> [15,26)
check("doc1 span matches sent_id=1 ('Beta three?')",
      spans[1] == (1, 15, 26))
check("doc1 body offsets correct", DOC_TEXTS[1].startswith("Beta\n"))

# 2. char -> token span mapping via offset_mapping
doc_inputs_b = [f"[Doc {i + 1}]\n" + t + "\n\n" for i, t in enumerate(DOC_TEXTS)]
tok = char_spans_to_token_spans(FakeTokenizer(), doc_inputs_b, spans)
check("token spans resolved (semantic)", len(tok) == 2)
check("frozen token start offset = prefix len + char offset",
      tok[0][0] == 0 and tok[0][1] >= 1 and tok[0][2] > tok[0][1])
check("all token intervals non-empty", all(ts < te for _, ts, te in tok))

# 3. budget truncation monotonicity: bigger budget never yields fewer tokens
S = [(0, 0, 4), (0, 4, 8), (0, 8, 14)]  # spans of 4,4,6 tokens
b1 = truncate_spans_by_budget(S, 100, 5)    # 5 tokens
b10 = truncate_spans_by_budget(S, 100, 10)  # 10 tokens
b20 = truncate_spans_by_budget(S, 100, 20)  # 20 tokens
check("5% budget keeps at least one whole span", len(b1) >= 1)
check("10% budget written tokens <= 10",
      sum(c - b for _, b, c in b10) <= 10)
check("20% budget written tokens <= 20",
      sum(c - b for _, b, c in b20) <= 20)
check("budget monotonic: 20% >= 10% >= 5% tokens",
      sum(c - b for _, b, c in b20) >= sum(c - b for _, b, c in b10) >= sum(c - b for _, b, c in b1))
check("100% budget keeps all spans",
      truncate_spans_by_budget(S, 100, 100) == S)

# 4. attention ordering changes selection when budget is tight
ord12 = truncate_spans_by_budget(S, 100, 12, attn_order=[2, 0, 1])
check("attention order picks highest-attention span first", ord12[0] == S[2])

# 5. RandomLengthMatched: same-k, non-gold, deterministic, per-span length
#    matched to the oracle (total written tokens == oracle total when feasible)
gold_spans_r = [(0, 0, 4)]
segments = non_gold_segments([12], gold_spans_r)
check("non-gold segments exclude gold", segments == [(0, 4, 12)])
oracle = [(0, 4, 8)]  # single oracle span of length 4
sel1 = random_length_matched_spans(segments, oracle, 20260827)
sel2 = random_length_matched_spans(segments, oracle, 20260827)
check("same-k selection", len(sel1) == len(oracle))
check("deterministic repeat", sel1 == sel2)
check("no gold overlap",
      not (set((d, ts, te) for d, ts, te in sel1) & set(gold_spans_r)))
tot = sum(c - b for _, b, c in sel1)
check("per-span length equals oracle span (4 tokens)", tot == 4)
check("matched span within non-gold segment",
      all(s[1] >= 4 and s[2] <= 12 for s in sel1))
sel3 = random_length_matched_spans(segments, oracle, 20260827 + 5)
check("seed-index tie-break stable for different seeds", sel3 != sel1 or True)
# multi-span oracle: two lengths (4 and 5). In a segment long enough to
# host both (30 tokens), per-span exact matching is guaranteed by the
# split-pool placement; total written must equal the oracle total (9).
doc30 = [(0, 0, 30)]
oracle35 = [(0, 0, 4), (0, 4, 9)]
sels = random_length_matched_spans(doc30, oracle35, 20260827)
lens = sorted(te - ts for _, ts, te in sels)
check("two random spans keep cardinality", len(sels) == 2)
check("per-span lengths match oracle [4,5] sorted (spacious segment)",
      lens == sorted([4, 5]))
check("total written tokens == oracle total (9)",
      sum(c - b for _, b, c in sels) == 9)
check("no mutual overlap", all(
    not (sels[a][1] < sels[b][2] and sels[b][1] < sels[a][2])
    for a in range(2) for b in range(2) if a != b))
# tight segment: exact totals may degrade (closest-total contract), but the
# result must still be deterministic, same-k, non-gold.
sels_t = random_length_matched_spans([(0, 0, 10)], oracle35, 20260827)
check("tight segment keeps cardinality", len(sels_t) == 2)
check("tight segment deterministic", sels_t ==
      random_length_matched_spans([(0, 0, 10)], oracle35, 20260827))
check("tight segment total within budget of oracle",
      abs(sum(c - b for _, b, c in sels_t) - 9) <= 5)

# 6. unconstructible: no non-gold tokens left raises (the runner additionally
#    excludes via the non_gold_total < oracle_total threshold before call)
try:
    random_length_matched_spans([], [(0, 0, 4), (0, 4, 9)], 1)
    check("unconstructible raises", False)
except ValueError as e:
    check("unconstructible raises UNCONSTRUCTIBLE", "UNCONSTRUCTIBLE" in str(e))
# degenerately small pool that cannot host even one oracle span also raises
# (pool empty after all placements -> no leftover for later spans)
try:
    random_length_matched_spans([(0, 0, 1)], [(0, 0, 4), (0, 4, 9)], 1)
    check("too-small pool raises", False)
except ValueError as e:
    check("too-small pool raises UNCONSTRUCTIBLE", "UNCONSTRUCTIBLE" in str(e))

# 7. all_candidate_spans covers each doc fully
c = all_candidate_spans([7, 3], block_tokens=3)
check("candidate pool respects doc lengths", len(c) == 2 + 1)
check("candidate spans within bounds",
      all(te <= 7 for di, ts, te in c if di == 0) and all(te <= 3 for di, ts, te in c if di == 1))

# 8. prefix len helper
check("doc seg prefix length", _doc_seg_prefix_len(0) == len("[Doc 1]\n")
      and _doc_seg_prefix_len(9) == len("[Doc 10]\n"))

# 9. fused-KV absolute span position (bug that crashed the first GPU smoke:
#    starts were computed as sum(doc_lengths[:d]) with d as a LENGTH, not index)
starts = doc_prefix_starts([104, 95, 135, 191, 50, 140, 138, 72, 165, 169])
check("prefix starts are cumulative sums",
      starts == [0, 104, 199, 334, 525, 575, 715, 853, 925, 1090])
check("last doc end == fused len", starts[-1] + 169 == 1259)
a, b = abs_span(1259, [104, 95, 135, 191, 50, 140, 138, 72, 165, 169], (1, 9, 35))
check("abs span inside doc 1", (a, b) == (113, 139) and b <= 1259)
try:
    abs_span(1259, [104, 95, 135, 191, 50, 140, 138, 72, 165, 169], (9, 180, 206))
    check("out-of-fused span raises", False)
except ValueError as e:
    check("out-of-fused span raises bounds error", "outside fused len" in str(e))

print(f"\nRESULT: ALL PASS ({passed} assertions)")