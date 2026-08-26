#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Teacher-attention ordering asset builder for R7 OracleRankedBudget (and R8).

For each sample in the frozen R6 window, runs ONE forward pass of the teacher
model (Qwen2-7B-Instruct) over the EXACT Route-A full prompt (reusing the R6
pretokenize_for_routes text, so tokenization/order match the paired Runner-A
baseline). Then:

  - takes the prompt's LAST token position (the final 'Answer:' position) as
    the query position;
  - averages its attention distribution over the last 4 layers across all
    heads -> attn_probs over prompt tokens;
  - for every gold supporting-sentence token span (R7 gold_token_spans
    coordinates), sums attn_probs over the span tokens;
  - emits span_order = indices of gold_token_spans sorted by summed teacher
    attention, HIGHEST first.

Output JSONL (one line per sample):
  {"sample_index": i, "span_order": [idx...], "span_attn_sum": [...]}

This asset lets OracleRankedBudget order gold spans by teacher attention
without forging or inventing an ordering; the actual R7 budget truncation is
done by the runner (truncate_spans_by_budget).
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))
import length_sweep_attention_decode_eval_maxL_fixed_v2 as r6  # noqa: E402
import r7_pure_logic as P  # noqa: E402

FROZEN_SEP = "\n\n"
TEACHER_LAST_LAYERS = 4


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_model", type=str, required=True)
    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--dataset_name", type=str, default="hotpotqa")
    ap.add_argument("--dataset_config", type=str, default="distractor")
    ap.add_argument("--dataset_split", type=str, default="validation")
    ap.add_argument("--no_fallback_online", action="store_true")
    ap.add_argument("--max_samples", type=int, default=50)
    ap.add_argument("--max_L", type=int, default=2050)
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--last_layers", type=int, default=TEACHER_LAST_LAYERS)
    return ap.parse_args()


def teacher_span_attn_order(
    teacher: Any,
    tokenizer: Any,
    doc_texts: Sequence[str],
    query: str,
    gold_token_spans: Sequence[Tuple[int, int, int]],
    last_layers: int,
) -> Tuple[List[int], List[float]]:
    """One teacher forward over the Route-A full prompt; return span_order.

    Walks the decoder layers MANUALLY so only the last `last_layers` request
    attention weights (output_attentions=True). HF's model forward with
    output_attentions=True materialises ALL layers' [B, H, T, T] tensors for
    the whole forward, which OOMs 7B on 4x11GB; per-layer we take the final
    query position, average over heads, accumulate on CPU, and let each layer's
    tensor be freed immediately.
    """
    full_ids, _, _, _L_A, _L_B, _fp, _bp = r6.pretokenize_for_routes(
        tokenizer, doc_texts, query, sep=FROZEN_SEP)
    model = teacher.model
    layers = model.layers
    n_layers = len(layers)
    keep_from = max(0, n_layers - last_layers) if last_layers > 0 else 0

    dev0 = model.embed_tokens.weight.device
    input_ids = full_ids.to(dev0)
    T = int(input_ids.shape[1])

    hidden = model.embed_tokens(input_ids)
    pos_ids = torch.arange(T, device=dev0).unsqueeze(0).long()
    dtype = hidden.dtype
    neg_inf = torch.finfo(dtype).min
    causal = torch.zeros(1, 1, T, T, device=dev0, dtype=dtype)
    causal = causal.masked_fill(
        torch.triu(torch.ones(1, 1, T, T, device=dev0), diagonal=1).bool(),
        neg_inf)
    position_embeddings = model.rotary_emb(hidden, pos_ids)

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
            # Replicate Qwen2DecoderLayer.forward manually so we can read the
            # post-softmax attn_weights that the layer API discards.
            residual = hidden
            ln = layer.input_layernorm(hidden)
            attn_out, aw = layer.self_attn(
                hidden_states=ln,
                attention_mask=causal_d,
                position_ids=pos_d,
                use_cache=False,
                position_embeddings=pe_d,
            )
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
        raise RuntimeError("teacher loop captured no attention weights")
    probs = acc / count  # [T] averaged over kept layers and heads

    # sum attention over each gold span's tokens
    sums: List[float] = []
    for di, ts, te in gold_token_spans:
        # spans are in FUSED docs-block coordinates; the teacher prompt is the
        # FULL Route-A prompt whose docs-block layout matches the fused KV, so
        # project directly by token position using doc base recomputed below.
        base = _doc_base(doc_texts, tokenizer, di)
        s = float(probs[base + ts: base + te].sum().item())
        sums.append(s)
    order = sorted(range(len(gold_token_spans)), key=lambda i: (-sums[i], i))
    return order, [sums[i] for i in order]


def _doc_base(doc_texts: Sequence[str], tokenizer: Any, di: int) -> int:
    """Fused docs-block start of doc di (Route-A prompt excludes query):
    sum of tokenized lengths of doc_inputs_b[:di]."""
    doc_inputs_b = [seg + FROZEN_SEP for seg in r6.make_doc_segments(doc_texts)]
    total = 0
    for seg in doc_inputs_b[:di]:
        ids = tokenizer(seg, add_special_tokens=False, truncation=False)["input_ids"]
        total += len(ids)
    return total


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


def main():
    args = parse_args()
    torch.manual_seed(1234)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    model_dtype = dtype_map[args.dtype]

    tokenizer = r6.AutoTokenizer.from_pretrained(args.teacher_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        teacher = r6.AutoModelForCausalLM.from_pretrained(
            args.teacher_model, dtype=model_dtype, device_map=args.device_map)
    except TypeError:
        teacher = r6.AutoModelForCausalLM.from_pretrained(
            args.teacher_model, torch_dtype=model_dtype, device_map=args.device_map)
    teacher.eval()
    set_eager_attention(teacher)

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
            char_spans = P.supporting_sentence_spans_in_doc_text(sample, doc_texts)
            doc_inputs_b = [seg + FROZEN_SEP for seg in r6.make_doc_segments(doc_texts)]
            tok_spans = sorted(P.char_spans_to_token_spans(tokenizer, doc_inputs_b, char_spans))

            order, sums = teacher_span_attn_order(
                teacher, tokenizer, doc_texts, query, tok_spans, args.last_layers)
            f.write(json.dumps({
                "sample_index": i,
                "span_order": order,
                "span_attn_sum": sums,
                "num_gold_spans": len(tok_spans),
                "gold_token_spans": tok_spans,
            }, ensure_ascii=False) + "\n")
            print(f"  sample {i}: spans={len(tok_spans)} order={order[:5]}", flush=True)

    print(f"Teacher attention order asset written to: {args.output}")


if __name__ == "__main__":
    main()