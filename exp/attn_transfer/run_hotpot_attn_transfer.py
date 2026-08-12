#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Teacher-attention transfer experiment for HotpotQA.

The teacher and student use their own tokenizers. Teacher attention over teacher
tokens is projected to student token positions by character-span overlap, then
mixed into selected student attention layers.

This first version recomputes full-sequence forwards at every decode step. That
keeps the intervention easy to audit and avoids cache-position edge cases while
the experiment is still being validated.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------
# HotpotQA helpers
# ----------------------------


def _extract_gold_titles(sample: Dict[str, Any]) -> set:
    gold_titles = set()
    for item in sample.get("supporting_facts", []) or []:
        if isinstance(item, (list, tuple)) and item:
            gold_titles.add(str(item[0]))
    return gold_titles


def _extract_docs_from_context(ctx: Any) -> List[Tuple[str, List[str]]]:
    docs: List[Tuple[str, List[str]]] = []

    if isinstance(ctx, dict):
        titles = ctx.get("title", [])
        sents = ctx.get("sentences", [])
        if isinstance(titles, list) and isinstance(sents, list):
            for title, ss in zip(titles, sents):
                if isinstance(ss, list):
                    docs.append((str(title), [str(x) for x in ss]))
        return docs

    if isinstance(ctx, list):
        for item in ctx:
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                continue
            title, ss = item
            if isinstance(ss, list):
                docs.append((str(title), [str(x) for x in ss]))
        return docs

    return docs


def build_docs_and_query(sample: Dict[str, Any], doc_mode: str = "all") -> Tuple[List[str], str]:
    docs = _extract_docs_from_context(sample.get("context"))
    gold_titles = _extract_gold_titles(sample) if doc_mode == "gold_only" else set()

    doc_texts: List[str] = []
    for title, sents in docs:
        if doc_mode == "gold_only" and gold_titles and title not in gold_titles:
            continue
        text = (title.strip() + "\n" + " ".join(sents)).strip()
        if text:
            doc_texts.append(text)

    question = str(sample.get("question", "")).strip()
    query = f"Question: {question}\nAnswer:"
    return doc_texts, query


def make_prompt(doc_texts: Sequence[str], query: str, sep: str = "\n\n") -> str:
    doc_segments = [f"[Doc {i + 1}]\n{doc}" for i, doc in enumerate(doc_texts)]
    docs_block = sep.join(doc_segments)
    return (docs_block + sep + query).strip() if docs_block else query.strip()


# ----------------------------
# EM/F1 helpers
# ----------------------------


_ARTICLES = {"a", "an", "the"}
_NUM_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+|\b\d+\b")


def _remove_articles(text: str) -> str:
    return " ".join(w for w in text.split() if w not in _ARTICLES)


def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


def _remove_punc(text: str) -> str:
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def normalize_answer(s: str) -> str:
    return _white_space_fix(_remove_articles(_remove_punc(str(s).lower())))


def exact_match_score(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    counts: Dict[str, int] = {}
    for tok in pred_tokens:
        counts[tok] = counts.get(tok, 0) + 1
    num_same = 0
    for tok in gold_tokens:
        if counts.get(tok, 0) > 0:
            num_same += 1
            counts[tok] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _looks_like_yesno(gold: str) -> bool:
    return normalize_answer(gold) in {"yes", "no"}


def _looks_like_number_answer(gold: str) -> bool:
    return any(ch.isdigit() for ch in str(gold))


def extract_final_answer(gen_text: str, gold_answer: str) -> str:
    text = str(gen_text or "").strip()
    for marker in ["\nQuestion:", "\nQ:", "\n[Doc", "\nDocument "]:
        if marker in text:
            text = text.split(marker, 1)[0].rstrip()
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()
    if not text:
        return ""
    text = text.splitlines()[0].strip()
    for end in [".", "?", "!"]:
        if end in text:
            text = text.split(end, 1)[0].strip()
    if _looks_like_yesno(gold_answer):
        m = re.search(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
        if m:
            return m.group(1).lower()
    if _looks_like_number_answer(gold_answer):
        m = _NUM_RE.search(text)
        if m:
            return m.group(0)
    return text


# ----------------------------
# Dataset/model helpers
# ----------------------------


def str2bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected bool, got {val!r}")


def str2dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_hotpot_dataset(args) -> Iterable[Dict[str, Any]]:
    if args.dataset_name == "synthetic":
        return [
            {
                "context": {
                    "title": ["Mars", "Venus", "Jupiter"],
                    "sentences": [
                        [
                            "Mars is the fourth planet from the Sun.",
                            "It is often called the Red Planet because of iron oxide on its surface.",
                        ],
                        [
                            "Venus is the second planet from the Sun.",
                            "It has a thick carbon dioxide atmosphere.",
                        ],
                        [
                            "Jupiter is the largest planet in the Solar System.",
                            "It is a gas giant with a prominent Great Red Spot.",
                        ],
                    ],
                },
                "question": "Which planet is often called the Red Planet?",
                "answer": "Mars",
                "supporting_facts": [["Mars", 1]],
            }
        ]

    if args.streaming:
        if args.dataset_path:
            raise ValueError("--streaming true expects an online dataset, not --dataset_path.")
        parquet_url = args.hf_parquet_url
        name = args.dataset_name
        if name in {"hotpotqa", "hotpot_qa", "hotpotqa/hotpot_qa"}:
            config = args.dataset_config
            split = args.dataset_split
            if split == "train":
                parquet_url = parquet_url or (
                    f"hf://datasets/hotpotqa/hotpot_qa/{config}/train-*.parquet"
                )
            else:
                parquet_url = parquet_url or (
                    f"hf://datasets/hotpotqa/hotpot_qa/{config}/{split}-00000-of-00001.parquet"
                )
        if parquet_url:
            return load_dataset("parquet", data_files=parquet_url, split="train", streaming=True)
        return load_dataset(name, args.dataset_config, split=args.dataset_split, streaming=True)

    if args.dataset_path:
        path = Path(args.dataset_path)
        if path.exists():
            obj = load_from_disk(str(path))
            if isinstance(obj, DatasetDict):
                split = args.dataset_split
                if split not in obj:
                    split = next(iter(obj.keys()))
                return obj[split]
            if isinstance(obj, Dataset):
                return obj
            raise TypeError(f"Unsupported dataset object from {path}: {type(obj)}")

    dataset_name = args.dataset_name
    if dataset_name in {"hotpotqa", "hotpot_qa"}:
        dataset_name = "hotpotqa/hotpot_qa"
    return load_dataset(dataset_name, args.dataset_config, split=args.dataset_split)


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


def load_model_and_tokenizer(path: str, dtype: torch.dtype, device: str):
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=dtype,
            local_files_only=True,
            attn_implementation="eager",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=dtype,
            local_files_only=True,
        )
    set_eager_attention(model)
    model.to(device)
    model.eval()
    return tokenizer, model


# ----------------------------
# Attention projection
# ----------------------------


def encode_with_offsets(tokenizer, text: str, device: Optional[str] = None):
    enc = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )
    offsets = [(int(a), int(b)) for a, b in enc.pop("offset_mapping")[0].tolist()]
    if device is not None:
        enc = {k: v.to(device) for k, v in enc.items()}
    return enc, offsets


def project_teacher_prior_to_student(
    teacher_prior: torch.Tensor,
    teacher_offsets: Sequence[Tuple[int, int]],
    student_offsets: Sequence[Tuple[int, int]],
    eps: float = 1e-12,
) -> torch.Tensor:
    """Project teacher token mass to student token positions by char overlap."""
    teacher_prior = teacher_prior.detach().float().cpu()
    student_mass = torch.zeros(len(student_offsets), dtype=torch.float32)

    j = 0
    for i, (t0, t1) in enumerate(teacher_offsets):
        if i >= teacher_prior.numel():
            break
        if t1 <= t0:
            continue
        while j < len(student_offsets) and student_offsets[j][1] <= t0:
            j += 1
        k = j
        denom = max(1, t1 - t0)
        while k < len(student_offsets):
            s0, s1 = student_offsets[k]
            if s0 >= t1:
                break
            overlap = max(0, min(t1, s1) - max(t0, s0))
            if overlap > 0:
                student_mass[k] += float(teacher_prior[i]) * (overlap / denom)
            k += 1

    total = float(student_mass.sum())
    if not math.isfinite(total) or total <= eps:
        valid = torch.tensor([1.0 if b > a else 0.0 for a, b in student_offsets], dtype=torch.float32)
        if float(valid.sum()) <= 0:
            valid = torch.ones(len(student_offsets), dtype=torch.float32)
        student_mass = valid / valid.sum()
    else:
        student_mass = student_mass / total
    return student_mass


def _normalize_prob_tensor(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x.detach().float().cpu()
    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    x = x.clamp_min(0.0)
    total = x.sum()
    if float(total) <= eps:
        return torch.full_like(x, 1.0 / max(1, x.numel()))
    return x / total


def pearson_corr_tensor(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> Optional[float]:
    n = min(a.numel(), b.numel())
    if n <= 1:
        return None
    a = a[:n].detach().float().cpu()
    b = b[:n].detach().float().cpu()
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt((a * a).sum() * (b * b).sum())
    if float(denom) <= eps:
        return None
    value = float((a * b).sum() / denom)
    return value if math.isfinite(value) else None


def js_divergence_tensor(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> Optional[float]:
    n = min(a.numel(), b.numel())
    if n <= 1:
        return None
    p = _normalize_prob_tensor(a[:n], eps).clamp_min(eps)
    q = _normalize_prob_tensor(b[:n], eps).clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum()
    kl_qm = (q * (q.log() - m.log())).sum()
    value = float(0.5 * (kl_pm + kl_qm))
    return value if math.isfinite(value) else None


def select_teacher_attention_row(
    attn: torch.Tensor,
    teacher_head_mode: str = "mean_all",
    teacher_top_k: int = 0,
    eps: float = 1e-12,
    text: Optional[str] = None,
    token_offsets: Optional[Sequence[Tuple[int, int]]] = None,
) -> torch.Tensor:
    """Reduce teacher heads to one attention row for transfer."""
    rows = attn[0, :, -1, :].float()  # [H, K]
    num_heads = int(rows.shape[0])
    if teacher_head_mode == "mean_all":
        return rows.mean(dim=0)

    k = int(teacher_top_k)
    if k <= 0:
        raise ValueError(f"teacher_top_k must be positive for {teacher_head_mode}: {teacher_top_k}")
    k = min(k, num_heads)

    probs = rows / rows.sum(dim=-1, keepdim=True).clamp_min(eps)
    if teacher_head_mode == "topk_entropy_low":
        entropy = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=-1)
        head_indices = torch.topk(-entropy, k=k).indices
    elif teacher_head_mode == "topk_maxprob":
        max_prob = probs.max(dim=-1).values
        head_indices = torch.topk(max_prob, k=k).indices
    elif teacher_head_mode == "topk_doc_focus":
        if text is None or token_offsets is None:
            raise ValueError("topk_doc_focus requires text and token_offsets")
        entropy = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=-1)
        focus = 1.0 - entropy / math.log(max(2, int(probs.shape[-1])))
        doc_mask, question_mask = prompt_region_masks(text, token_offsets, rows.device)
        region_mass = (probs * doc_mask.view(1, -1)).sum(dim=-1)
        region_mass = region_mass + 0.5 * (probs * question_mask.view(1, -1)).sum(dim=-1)
        scores = focus.clamp_min(0.0) * region_mass
        head_indices = torch.topk(scores, k=k).indices
    else:
        raise ValueError(f"Unsupported teacher_head_mode: {teacher_head_mode}")

    return rows.index_select(0, head_indices).mean(dim=0)


def _find_prompt_regions(text: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    q_start = text.rfind("\nQuestion:")
    if q_start < 0:
        q_start = text.find("Question:")
    if q_start < 0:
        return (0, len(text)), (0, 0)

    answer_start = text.find("\nAnswer:", q_start)
    if answer_start < 0:
        answer_start = text.find("Answer:", q_start)
    if answer_start < 0:
        answer_start = len(text)

    return (0, max(0, q_start)), (q_start, max(q_start, answer_start))


def prompt_region_masks(
    text: str,
    token_offsets: Sequence[Tuple[int, int]],
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    doc_span, question_span = _find_prompt_regions(text)

    def make_mask(span: Tuple[int, int]) -> torch.Tensor:
        start, end = span
        vals = []
        for a, b in token_offsets:
            vals.append(1.0 if b > start and a < end and b > a else 0.0)
        mask = torch.tensor(vals, device=device, dtype=torch.float32)
        if float(mask.sum()) <= 0:
            mask = torch.zeros(len(token_offsets), device=device, dtype=torch.float32)
        return mask

    return make_mask(doc_span), make_mask(question_span)


def project_teacher_heads_to_student(
    teacher_rows: torch.Tensor,
    teacher_offsets: Sequence[Tuple[int, int]],
    student_offsets: Sequence[Tuple[int, int]],
    eps: float = 1e-12,
) -> torch.Tensor:
    """Project every teacher head row to student token coordinates."""
    projected = [
        project_teacher_prior_to_student(row, teacher_offsets, student_offsets, eps)
        for row in teacher_rows.detach().float().cpu()
    ]
    return torch.stack(projected, dim=0)


def _normalize_prob_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x.detach().float().cpu()
    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    x = x.clamp_min(0.0)
    total = x.sum(dim=-1, keepdim=True).clamp_min(eps)
    return x / total


def match_teacher_heads_by_jsd(
    student_rows: torch.Tensor,
    teacher_projected_rows: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return one projected teacher prior for each student head."""
    n = min(student_rows.shape[-1], teacher_projected_rows.shape[-1])
    if n <= 1:
        return teacher_projected_rows.mean(dim=0, keepdim=True).repeat(student_rows.shape[0], 1)

    s = _normalize_prob_rows(student_rows[:, :n], eps)
    t = _normalize_prob_rows(teacher_projected_rows[:, :n], eps)
    s3 = s[:, None, :].clamp_min(eps)
    t3 = t[None, :, :].clamp_min(eps)
    m = 0.5 * (s3 + t3)
    jsd = 0.5 * (
        (s3 * (s3.log() - m.log())).sum(dim=-1)
        + (t3 * (t3.log() - m.log())).sum(dim=-1)
    )
    matched = torch.argmin(jsd, dim=1)
    return teacher_projected_rows.index_select(0, matched)


def attention_row_metrics(student_outputs, layer_to_prior: Dict[int, torch.Tensor]) -> Dict[str, Any]:
    if student_outputs.attentions is None:
        return {}
    metrics: Dict[str, Any] = {}
    for layer_idx, prior in layer_to_prior.items():
        attn = student_outputs.attentions[layer_idx]  # [B, H, Q, K]
        row = attn[0, :, -1, :].float().mean(dim=0).detach().cpu()
        p = prior.detach().float().cpu()
        if p.dim() == 2:
            p = p.mean(dim=0)
        metrics[str(layer_idx)] = {
            "jsd": js_divergence_tensor(row, p),
            "pearson": pearson_corr_tensor(row, p),
        }
    return metrics


@torch.no_grad()
def collect_teacher_priors(
    teacher_model,
    teacher_tokenizer,
    student_tokenizer,
    text: str,
    layer_indices: Sequence[int],
    device: str,
    teacher_head_mode: str = "mean_all",
    teacher_top_k: int = 0,
    eps: float = 1e-12,
    student_model=None,
) -> Tuple[Dict[int, torch.Tensor], int, int]:
    t_enc, t_offsets = encode_with_offsets(teacher_tokenizer, text, device=device)
    _, s_offsets = encode_with_offsets(student_tokenizer, text, device=None)

    out = teacher_model(
        **t_enc,
        use_cache=False,
        output_attentions=True,
        output_hidden_states=False,
    )
    if out.attentions is None:
        raise RuntimeError("Teacher did not return attentions. Check eager attention mode.")

    n_layers = len(out.attentions)
    teacher_prompt_len = int(t_enc["input_ids"].shape[1])
    if teacher_head_mode == "head_match_jsd":
        if student_model is None:
            raise ValueError("head_match_jsd requires student_model for student-head attention rows.")

        teacher_projected: Dict[int, torch.Tensor] = {}
        for layer_idx in layer_indices:
            teacher_real_idx = layer_idx if layer_idx >= 0 else n_layers + layer_idx
            student_real_idx = layer_idx
            if student_real_idx < 0:
                student_real_idx = n_layers + student_real_idx
            attn = out.attentions[teacher_real_idx]  # [B, Ht, Q, K]
            teacher_rows = attn[0, :, -1, :].float()
            teacher_projected[student_real_idx] = project_teacher_heads_to_student(
                teacher_rows, t_offsets, s_offsets, eps
            )
        del out

        s_enc = student_tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False,
        ).to(device)
        s_out = student_model(
            **s_enc,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=False,
        )
        if s_out.attentions is None:
            raise RuntimeError("Student did not return attentions for head matching.")

        priors: Dict[int, torch.Tensor] = {}
        student_n_layers = len(s_out.attentions)
        for layer_idx in layer_indices:
            student_real_idx = layer_idx if layer_idx >= 0 else student_n_layers + layer_idx
            student_rows = s_out.attentions[student_real_idx][0, :, -1, :].float()
            priors[student_real_idx] = match_teacher_heads_by_jsd(
                student_rows,
                teacher_projected[student_real_idx],
                eps,
            )
        return priors, teacher_prompt_len, len(s_offsets)

    priors: Dict[int, torch.Tensor] = {}
    for layer_idx in layer_indices:
        real_idx = layer_idx if layer_idx >= 0 else n_layers + layer_idx
        attn = out.attentions[real_idx]  # [B, H, Q, K]
        row = select_teacher_attention_row(attn, teacher_head_mode, teacher_top_k, eps, text, t_offsets)
        projected = project_teacher_prior_to_student(row, t_offsets, s_offsets, eps)
        priors[real_idx] = projected
    return priors, teacher_prompt_len, len(s_offsets)


# ----------------------------
# Qwen2 attention steering patch
# ----------------------------


@dataclass
class SteeringState:
    layer_to_prior: Dict[int, torch.Tensor]
    lambda_value: float
    eps: float
    mode: str
    enabled: bool = True


def _get_steering_state(module) -> Optional[SteeringState]:
    model_ref = getattr(module, "_attn_transfer_model_ref", None)
    if model_ref is None:
        return None
    return getattr(model_ref, "_attn_transfer_state", None)


def _get_layer_prior(module, key_len: int, device, eps: float) -> Optional[torch.Tensor]:
    state = _get_steering_state(module)
    if state is None or not state.enabled:
        return None

    layer_idx = getattr(module, "layer_idx", None)
    if layer_idx is None:
        layer_idx = getattr(module, "_attn_transfer_layer_idx", None)
    if layer_idx not in state.layer_to_prior:
        return None

    prior = state.layer_to_prior[layer_idx].to(device=device, dtype=torch.float32)
    if prior.dim() == 1:
        if prior.numel() < key_len:
            pad = torch.full((key_len - prior.numel(),), state.eps, device=prior.device, dtype=prior.dtype)
            prior = torch.cat([prior, pad], dim=0)
        prior = prior[:key_len]
        return prior / prior.sum().clamp_min(eps)

    if prior.dim() == 2:
        if prior.shape[-1] < key_len:
            pad = torch.full(
                (prior.shape[0], key_len - prior.shape[-1]),
                state.eps,
                device=prior.device,
                dtype=prior.dtype,
            )
            prior = torch.cat([prior, pad], dim=-1)
        prior = prior[:, :key_len]
        return prior / prior.sum(dim=-1, keepdim=True).clamp_min(eps)

    raise ValueError(f"Unsupported prior shape for attention steering: {tuple(prior.shape)}")


def _apply_steering_log_prior(module, attn_weights: torch.Tensor) -> torch.Tensor:
    state = _get_steering_state(module)
    if state is None or state.mode != "log_prior":
        return attn_weights
    prior = _get_layer_prior(module, attn_weights.shape[-1], attn_weights.device, state.eps)
    if prior is None:
        return attn_weights
    bias = state.lambda_value * torch.log(prior.clamp_min(state.eps))
    if bias.dim() == 1:
        bias = bias.view(1, 1, -1)
    else:
        if bias.shape[0] != attn_weights.shape[1]:
            raise ValueError(
                f"Per-head prior has {bias.shape[0]} heads, but attention has {attn_weights.shape[1]} heads."
            )
        bias = bias.view(1, bias.shape[0], -1)
    attn_weights[:, :, -1, :] = attn_weights[:, :, -1, :] + bias.to(attn_weights.dtype)
    return attn_weights


def _apply_steering_prob_mix(module, attn_probs: torch.Tensor) -> torch.Tensor:
    state = _get_steering_state(module)
    if state is None or state.mode != "prob_mix":
        return attn_probs
    prior = _get_layer_prior(module, attn_probs.shape[-1], attn_probs.device, state.eps)
    if prior is None:
        return attn_probs
    alpha = state.lambda_value / (1.0 + state.lambda_value)
    row = attn_probs[:, :, -1, :].float()
    if prior.dim() == 1:
        prior = prior.view(1, 1, -1)
    else:
        if prior.shape[0] != attn_probs.shape[1]:
            raise ValueError(
                f"Per-head prior has {prior.shape[0]} heads, but attention has {attn_probs.shape[1]} heads."
            )
        prior = prior.view(1, prior.shape[0], -1)
    mixed = (1.0 - alpha) * row + alpha * prior
    mixed = mixed / mixed.sum(dim=-1, keepdim=True).clamp_min(state.eps)
    attn_probs[:, :, -1, :] = mixed.to(attn_probs.dtype)
    return attn_probs


@contextlib.contextmanager
def qwen2_attention_steering(
    model,
    layer_to_prior: Dict[int, torch.Tensor],
    lambda_value: float,
    eps: float,
    mode: str,
):
    """Patch Qwen2Attention.forward to add a prior to the last query row."""
    try:
        import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb, repeat_kv
    except Exception as exc:
        raise RuntimeError("This experiment currently supports Qwen2 models only.") from exc

    original_forward = Qwen2Attention.forward

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if kwargs.get("padding_mask") is not None:
            attention_mask = kwargs["padding_mask"]

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError("Qwen2Attention layer_idx is required when using cache.")
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be {(bsz, self.num_heads, q_len, kv_seq_len)}, "
                f"but got {tuple(attn_weights.size())}"
            )

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = _apply_steering_log_prior(self, attn_weights)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = _apply_steering_prob_mix(self, attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be {(bsz, self.num_heads, q_len, self.head_dim)}, "
                f"but got {tuple(attn_output.size())}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    for idx, layer in enumerate(model.model.layers):
        layer.self_attn._attn_transfer_layer_idx = idx
        layer.self_attn._attn_transfer_model_ref = model

    model._attn_transfer_state = SteeringState(layer_to_prior, lambda_value, eps, mode)
    Qwen2Attention.forward = patched_forward
    try:
        yield
    finally:
        Qwen2Attention.forward = original_forward
        model._attn_transfer_state = None


# ----------------------------
# Decode/eval
# ----------------------------


def selected_layer_indices(num_layers: int, scope: str) -> List[int]:
    if scope == "last_layer":
        return [num_layers - 1]
    if scope == "last_4_layers":
        return list(range(max(0, num_layers - 4), num_layers))
    if scope.startswith("last_") and scope.endswith("_layers"):
        n_text = scope[len("last_") : -len("_layers")]
        try:
            n = int(n_text)
        except ValueError as exc:
            raise ValueError(f"Invalid layer_scope: {scope}") from exc
        if n <= 0:
            raise ValueError(f"layer_scope must select at least one layer: {scope}")
        return list(range(max(0, num_layers - n), num_layers))
    if scope == "all_layers":
        return list(range(num_layers))
    if scope.startswith("layers:"):
        parts = [p.strip() for p in scope.split(":", 1)[1].split(",") if p.strip()]
        out = []
        for p in parts:
            idx = int(p)
            out.append(idx if idx >= 0 else num_layers + idx)
        return sorted(set(out))
    raise ValueError(f"Unsupported layer_scope: {scope}")


@torch.no_grad()
def greedy_decode_plain(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str,
    stop_on_eos: bool = True,
    stop_on_question: bool = True,
) -> str:
    generated_text = ""
    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        current_text = prompt + generated_text
        enc = tokenizer(current_text, return_tensors="pt", add_special_tokens=False, truncation=False).to(device)
        out = model(**enc, use_cache=False, output_attentions=False)
        next_id = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
        if stop_on_eos and eos_id is not None and next_id == eos_id:
            break
        generated_text += tokenizer.decode([next_id], skip_special_tokens=True)
        if stop_on_question and ("\nQuestion:" in generated_text or "\nQ:" in generated_text):
            generated_text = generated_text.split("\nQuestion:", 1)[0].split("\nQ:", 1)[0].rstrip()
            break

    return generated_text


@torch.no_grad()
def greedy_decode_steered(
    teacher_model,
    teacher_tokenizer,
    student_model,
    student_tokenizer,
    prompt: str,
    layer_indices: Sequence[int],
    lambda_value: float,
    max_new_tokens: int,
    device: str,
    eps: float,
    steering_mode: str,
    record_attention_metrics: bool,
    teacher_head_mode: str = "mean_all",
    teacher_top_k: int = 0,
    stop_on_eos: bool = True,
    stop_on_question: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    generated_text = ""
    eos_id = student_tokenizer.eos_token_id
    step_debug: List[Dict[str, Any]] = []

    for step in range(max_new_tokens):
        current_text = prompt + generated_text
        layer_to_prior, teacher_len, student_len = collect_teacher_priors(
            teacher_model,
            teacher_tokenizer,
            student_tokenizer,
            current_text,
            layer_indices,
            device,
            teacher_head_mode,
            teacher_top_k,
            eps,
            student_model=student_model,
        )
        enc = student_tokenizer(
            current_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False,
        ).to(device)

        before_metrics: Dict[str, Any] = {}
        if record_attention_metrics:
            before_out = student_model(**enc, use_cache=False, output_attentions=True)
            before_metrics = attention_row_metrics(before_out, layer_to_prior)
            del before_out

        with qwen2_attention_steering(student_model, layer_to_prior, lambda_value, eps, steering_mode):
            out = student_model(
                **enc,
                use_cache=False,
                output_attentions=record_attention_metrics,
            )
        next_id = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
        after_metrics = attention_row_metrics(out, layer_to_prior) if record_attention_metrics else {}

        step_debug.append(
            {
                "step": step,
                "teacher_len": teacher_len,
                "student_len": student_len,
                "next_id": next_id,
                "teacher_head_mode": teacher_head_mode,
                "teacher_top_k": teacher_top_k,
                "attn_before": before_metrics,
                "attn_after": after_metrics,
            }
        )

        if stop_on_eos and eos_id is not None and next_id == eos_id:
            break

        generated_text += student_tokenizer.decode([next_id], skip_special_tokens=True)
        if stop_on_question and ("\nQuestion:" in generated_text or "\nQ:" in generated_text):
            generated_text = generated_text.split("\nQuestion:", 1)[0].split("\nQ:", 1)[0].rstrip()
            break

    return generated_text, step_debug


def score_prediction(raw_text: str, gold: str, enable_postprocess: bool) -> Dict[str, Any]:
    clean = extract_final_answer(raw_text, gold) if enable_postprocess else raw_text
    return {
        "raw": raw_text,
        "clean": clean,
        "em": exact_match_score(raw_text, gold),
        "f1": f1_score(raw_text, gold),
        "clean_em": exact_match_score(clean, gold),
        "clean_f1": f1_score(clean, gold),
    }


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.replace(" ", ",").split(",") if x.strip()]


def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.replace(" ", ",").split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser("HotpotQA teacher-attention transfer")
    p.add_argument("--teacher_model_path", default="/mnt/8T/cl/models/Qwen2-7B-Instruct")
    p.add_argument("--student_model_path", default="/mnt/8T/cl/models/Qwen2-1.5B-Instruct")
    p.add_argument("--dataset_name", default="hotpot_qa")
    p.add_argument("--dataset_config", default="distractor")
    p.add_argument("--dataset_path", default="")
    p.add_argument("--dataset_split", default="validation")
    p.add_argument("--streaming", type=str2bool, default=False)
    p.add_argument("--hf_parquet_url", default="")
    p.add_argument("--doc_mode", default="all", choices=["all", "gold_only"])
    p.add_argument("--max_samples", type=int, default=1)
    p.add_argument("--max_L", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lambdas", default="1.0")
    p.add_argument("--layer_scopes", default="last_layer")
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--steering_mode", default="prob_mix", choices=["prob_mix", "log_prior"])
    p.add_argument(
        "--teacher_head_mode",
        default="mean_all",
        choices=["mean_all", "topk_entropy_low", "topk_maxprob", "topk_doc_focus", "head_match_jsd"],
    )
    p.add_argument("--teacher_top_k", type=int, default=0)
    p.add_argument("--output_dir", default="exp/attn_transfer/out")
    p.add_argument("--output_name", default="hotpot_attn_transfer.jsonl")
    p.add_argument("--enable_postprocess", type=str2bool, default=True)
    p.add_argument("--stop_on_question", type=str2bool, default=True)
    p.add_argument("--record_attention_metrics", type=str2bool, default=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    dtype = str2dtype(args.dtype)
    device = args.device
    lambdas = parse_float_list(args.lambdas)
    layer_scopes = parse_str_list(args.layer_scopes)
    out_path = os.path.join(args.output_dir, args.output_name)

    print(f">>> loading dataset: path={args.dataset_path!r}, name={args.dataset_name}, split={args.dataset_split}")
    ds = load_hotpot_dataset(args)

    print(f">>> loading teacher: {args.teacher_model_path}")
    teacher_tokenizer, teacher_model = load_model_and_tokenizer(args.teacher_model_path, dtype, device)
    print(f">>> loading student: {args.student_model_path}")
    student_tokenizer, student_model = load_model_and_tokenizer(args.student_model_path, dtype, device)

    n_layers = int(getattr(student_model.config, "num_hidden_layers"))
    configs = []
    for scope in layer_scopes:
        layer_indices = selected_layer_indices(n_layers, scope)
        for lam in lambdas:
            configs.append((scope, layer_indices, lam))

    kept = 0
    seen = 0
    skipped_too_long = 0
    skipped_bad = 0

    for sample in tqdm(ds, desc="samples"):
        seen += 1
        try:
            doc_texts, query = build_docs_and_query(sample, args.doc_mode)
            prompt = make_prompt(doc_texts, query)
            gold = str(sample.get("answer", "")).strip()
            if not prompt or not gold:
                skipped_bad += 1
                continue

            s_len = len(student_tokenizer(prompt, add_special_tokens=False)["input_ids"])
            t_len = len(teacher_tokenizer(prompt, add_special_tokens=False)["input_ids"])
            if max(s_len, t_len) > args.max_L:
                skipped_too_long += 1
                continue

            baseline_raw = greedy_decode_plain(
                student_model,
                student_tokenizer,
                prompt,
                args.max_new_tokens,
                device,
                stop_on_question=args.stop_on_question,
            )
            baseline_score = score_prediction(baseline_raw, gold, args.enable_postprocess)

            record: Dict[str, Any] = {
                "sample_index": seen - 1,
                "question": sample.get("question", ""),
                "gold": gold,
                "num_docs": len(doc_texts),
                "prompt_chars": len(prompt),
                "teacher_prompt_tokens": t_len,
                "student_prompt_tokens": s_len,
                "baseline": baseline_score,
                "steering": [],
            }

            for scope, layer_indices, lam in configs:
                steered_raw, debug = greedy_decode_steered(
                    teacher_model,
                    teacher_tokenizer,
                    student_model,
                    student_tokenizer,
                    prompt,
                    layer_indices,
                    lam,
                    args.max_new_tokens,
                    device,
                    args.eps,
                    args.steering_mode,
                    args.record_attention_metrics,
                    args.teacher_head_mode,
                    args.teacher_top_k,
                    stop_on_question=args.stop_on_question,
                )
                score = score_prediction(steered_raw, gold, args.enable_postprocess)
                record["steering"].append(
                    {
                        "layer_scope": scope,
                        "layer_indices": layer_indices,
                        "lambda": lam,
                        "steering_mode": args.steering_mode,
                        "teacher_head_mode": args.teacher_head_mode,
                        "teacher_top_k": args.teacher_top_k,
                        "score": score,
                        "decode_debug": debug,
                    }
                )

            append_jsonl(out_path, record)
            kept += 1
            print(
                f">>> kept={kept} baseline clean_f1={baseline_score['clean_f1']:.3f} "
                f"raw={baseline_raw[:80]!r}"
            )
            if kept >= args.max_samples:
                break
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
            raise

    summary = {
        "seen": seen,
        "kept": kept,
        "skipped_too_long": skipped_too_long,
        "skipped_bad": skipped_bad,
        "output_path": out_path,
        "configs": [
            {
                "layer_scope": scope,
                "layer_indices": layer_indices,
                "lambda": lam,
                "steering_mode": args.steering_mode,
                "teacher_head_mode": args.teacher_head_mode,
                "teacher_top_k": args.teacher_top_k,
            }
            for scope, layer_indices, lam in configs
        ],
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
