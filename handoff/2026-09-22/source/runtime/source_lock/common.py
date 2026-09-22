#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared helpers for the adapter attention-transfer experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
ATTN_TRANSFER_DIR = ROOT / "exp" / "attn_transfer"
if str(ATTN_TRANSFER_DIR) not in sys.path:
    sys.path.insert(0, str(ATTN_TRANSFER_DIR))

from run_hotpot_attn_transfer import (
    build_docs_and_query,
    encode_with_offsets,
    load_model_and_tokenizer,
    make_prompt,
    project_teacher_heads_to_student,
    score_prediction,
    selected_layer_indices,
    set_eager_attention,
    str2bool,
    str2dtype,
)


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


def sample_to_prompt(
    sample: Dict[str, Any], doc_mode: str = "all"
) -> Tuple[str, str, str]:
    if "prompt" in sample and "gold" in sample:
        return (
            str(sample["prompt"]),
            str(sample["gold"]),
            str(sample.get("question", "")),
        )
    doc_texts, query = build_docs_and_query(sample, doc_mode)
    return (
        make_prompt(doc_texts, query),
        str(sample.get("answer", "")).strip(),
        str(sample.get("question", "")),
    )


def tensor_ids(ids: Sequence[int], device: str) -> torch.Tensor:
    return torch.tensor([list(ids)], device=device, dtype=torch.long)


@torch.no_grad()
def next_token_id(model, ids: Sequence[int], device: str) -> int:
    out = model(
        input_ids=tensor_ids(ids, device),
        use_cache=False,
        output_attentions=False,
    )
    return int(torch.argmax(out.logits[:, -1, :], dim=-1).item())


def gold_target_ids(tokenizer, prompt: str, gold: str) -> List[int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + " " + gold, add_special_tokens=False)["input_ids"]
    if len(full_ids) > len(prompt_ids) and full_ids[: len(prompt_ids)] == prompt_ids:
        return full_ids[len(prompt_ids) :]
    return tokenizer(" " + gold, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def collect_projected_teacher_heads(
    teacher_model,
    teacher_tokenizer,
    student_tokenizer,
    text: str,
    teacher_layer_indices: Sequence[int],
    device: str,
    eps: float,
) -> torch.Tensor:
    t_enc, t_offsets = encode_with_offsets(
        teacher_tokenizer, text, device=device
    )
    _, s_offsets = encode_with_offsets(student_tokenizer, text, device=None)
    from steering import qwen2_teacher_row_observer

    captured_rows: Dict[int, torch.Tensor] = {}
    with qwen2_teacher_row_observer(
        teacher_model, teacher_layer_indices, captured_rows
    ):
        teacher_model(
            **t_enc,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
        )
    expected = {
        idx
        if idx >= 0
        else int(teacher_model.config.num_hidden_layers) + idx
        for idx in teacher_layer_indices
    }
    if set(captured_rows) != expected:
        raise RuntimeError(
            f"Teacher streaming observer captured {sorted(captured_rows)}, "
            f"expected {sorted(expected)}"
        )
    projected: List[torch.Tensor] = []
    for idx in teacher_layer_indices:
        real_idx = (
            idx
            if idx >= 0
            else int(teacher_model.config.num_hidden_layers) + idx
        )
        rows = captured_rows[real_idx].float()
        projected.append(
            project_teacher_heads_to_student(rows, t_offsets, s_offsets, eps)
        )
    return torch.stack(projected, dim=0)


@torch.no_grad()
def collect_student_rows(
    student_model,
    input_ids: torch.Tensor,
    student_layer_indices: Sequence[int],
) -> torch.Tensor:
    out = student_model(
        input_ids=input_ids,
        use_cache=False,
        output_attentions=True,
        output_hidden_states=False,
    )
    if out.attentions is None:
        raise RuntimeError(
            "Student did not return attentions. Check eager attention mode."
        )
    n_layers = len(out.attentions)
    rows = []
    for idx in student_layer_indices:
        real_idx = idx if idx >= 0 else n_layers + idx
        rows.append(out.attentions[real_idx][0, :, -1, :].float().detach())
    return torch.stack(rows, dim=0)


def adapter_outputs_to_layer_maps(
    prior: torch.Tensor,
    alpha: torch.Tensor,
    student_layer_indices: Sequence[int],
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    layer_to_prior: Dict[int, torch.Tensor] = {}
    layer_to_alpha: Dict[int, torch.Tensor] = {}
    for rel_idx, layer_idx in enumerate(student_layer_indices):
        layer_to_prior[int(layer_idx)] = prior[rel_idx]
        layer_to_alpha[int(layer_idx)] = alpha[rel_idx]
    return layer_to_prior, layer_to_alpha
