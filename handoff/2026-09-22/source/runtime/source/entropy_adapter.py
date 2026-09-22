"""Entropy-conditioned alpha wrapper around the locked prior adapter."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def normalized_attention_entropy(
    student_rows: torch.Tensor,
    attention_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    if student_rows.dim() != 3:
        raise ValueError(f"student_rows must be [layers, heads, key_len], got {tuple(student_rows.shape)}")
    mask = torch.as_tensor(attention_mask, device=student_rows.device).bool().reshape(-1)
    if mask.numel() != student_rows.shape[-1]:
        raise ValueError("attention_mask length does not match student_rows key length")
    valid_count = int(mask.sum().item())
    if valid_count <= 1:
        return torch.zeros(student_rows.shape[:2], device=student_rows.device, dtype=torch.float32)
    values = student_rows.float().detach().masked_fill(~mask.view(1, 1, -1), 0.0).clamp_min(0.0)
    denom = values.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    probs = values / denom
    terms = torch.where(probs > 0, -probs * probs.log(), torch.zeros_like(probs))
    entropy = terms.sum(dim=-1) / torch.log(torch.tensor(float(valid_count), device=values.device))
    return entropy.detach().float()


class EntropyConditionedAttentionAdapter(nn.Module):
    def __init__(self, base_adapter: nn.Module):
        super().__init__()
        self.base = base_adapter
        self.entropy_gate_weight = nn.Parameter(torch.zeros_like(base_adapter.gate_logits))

    @property
    def config(self) -> Any:
        return self.base.config

    @property
    def gate_logits(self) -> torch.Tensor:
        return self.base.gate_logits

    @property
    def w(self) -> torch.Tensor:
        return self.entropy_gate_weight

    def forward(
        self,
        teacher_projected: torch.Tensor,
        student_rows: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prior, _static_alpha = self.base(teacher_projected, student_rows)
        entropy = normalized_attention_entropy(student_rows, attention_mask, self.config.eps)
        logits = self.base.gate_logits + self.entropy_gate_weight * entropy
        alpha = torch.sigmoid(logits) * self.config.max_alpha
        return prior, alpha, entropy
