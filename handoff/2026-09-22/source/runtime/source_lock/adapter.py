#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Small trainable adapter for teacher-to-student attention priors."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttentionAdapterConfig:
    num_teacher_layers: int
    num_teacher_heads: int
    num_student_layers: int
    num_student_heads: int
    max_alpha: float = 0.35
    eps: float = 1e-8

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "AttentionAdapterConfig":
        return cls(**data)


class AttentionPriorAdapter(nn.Module):
    """Map projected teacher heads into per-student-head attention priors.

    Inputs:
      teacher_projected: [T, Ht, K]
      student_rows:      [S, Hs, K]

    Outputs:
      prior: [S, Hs, K]
      alpha: [S, Hs]
    """

    def __init__(self, config: AttentionAdapterConfig):
        super().__init__()
        self.config = config

        self.mix_logits = nn.Parameter(
            torch.zeros(
                config.num_student_layers,
                config.num_student_heads,
                config.num_teacher_layers,
                config.num_teacher_heads,
            )
        )
        self.residual_logits = nn.Parameter(torch.full((config.num_student_layers, config.num_student_heads), -1.5))
        self.gate_logits = nn.Parameter(torch.full((config.num_student_layers, config.num_student_heads), -2.0))
        self.log_temperature = nn.Parameter(torch.zeros(config.num_student_layers, config.num_student_heads))

    def forward(self, teacher_projected: torch.Tensor, student_rows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        if teacher_projected.dim() != 3:
            raise ValueError(f"teacher_projected must be [T,H,K], got {tuple(teacher_projected.shape)}")
        if student_rows.dim() != 3:
            raise ValueError(f"student_rows must be [S,H,K], got {tuple(student_rows.shape)}")
        if teacher_projected.shape[:2] != (cfg.num_teacher_layers, cfg.num_teacher_heads):
            raise ValueError(
                "teacher_projected shape mismatch: "
                f"expected {(cfg.num_teacher_layers, cfg.num_teacher_heads)}, got {tuple(teacher_projected.shape[:2])}"
            )
        if student_rows.shape[:2] != (cfg.num_student_layers, cfg.num_student_heads):
            raise ValueError(
                "student_rows shape mismatch: "
                f"expected {(cfg.num_student_layers, cfg.num_student_heads)}, got {tuple(student_rows.shape[:2])}"
            )

        teacher = normalize_rows(teacher_projected, cfg.eps)
        student = normalize_rows(student_rows, cfg.eps)

        flat_teacher = teacher.reshape(cfg.num_teacher_layers * cfg.num_teacher_heads, teacher.shape[-1])
        weights = F.softmax(self.mix_logits.reshape(cfg.num_student_layers, cfg.num_student_heads, -1), dim=-1)
        teacher_mix = torch.einsum("sgx,xk->sgk", weights, flat_teacher)
        teacher_mix = normalize_rows(teacher_mix, cfg.eps)

        residual = torch.sigmoid(self.residual_logits).unsqueeze(-1)
        base = (1.0 - residual) * teacher_mix + residual * student
        base = normalize_rows(base, cfg.eps)

        temperature = self.log_temperature.exp().clamp(0.25, 4.0).unsqueeze(-1)
        prior = F.softmax(torch.log(base.clamp_min(cfg.eps)) / temperature, dim=-1)
        alpha = torch.sigmoid(self.gate_logits) * cfg.max_alpha
        return prior, alpha


def normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    x = x.float().clamp_min(0.0)
    return x / x.sum(dim=-1, keepdim=True).clamp_min(eps)


def entropy_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = normalize_rows(x, eps).clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def adapter_regularization(
    prior: torch.Tensor,
    alpha: torch.Tensor,
    teacher_projected: torch.Tensor,
    beta_kl: float,
    gamma_alpha: float,
    delta_sharpness: float,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    teacher_mean = normalize_rows(teacher_projected.mean(dim=(0, 1), keepdim=False), eps)
    prior_mean = normalize_rows(prior.mean(dim=(0, 1), keepdim=False), eps)
    kl = (prior_mean * (prior_mean.clamp_min(eps).log() - teacher_mean.clamp_min(eps).log())).sum()
    alpha_mean = alpha.mean()
    max_entropy = math.log(max(2, int(prior.shape[-1])))
    entropy = entropy_rows(prior, eps).mean()
    sharpness = (max_entropy - entropy) / max_entropy
    reg = beta_kl * kl + gamma_alpha * alpha_mean + delta_sharpness * sharpness
    stats = {
        "reg_kl": float(kl.detach().cpu()),
        "reg_alpha_mean": float(alpha_mean.detach().cpu()),
        "reg_sharpness": float(sharpness.detach().cpu()),
    }
    return reg, stats


def checkpoint_payload(adapter: AttentionPriorAdapter, extra: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "config": adapter.config.to_json(),
        "state_dict": adapter.state_dict(),
        "extra": extra,
    }


def load_adapter_checkpoint(path: str, map_location: str | torch.device = "cpu") -> Tuple[AttentionPriorAdapter, Dict[str, Any]]:
    payload = torch.load(path, map_location=map_location)
    config = AttentionAdapterConfig.from_json(payload["config"])
    adapter = AttentionPriorAdapter(config)
    adapter.load_state_dict(payload["state_dict"])
    return adapter, payload.get("extra", {})
