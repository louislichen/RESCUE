"""RESCUE 002 / P3 minimal fix: per-prefix sequence-level supervision.

Problem (measured, not assumed): in the frozen 001 implementation
``sequence_treatment_step`` installs the Adapter steering hooks only around the
prompt prefill.  Every later answer-token CE is produced by decode forwards that
run outside the hook scope, so those terms carry *exactly zero* gradient to the
Adapter (verified: ``adapter_gradient_diagnostic.json``).  The reported
"sequence loss" was therefore (1/T) * first-token CE.

Minimal fix, staying inside the frozen prefill-only / last-query-row protocol:
supervise answer position t with its own prefill of ``prompt + gold[:t]`` so that
the steered last query row is exactly the row that predicts y_t.  The Adapter
prior/alpha are still computed once from the prompt only (no gold leakage), the
intervention is still prefill-only and still last-row, and the objective is the
plain mean over answer positions.

Gradient accumulation over positions is used instead of one shared graph so that
peak memory stays at a single prefill graph.  Because gradients add linearly,
this is mathematically identical to backpropagating the mean loss.
"""
from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def load_backend_module(source_dir: str, source_lock: str):
    sys.path.insert(0, str(Path(source_dir).resolve()))
    sys.path.insert(0, str(Path(source_lock).resolve()))
    return (importlib.import_module("qwen_adapter_backend"),
            importlib.import_module("sequence_loss_runner"))


class PrefixSequenceBackend:
    """Wrap a real QwenAdapterBackend and add per-prefix sequence supervision."""

    def __init__(self, backend):
        self.backend = backend
        self.adapter = backend.adapter
        self.student = backend.student
        self.optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=0.003, weight_decay=0.0)
        self.device = backend.device
        self.eps = backend.eps
        self.layers = tuple(backend.student_layers)
        self.supervision_forward_count = 0
        self.supervision_backward_count = 0
        self.supervised_token_count = 0

    def _maps(self, target, teacher_prior, student_rows):
        prior, alpha, entropy = self.adapter(
            teacher_prior.to(self.device), student_rows,
            torch.tensor(target.attention_mask, device=self.device))
        maps = {l: prior[i] for i, l in enumerate(self.layers)}
        gates = {l: alpha[i] for i, l in enumerate(self.layers)}
        return maps, gates, prior, alpha, entropy

    def prefix_logits(self, target, prefix_len: int, maps, gates):
        ids = list(target.input_ids) + list(target.answer_ids[:prefix_len])
        input_ids = torch.tensor([ids], device=self.device, dtype=torch.long)
        mask = torch.ones((1, len(ids)), device=self.device, dtype=torch.long)
        with self.backend._steering(self.student, maps, gates, self.eps,
                                    selected_layers=set(self.layers), phase="TRAIN_PREFILL"):
            out = self.student(input_ids=input_ids, attention_mask=mask,
                               use_cache=False, output_attentions=False)
        self.supervision_forward_count += 1
        return out.logits[:, -1, :].float()

    def position_losses(self, target, maps, gates):
        return [F.cross_entropy(self.prefix_logits(target, t, maps, gates),
                                torch.tensor([int(g)], device=self.device))
                for t, g in enumerate(target.answer_ids)]

    def train_step(self, target, teacher_prior, student_rows=None):
        rows = self.backend._student_rows(target) if student_rows is None else student_rows
        self.optimizer.zero_grad(set_to_none=True)
        n = len(target.answer_ids)
        token_losses, grads_ok = [], True
        for t, gold in enumerate(target.answer_ids):
            maps, gates, prior, alpha, entropy = self._maps(target, teacher_prior, rows)
            ce = F.cross_entropy(self.prefix_logits(target, t, maps, gates),
                                 torch.tensor([int(gold)], device=self.device))
            (ce / n).backward()
            self.supervision_backward_count += 1
            self.supervised_token_count += 1
            token_losses.append(float(ce.detach()))
        grads = [p.grad for p in self.adapter.parameters()]
        if any(g is None or not torch.isfinite(g).all() for g in grads):
            grads_ok = False
        if not any(torch.any(g != 0) for g in grads):
            raise RuntimeError("all Adapter gradients are zero after per-prefix accumulation")
        self.optimizer.step()
        tl = torch.tensor(token_losses)
        return {"loss": float(tl.mean()), "first_loss": token_losses[0],
                "later_loss": (float(tl[1:].mean()) if n > 1 else None),
                "answer_token_count": n, "supervised_tokens": n, "grads_ok": grads_ok}

    def first_gold_ce(self, target, teacher_prior, student_rows=None, null=False):
        rows = self.backend._student_rows(target) if student_rows is None else student_rows
        p = teacher_prior.roll(1, dims=-1) if null else teacher_prior
        with torch.no_grad():
            maps, gates, _, _, _ = self._maps(target, p, rows)
            logits = self.prefix_logits(target, 0, maps, gates)
            return float(F.cross_entropy(logits, torch.tensor([int(target.target_id)],
                                                              device=self.device)))

    def sequence_ce(self, target, teacher_prior, student_rows=None):
        rows = self.backend._student_rows(target) if student_rows is None else student_rows
        with torch.no_grad():
            maps, gates, _, _, _ = self._maps(target, teacher_prior, rows)
            return float(torch.tensor(self.position_losses(target, maps, gates)).mean())
