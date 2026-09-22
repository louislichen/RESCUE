"""Real Qwen2 adapter backend for DEVELOPMENT_BASELINE_V1.

This module keeps the locked attention hook and adapter implementation as the
source of truth. It owns only model construction, first-gold-token semantics,
per-step trainable mixing, checkpoint identity gates, and decode boundaries.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from entropy_adapter import EntropyConditionedAttentionAdapter


class RealQwenRuntimeError(RuntimeError):
    pass


def _source_lock(explicit: str | None = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    here = Path(__file__).resolve()
    candidates.extend((here.parents[2] / "postmeeting_001" / "source_lock",
                       here.parents[3] / "postmeeting_001" / "source_lock"))
    for path in candidates:
        if (path / "adapter.py").exists() and (path / "steering.py").exists():
            return path
    raise RealQwenRuntimeError("locked source_lock with adapter.py and steering.py was not found")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _identity(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    if not root.is_dir():
        raise RealQwenRuntimeError(f"model path is not a directory: {root}")
    config = root / "config.json"
    tokenizer = root / "tokenizer.json"
    if not config.exists() or not tokenizer.exists():
        raise RealQwenRuntimeError(f"model snapshot lacks config/tokenizer: {root}")
    config_obj = json.loads(config.read_text(encoding="utf-8"))
    name = str(config_obj.get("model_type", ""))
    if name != "qwen2":
        raise RealQwenRuntimeError(f"expected Qwen2 model_type, got {name!r}")
    return {"path": str(root), "config_sha256": _sha256(config),
            "tokenizer_sha256": _sha256(tokenizer),
            "model_type": name, "num_hidden_layers": config_obj.get("num_hidden_layers"),
            "hidden_size": config_obj.get("hidden_size"),
            "num_attention_heads": config_obj.get("num_attention_heads")}


@dataclass(frozen=True)
class PromptTarget:
    prompt: str
    answer: str
    input_ids: list[int]
    attention_mask: list[int]
    target_id: int
    target_position: int
    answer_ids: list[int]


class QwenAdapterBackend:
    """Qwen2 Teacher/Student pair with Adapter-only optimization."""

    def __init__(self, teacher_path: str, student_path: str, device: str,
                 dtype: torch.dtype = torch.bfloat16, max_context: int = 2048,
                 source_lock: str | None = None, eps: float = 1e-8,
                 load_teacher: bool = False):
        if not torch.cuda.is_available() and str(device).startswith("cuda"):
            raise RealQwenRuntimeError("CUDA is required for the real Qwen backend")
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_context = int(max_context)
        self.eps = float(eps)
        self.lock = _source_lock(source_lock)
        if str(self.lock) not in sys.path:
            sys.path.insert(0, str(self.lock))
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from adapter import AttentionAdapterConfig, AttentionPriorAdapter
            from common import collect_projected_teacher_heads, selected_layer_indices, set_eager_attention
            from steering import qwen2_adapter_steering
        except Exception as exc:
            raise RealQwenRuntimeError(f"Qwen runtime imports failed: {exc}") from exc
        self._AutoModelForCausalLM = AutoModelForCausalLM
        self._AutoTokenizer = AutoTokenizer
        self._AdapterConfig = AttentionAdapterConfig
        self._Adapter = AttentionPriorAdapter
        self._collect_teacher = collect_projected_teacher_heads
        self._selected_layers = selected_layer_indices
        self._set_eager = set_eager_attention
        self._steering = qwen2_adapter_steering
        self.teacher_path = str(teacher_path)
        self.student_path = str(student_path)
        self.teacher_identity = _identity(teacher_path)
        self.student_identity = _identity(student_path)
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path, use_fast=True, local_files_only=True)
        self.student_tokenizer = AutoTokenizer.from_pretrained(student_path, use_fast=True, local_files_only=True)
        for tok in (self.teacher_tokenizer, self.student_tokenizer):
            if tok.pad_token_id is None and tok.eos_token_id is not None:
                tok.pad_token = tok.eos_token
        self.teacher = None
        if load_teacher:
            self.teacher = AutoModelForCausalLM.from_pretrained(
                teacher_path, torch_dtype=dtype, local_files_only=True, attn_implementation="eager"
            ).to(self.device)
        self.student = AutoModelForCausalLM.from_pretrained(
            student_path, torch_dtype=dtype, local_files_only=True, attn_implementation="eager"
        ).to(self.device)
        for model in (self.teacher, self.student):
            if model is None:
                continue
            self._set_eager(model)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
        teacher_config = self.teacher.config if self.teacher is not None else json.loads((Path(teacher_path) / "config.json").read_text(encoding="utf-8"))
        self.teacher_layers = tuple(self._selected_layers(int(teacher_config["num_hidden_layers"] if isinstance(teacher_config, dict) else teacher_config.num_hidden_layers), "last_4_layers"))
        self.student_layers = tuple(self._selected_layers(int(self.student.config.num_hidden_layers), "last_4_layers"))
        base_adapter = self._Adapter(self._AdapterConfig(
            len(self.teacher_layers), int(teacher_config["num_attention_heads"] if isinstance(teacher_config, dict) else teacher_config.num_attention_heads),
            len(self.student_layers), int(self.student.config.num_attention_heads),
            max_alpha=0.35, eps=self.eps,
        )).to(self.device)
        self.adapter = EntropyConditionedAttentionAdapter(base_adapter).to(self.device)
        self.adapter.train()
        self.optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=0.003, weight_decay=0.0)

    def prompt_target(self, prompt: str, answer: str) -> PromptTarget:
        prompt = str(prompt).strip()
        answer = str(answer).strip()
        if not prompt or not answer:
            raise RealQwenRuntimeError("empty prompt or answer; no default target is permitted")
        ids = self.student_tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = self.student_tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
        if not answer_ids:
            raise RealQwenRuntimeError("answer has no valid token target")
        if len(ids) > self.max_context:
            raise RealQwenRuntimeError(f"prompt exceeds {self.max_context} tokens")
        return PromptTarget(prompt, answer, list(ids), [1] * len(ids), int(answer_ids[0]), len(ids) - 1,
                            [int(x) for x in answer_ids])

    def _ids(self, target: PromptTarget) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([target.input_ids], device=self.device, dtype=torch.long),
                "attention_mask": torch.tensor([target.attention_mask], device=self.device, dtype=torch.long)}

    def frozen_teacher_prior(self, target: PromptTarget) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.teacher is None:
            raise RealQwenRuntimeError("Teacher prior requested after Teacher release")
        with torch.no_grad():
            projected = self._collect_teacher(
                self.teacher, self.teacher_tokenizer, self.student_tokenizer,
                target.prompt, self.teacher_layers, str(self.device), self.eps,
            ).float().detach().cpu()
        if not torch.isfinite(projected).all():
            raise RealQwenRuntimeError("Teacher prior contains non-finite values")
        meta = {"prompt_sha256": hashlib.sha256(target.prompt.encode()).hexdigest(),
                "input_ids_sha256": hashlib.sha256(bytes(target.input_ids)).hexdigest(),
                "shape": list(projected.shape), "dtype": str(projected.dtype),
                "content_sha256": hashlib.sha256(projected.numpy().tobytes()).hexdigest(),
                "teacher_identity": self.teacher_identity, "tokenizer_identity": self.teacher_identity.get("tokenizer_sha256"),
                "target_position": target.target_position}
        return projected, meta

    def _student_rows(self, target: PromptTarget) -> torch.Tensor:
        captured: dict[int, torch.Tensor] = {}
        with torch.no_grad():
            with self._steering(self.student, {}, {}, self.eps, capture_rows=captured,
                                selected_layers=set(self.student_layers), phase="TRAIN_PREFILL"):
                self.student(**self._ids(target), use_cache=False, output_attentions=False)
        if set(captured) != set(self.student_layers):
            raise RealQwenRuntimeError(f"Student row capture mismatch: {sorted(captured)}")
        return torch.stack([captured[i] for i in self.student_layers]).to(self.device).detach()

    def treatment_step(self, target: PromptTarget, teacher_prior: torch.Tensor,
                       student_rows: torch.Tensor | None = None) -> dict[str, Any]:
        rows = self._student_rows(target) if student_rows is None else student_rows.detach()
        prior, alpha, entropy = self.adapter(
            teacher_prior.to(self.device), rows, torch.tensor(target.attention_mask, device=self.device)
        )
        maps = {l: prior[i] for i, l in enumerate(self.student_layers)}
        gates = {l: alpha[i] for i, l in enumerate(self.student_layers)}
        # Training needs the adapter graph; validation selection calls this
        # method under no_grad to avoid retaining long-context graphs.
        grad_context = torch.enable_grad() if torch.is_grad_enabled() else torch.no_grad()
        with torch.inference_mode(False), grad_context:
            with self._steering(self.student, maps, gates, self.eps, phase="TRAIN_PREFILL"):
                output = self.student(**self._ids(target), use_cache=False, output_attentions=False)
        logits = output.logits[:, target.target_position, :].float()
        loss = F.cross_entropy(logits, torch.tensor([target.target_id], device=self.device))
        if not torch.isfinite(loss):
            raise RealQwenRuntimeError("gold CE is non-finite")
        return {"loss": loss, "prior": prior, "alpha": alpha, "output": output,
                "target": target, "rows": rows, "entropy": entropy}

    def sequence_treatment_step(self, target: PromptTarget, teacher_prior: torch.Tensor,
                                student_rows: torch.Tensor | None = None) -> dict[str, Any]:
        """Full-answer CE with one prompt-only Adapter prefill and gold-prefix decode."""
        rows = self._student_rows(target) if student_rows is None else student_rows.detach()
        prior, alpha, entropy = self.adapter(
            teacher_prior.to(self.device), rows, torch.tensor(target.attention_mask, device=self.device)
        )
        maps = {l: prior[i] for i, l in enumerate(self.student_layers)}
        gates = {l: alpha[i] for i, l in enumerate(self.student_layers)}
        grad_context = torch.enable_grad() if torch.is_grad_enabled() else torch.no_grad()
        with torch.inference_mode(False), grad_context:
            with self._steering(self.student, maps, gates, self.eps, phase="TRAIN_PREFILL"):
                prefill = self.student(**self._ids(target), use_cache=True, output_attentions=False)
            past = prefill.past_key_values
            logits = [prefill.logits[:, -1, :].float()]
            for index, previous_id in enumerate(target.answer_ids[:-1]):
                ids = torch.tensor([[int(previous_id)]], device=self.device, dtype=torch.long)
                mask = torch.ones((1, len(target.input_ids) + index + 1), device=self.device, dtype=torch.long)
                decoded = self.student(input_ids=ids, attention_mask=mask, past_key_values=past,
                                       use_cache=True, output_attentions=False)
                past = decoded.past_key_values
                logits.append(decoded.logits[:, -1, :].float())
            losses = [F.cross_entropy(output, torch.tensor([int(gold)], device=self.device))
                      for output, gold in zip(logits, target.answer_ids)]
            token_losses = torch.stack(losses)
            loss = token_losses.mean()
        if not torch.isfinite(loss):
            raise RealQwenRuntimeError("sequence gold CE is non-finite")
        if len(losses) > 1 and not torch.isfinite(token_losses[1:]).all():
            raise RealQwenRuntimeError("later-token CE is non-finite")
        return {"loss": loss, "first_loss": token_losses[0],
                "later_loss": token_losses[1:].mean() if len(losses) > 1 else None,
                "token_losses": token_losses, "answer_token_count": len(target.answer_ids),
                "prior": prior, "alpha": alpha, "output": prefill,
                "target": target, "rows": rows, "entropy": entropy,
                "prefill_adapter_invocations": len(self.student_layers),
                "cached_decode_adapter_invocations": 0}

    def step(self, target: PromptTarget, teacher_prior: torch.Tensor) -> dict[str, Any]:
        before = {n: p.detach().clone() for n, p in self.adapter.named_parameters()}
        self.optimizer.zero_grad(set_to_none=True)
        result = self.sequence_treatment_step(target, teacher_prior)
        result["loss"].backward()
        grads = [p.grad for p in self.adapter.parameters()]
        if any(g is None or not torch.isfinite(g).all() for g in grads):
            raise RealQwenRuntimeError("Adapter gradient is missing or non-finite")
        if not any(torch.any(g != 0) for g in grads):
            raise RealQwenRuntimeError("All Adapter gradients are zero")
        w_grad = self.adapter.entropy_gate_weight.grad
        if w_grad is None or not torch.isfinite(w_grad).all():
            raise RealQwenRuntimeError("Entropy gate w gradient is missing or non-finite")
        self.optimizer.step()
        changed = any(not torch.equal(before[n], p.detach()) for n, p in self.adapter.named_parameters())
        if not changed:
            raise RealQwenRuntimeError("Adapter parameters did not change after optimizer step")
        result["adapter_changed"] = True
        result["grad_norm"] = float(torch.sqrt(sum(g.float().pow(2).sum() for g in grads)).detach().cpu())
        result["w_grad_norm"] = float(torch.linalg.vector_norm(w_grad.float()).detach().cpu())
        result["w_grad_nonzero"] = bool(torch.any(w_grad != 0).item())
        return result

    def checkpoint(self, path: str | Path, step: int, prior_meta: dict[str, Any]) -> None:
        torch.save({"step": step, "adapter": self.adapter.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "teacher_identity": self.teacher_identity,
                    "student_identity": self.student_identity,
                    "prior_meta": prior_meta}, path)

    def load_checkpoint(self, path: str | Path) -> int:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload["teacher_identity"] != self.teacher_identity or payload["student_identity"] != self.student_identity:
            raise RealQwenRuntimeError("checkpoint base identity mismatch")
        self.adapter.load_state_dict(payload["adapter"])
        self.optimizer.load_state_dict(payload["optimizer"])
        return int(payload["step"])

    def cached_decode(self, target: PromptTarget, token: int) -> dict[str, Any]:
        with torch.no_grad():
            with self._steering(self.student, {}, {}, self.eps, phase="STUDENT_CACHED_DECODE"):
                prefill = self.student(**self._ids(target), use_cache=True, output_attentions=False)
                next_ids = torch.tensor([[int(token)]], device=self.device, dtype=torch.long)
                mask = torch.ones((1, len(target.input_ids) + 1), device=self.device, dtype=torch.long)
                decoded = self.student(input_ids=next_ids, attention_mask=mask,
                                       past_key_values=prefill.past_key_values,
                                       use_cache=True, output_attentions=False)
        return {"prefill_logits_finite": bool(torch.isfinite(prefill.logits).all()),
                "decode_logits_finite": bool(torch.isfinite(decoded.logits).all()),
                "adapter_invocations": 0, "teacher_calls": 0}

    def generate(self, target: PromptTarget, prior: torch.Tensor | None = None,
                 alpha: torch.Tensor | None = None, max_new_tokens: int = 16) -> str:
        layer_to_prior = {} if prior is None else {l: prior[i] for i, l in enumerate(self.student_layers)}
        layer_to_alpha = {} if alpha is None else {l: alpha[i] for i, l in enumerate(self.student_layers)}
        phase = "VALIDATION_PREFILL" if prior is not None else "STUDENT_CACHED_DECODE"
        with torch.no_grad():
            with self._steering(self.student, layer_to_prior, layer_to_alpha, self.eps,
                                selected_layers=set(self.student_layers), phase=phase):
                prefill = self.student(**self._ids(target), use_cache=True, output_attentions=False)
            past = prefill.past_key_values
            next_id = int(torch.argmax(prefill.logits[:, -1, :], dim=-1).item())
            generated = []
            for _ in range(max_new_tokens):
                if self.student_tokenizer.eos_token_id is not None and next_id == self.student_tokenizer.eos_token_id:
                    break
                generated.append(next_id)
                ids = torch.tensor([[next_id]], device=self.device, dtype=torch.long)
                mask = torch.ones((1, len(target.input_ids) + len(generated)), device=self.device, dtype=torch.long)
                out = self.student(input_ids=ids, attention_mask=mask, past_key_values=past,
                                   use_cache=True, output_attentions=False)
                past = out.past_key_values
                next_id = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
        return self.student_tokenizer.decode(generated, skip_special_tokens=True).strip()
