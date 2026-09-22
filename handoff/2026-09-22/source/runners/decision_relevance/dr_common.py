"""Shared helpers for RESCUE-TEACHER-PRIOR-DECISION-RELEVANCE-DIAGNOSTIC-001.

Frozen method variables (see task §2): Student, Teacher/Target, step80 checkpoint, Teacher
prior, Null prior, alpha, selected layers/heads, full-window intervention, tokenizer
projection, dtype, dataset, decoding — all unchanged from
RESCUE-MINIMAL-FOUR-ARM-VALID-RERUN-001.

This task only *measures*: at every decision position actually visited by the BASE
trajectory it evaluates the same prefix through
    A  stock Student
    C  Student + full-window Teacher prior
    D  Student + full-window Null prior
    T  Target Qwen2-7B
so that all four logit vectors come from the same sample, the same prefix and the same
token position (task §3, strict pairing).
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ARMS_PARENT = ("A_base", "B_prefill_only_teacher", "C_full_window_teacher",
               "D_full_window_null")
PREFILL_PHASE = "DR_PREFILL_STEERED"
DECODE_PHASE = "DR_CACHED_DECODE_STEERED"

#: decision-space metric fields every row must carry (§16 integrity)
ROW_REQUIRED_FIELDS = ("canonical_source_id", "condition", "position", "prefix_sha256",
                       "group", "y_base", "y_target", "logp_A_yT", "logp_C_yT", "logp_D_yT",
                       "dlogp_T_teacher", "dlogp_T_null", "M_A", "M_C", "M_D",
                       "dM_teacher", "dM_null", "cos_teacher", "cos_null",
                       "run_config_hash", "checkpoint_sha256")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_harness(harness_root):
    sys.path.insert(0, str(Path(harness_root).resolve()))
    from rescue_harness import harness as H
    return H


def load_modules(H, source_dir, source_lock):
    sys.path.insert(0, str(Path(source_dir).resolve()))
    sys.path.insert(0, str(Path(source_lock).resolve()))
    backend_mod = importlib.import_module("qwen_adapter_backend")
    runner_mod = importlib.import_module("sequence_loss_runner")
    steering = importlib.import_module("steering")
    fixed = H.load_harness_steering()
    steering.qwen2_adapter_steering = fixed
    steering._frozen_original = steering.qwen2_adapter_steering
    return backend_mod, runner_mod, steering


def build_backend(backend_mod, args, steering):
    backend = backend_mod.QwenAdapterBackend(
        args.teacher_model, args.student_model, args.device,
        source_lock=args.source_lock, load_teacher=False)
    backend._steering = steering.qwen2_adapter_steering
    payload = torch.load(args.checkpoint, map_location=backend.device, weights_only=False)
    backend.adapter.load_state_dict(payload["adapter"])
    return backend


def maps_for(backend, prior, alpha):
    lm = {} if prior is None else {l: prior[i] for i, l in enumerate(backend.student_layers)}
    la = {} if alpha is None else {l: alpha[i] for i, l in enumerate(backend.student_layers)}
    return lm, la


def ctx(steering_cm, backend, maps, alpha, layers, phase, capture_rows=None):
    """Harness steering CM (fresh instance) or a no-op when there is no prior."""
    if not maps:
        return _null()
    return steering_cm(backend.student, maps, alpha, backend.eps,
                       selected_layers=layers, phase=phase, capture_rows=capture_rows)


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def generate_greedy(backend, target, max_new_tokens: int) -> list[int]:
    """Stock greedy trajectory — identical code path to the parent run's arm A."""
    with torch.no_grad():
        o = backend.student(**backend._ids(target), use_cache=True, output_attentions=False)
        past = o.past_key_values
        nxt = int(torch.argmax(o.logits[:, -1, :], dim=-1).item())
        toks: list[int] = []
        for _ in range(max_new_tokens):
            if backend.student_tokenizer.eos_token_id is not None and \
                    nxt == backend.student_tokenizer.eos_token_id:
                break
            toks.append(nxt)
            ids = torch.tensor([[nxt]], device=backend.device, dtype=torch.long)
            mask = torch.ones((1, len(target.input_ids) + len(toks)),
                              device=backend.device, dtype=torch.long)
            o = backend.student(input_ids=ids, attention_mask=mask, past_key_values=past,
                                use_cache=True, output_attentions=False)
            past = o.past_key_values
            nxt = int(torch.argmax(o.logits[:, -1, :], dim=-1).item())
    return toks


def student_positions(backend, steering_cm, target, base_tokens, maps, alpha, window,
                      layers, vocab, capture_rows=False):
    """Feed prompt + base_tokens[:t] and return (logits list, natural top1 list, captures).

    window = "none" -> stock, "full" -> hook on prefill and on every cached-decode query.
    Returns logits at decision positions 0..T-1 where T = len(base_tokens).
    """
    hook = bool(maps) and window == "full"
    logits_out, natural, caps = [], [], []
    with torch.no_grad():
        with ctx(steering_cm, backend, maps, alpha, layers, PREFILL_PHASE,
                 capture_rows=({} if capture_rows and hook else None)) as st:
            o = backend.student(**backend._ids(target), use_cache=True,
                                output_attentions=False)
            caps.append(None if st is None or st.capture_rows is None
                        else {k: v.detach().float().cpu().clone()
                              for k, v in st.capture_rows.items()})
        past = o.past_key_values
        logits_out.append(o.logits[0, -1, :vocab].detach().float().cpu())
        natural.append(int(torch.argmax(logits_out[-1]).item()))
        for t in range(len(base_tokens) - 1):
            tok = base_tokens[t]
            total = past.get_seq_length() + 1
            ids = torch.tensor([[tok]], device=backend.device)
            mask = torch.ones((1, total), device=backend.device, dtype=torch.long)
            with ctx(steering_cm, backend, maps, alpha, layers, DECODE_PHASE,
                     capture_rows=({} if capture_rows and hook else None)) as st:
                o = backend.student(input_ids=ids, attention_mask=mask, past_key_values=past,
                                    use_cache=True, output_attentions=False)
                caps.append(None if st is None or st.capture_rows is None
                            else {k: v.detach().float().cpu().clone()
                                  for k, v in st.capture_rows.items()})
            past = o.past_key_values
            logits_out.append(o.logits[0, -1, :vocab].detach().float().cpu())
            natural.append(int(torch.argmax(logits_out[-1]).item()))
    return logits_out, natural, caps


def target_positions(model, device, vocab, prompt_ids, base_tokens):
    """Target logits at the same decision positions (incremental, one forward per position)."""
    zs = []
    with torch.no_grad():
        ids = torch.tensor([prompt_ids], device=device)
        mask = torch.ones((1, len(prompt_ids)), device=device, dtype=torch.long)
        o = model(input_ids=ids, attention_mask=mask, use_cache=True)
        cache = o.past_key_values
        zs.append(o.logits[0, -1, :vocab].detach().float().cpu())
        for t in range(len(base_tokens) - 1):
            tok = base_tokens[t]
            total = cache.get_seq_length() + 1
            o = model(input_ids=torch.tensor([[tok]], device=device),
                      attention_mask=torch.ones((1, total), device=device, dtype=torch.long),
                      past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            zs.append(o.logits[0, -1, :vocab].detach().float().cpu())
    return zs


# ------------------------------------------------------------------ metrics
def logprob_of(z: torch.Tensor, token: int) -> float:
    return float(F.log_softmax(z, dim=-1)[token])


def margin_of(z: torch.Tensor, token: int) -> float:
    """z[token] - max over j != token of z[j]  (§6)."""
    zz = z.clone()
    best = float(zz[token])
    zz[token] = -float("inf")
    return best - float(zz.max())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = float(a.norm()), float(b.norm())
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))


def l1_rows(p: torch.Tensor, q: torch.Tensor) -> float:
    """Mean over heads of the L1 distance between two attention rows [heads, keys]."""
    return float((p - q).abs().sum(dim=-1).mean())


def jsd_rows(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    """Mean over heads of the Jensen-Shannon divergence (log base 2)."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (a / b).log2()).sum(dim=-1)  # noqa: E731
    return float((0.5 * kl(p, m) + 0.5 * kl(q, m)).mean())


def bootstrap_ci(values, n=10000, seed=20260830):
    import random
    import statistics
    if not values:
        return None
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(values[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return {"n": k, "mean": statistics.fmean(values), "median": statistics.median(values),
            "ci95_low": means[int(0.025 * n)], "ci95_high": means[int(0.975 * n) - 1],
            "positive_fraction": sum(1 for v in values if v > 0) / k,
            "negative_fraction": sum(1 for v in values if v < 0) / k,
            "zero_fraction": sum(1 for v in values if v == 0) / k,
            "boot_n": n, "label": "DEVELOPMENT diagnostic (paired bootstrap), not confirmation"}


def paired_delta_ci(high, low, n=10000, seed=20260830):
    return bootstrap_ci([h - l for h, l in zip(high, low)], n=n, seed=seed)


def spearman(xs, ys):
    """Rank correlation; reported as a diagnostic association, never as causation."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    if len(xs) < 3:
        return None
    rx, ry = rank(list(xs)), rank(list(ys))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return None if den == 0 else num / den
