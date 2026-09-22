"""RESCUE 002 / P1: real Adapter action-path + task-gradient diagnostic.

Read-only w.r.t. every existing artifact. Loads an existing frozen checkpoint,
runs paired conditions on a predeclared smoke slice, and measures whether the
Adapter's input-dependent output actually reaches the supervised predictions.

Outputs (into --out):
  adapter_path_diagnostic.jsonl
  adapter_gradient_diagnostic.json
  adapter_path_trace.json
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def select_smoke_rows(rows, count=6):
    """Predeclared rule: source_index ascending; first and midpoint of each third."""
    ordered = sorted(rows, key=lambda r: int(r["source_index"]))
    n = len(ordered)
    picked, seen = [], set()
    for third in range(3):
        lo = (third * n) // 3
        hi = ((third + 1) * n) // 3
        block = ordered[lo:hi]
        if not block:
            continue
        for pos in (0, len(block) // 2):
            row = block[pos]
            if row["source_index"] not in seen:
                seen.add(row["source_index"])
                picked.append(row)
    return picked[:count]


def jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    p = p.float().clamp_min(0)
    q = q.float().clamp_min(0)
    p = p / p.sum(-1, keepdim=True).clamp_min(eps)
    q = q / q.sum(-1, keepdim=True).clamp_min(eps)
    m = 0.5 * (p + q)
    def kl(a, b):
        return (a * ((a.clamp_min(eps)).log() - (b.clamp_min(eps)).log())).sum(-1)
    return float((0.5 * kl(p, m) + 0.5 * kl(q, m)).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--teacher-model", required=True)
    ap.add_argument("--student-model", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--dev-ledger", required=True)
    ap.add_argument("--source-lock", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--samples", type=int, default=6)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = str(Path(args.source_dir).resolve())
    sys.path.insert(0, src)
    sys.path.insert(0, str(Path(args.source_lock).resolve()))

    runner = importlib.import_module("sequence_loss_runner")
    backend_mod = importlib.import_module("qwen_adapter_backend")
    steering = importlib.import_module("steering")

    # ---- real measurement of the actual intervention channel -------------
    events: list[dict] = []
    original_mix = steering._apply_adapter_prob_mix

    def recording_mix(module, attn_probs):
        before = attn_probs[:, :, -1, :].detach().float().clone()
        after = original_mix(module, attn_probs)
        after_row = after[:, :, -1, :].detach().float()
        events.append({
            "layer": int(getattr(module, "_attn_adapter_layer_idx", -1)),
            "pre": before.cpu(),
            "post": after_row.cpu(),
        })
        return after

    steering._apply_adapter_prob_mix = recording_mix

    rows = runner.load_rows(Path(args.dev_ledger), Path(args.dataset_path))
    smoke = select_smoke_rows(rows, args.samples)
    selection_rule = ("source_index ascending; first and midpoint of each actual "
                      "prior-length third; selected before any condition ran")

    backend = backend_mod.QwenAdapterBackend(
        args.teacher_model, args.student_model, args.device,
        source_lock=args.source_lock, load_teacher=False)
    step = backend.load_checkpoint(args.checkpoint)
    adapter = backend.adapter
    has_sequence = hasattr(backend, "sequence_treatment_step")

    def prefill_steered(target, prior_map=None, alpha_map=None, phase="VALIDATION_PREFILL"):
        """Real prefill: returns last-position logits and the attention events."""
        events.clear()
        empty = {} if prior_map is None else prior_map
        alpha = {} if alpha_map is None else alpha_map
        with torch.no_grad():
            with backend._steering(backend.student, empty, alpha, backend.eps,
                                   selected_layers=set(backend.student_layers), phase=phase):
                o = backend.student(**backend._ids(target), use_cache=False, output_attentions=False)
        logits = o.logits[:, -1, :].float().detach().cpu().squeeze(0)
        attn = [{"layer": e["layer"],
                 "pre": e["pre"], "post": e["post"]} for e in events]
        return logits, attn

    def plain_prefill(target):
        with torch.no_grad():
            o = backend.student(**backend._ids(target), use_cache=False, output_attentions=False)
        return o.logits[:, -1, :].float().detach().cpu().squeeze(0)

    # ---- gradient helpers -------------------------------------------------
    PARAM_GROUPS = ["base.mix_logits", "base.residual_logits", "base.gate_logits",
                    "base.log_temperature", "entropy_gate_weight"]

    def group_grads():
        g = {}
        for name, p in adapter.named_parameters():
            g[name] = None if p.grad is None else p.grad.detach().float().clone()
        return g

    def summarize_grads(g):
        s = {}
        for name in PARAM_GROUPS:
            t = g.get(name)
            if t is None:
                s[name] = {"present": False, "norm": None, "nonzero": None}
            else:
                s[name] = {"present": True, "norm": float(t.norm()),
                           "nonzero": bool(torch.any(t != 0).item()),
                           "max_abs": float(t.abs().max())}
        total = sum(t.float().pow(2).sum() for t in g.values() if t is not None)
        s["_all"] = {"norm": float(torch.sqrt(total)) if torch.numel(total) else 0.0}
        return s

    per_sample, grad_records = [], []
    for idx, row in enumerate(smoke, 1):
        target = backend.prompt_target(row["prompt"], row["answer"])
        prior = torch.load(Path(args.checkpoint).parent / "validation_teacher_priors.pt",
                           map_location="cpu", weights_only=True)[row["canonical_source_id"]]
        base = plain_prefill(target)
        with torch.no_grad():
            student_rows = backend._student_rows(target)
            mask = torch.tensor(target.attention_mask, device=backend.device)
            mapped_t, alpha_t, entropy = adapter(prior.to(backend.device), student_rows, mask)
            null_prior = prior.roll(1, dims=-1)
            mapped_n, alpha_n, null_entropy = adapter(null_prior.to(backend.device), student_rows, mask)
        maps_t = {l: mapped_t[i] for i, l in enumerate(backend.student_layers)}
        gates_t = {l: alpha_t[i] for i, l in enumerate(backend.student_layers)}
        maps_n = {l: mapped_n[i] for i, l in enumerate(backend.student_layers)}
        gates_n = {l: alpha_n[i] for i, l in enumerate(backend.student_layers)}
        gates_0 = {l: torch.zeros_like(alpha_t[i]) for i, l in enumerate(backend.student_layers)}

        logits_ad, attn_ad = prefill_steered(target, maps_t, gates_t)
        logits_nu, attn_nu = prefill_steered(target, maps_n, gates_n)
        logits_a0, attn_a0 = prefill_steered(target, maps_t, gates_0)

        # real decode/generation invocations of the intervention channel
        events.clear()
        with torch.no_grad():
            with backend._steering(backend.student, maps_t, gates_t, backend.eps,
                                   selected_layers=set(backend.student_layers), phase="VALIDATION_PREFILL"):
                pf = backend.student(**backend._ids(target), use_cache=True, output_attentions=False)
            n_prefill_events = len(events)
            past = pf.past_key_values
            nxt = int(torch.argmax(pf.logits[:, -1, :], dim=-1).item())
            decode_events = 0
            decode_hook_installed = 0
            for _ in range(args.max_new_tokens):
                if backend.student_tokenizer.eos_token_id is not None and nxt == backend.student_tokenizer.eos_token_id:
                    break
                if getattr(backend.student, "_attn_adapter_state", None) is not None:
                    decode_hook_installed += 1
                ids = torch.tensor([[nxt]], device=backend.device, dtype=torch.long)
                m = torch.ones((1, len(target.input_ids) + 1), device=backend.device, dtype=torch.long)
                before = len(events)
                outd = backend.student(input_ids=ids, attention_mask=m, past_key_values=past,
                                       use_cache=True, output_attentions=False)
                decode_events += len(events) - before
                past = outd.past_key_values
                nxt = int(torch.argmax(outd.logits[:, -1, :], dim=-1).item())

        gold = int(target.target_id)
        def prob_of(logits_row, tid):
            return float(torch.softmax(logits_row, dim=-1)[tid])
        def ce_of(logits_row, tid):
            return float(torch.nn.functional.cross_entropy(logits_row.view(1, -1),
                                                           torch.tensor([tid])))

        attn_delta = []
        for e_ad, e_a0 in zip(attn_ad, attn_a0):
            attn_delta.append({
                "layer": e_ad["layer"],
                "l1_treatment_vs_base": float((e_ad["post"] - e_ad["pre"]).abs().sum()),
                "l1_alpha0_vs_base": float((e_a0["post"] - e_a0["pre"]).abs().sum()),
                "l1_treatment_vs_null": None,
            })
        for d, e_ad, e_nu in zip(attn_delta, attn_ad, attn_nu):
            d["l1_treatment_vs_null"] = float((e_ad["post"] - e_nu["post"]).abs().sum())

        rec = {
            "sample": idx, "canonical_source_id": row["canonical_source_id"],
            "source_index": row["source_index"], "answer": row["answer"],
            "answer_token_count": len(target.answer_ids), "prompt_len": len(target.input_ids),
            "checkpoint_step": step,
            "alpha": {
                "treatment_mean": float(alpha_t.mean()), "treatment_std": float(alpha_t.std()),
                "null_mean": float(alpha_n.mean()), "null_std": float(alpha_n.std()),
                "max_abs_treatment_minus_null": float((alpha_t - alpha_n).abs().max()),
                "alpha0_is_exact_zero": bool(torch.all(gates_0[backend.student_layers[0]] == 0)),
            },
            "entropy": {
                "mean": float(entropy.mean()), "std": float(entropy.std()),
                "max_abs_entropy_minus_null_entropy": float((entropy - null_entropy).abs().max()),
            },
            "prior": {
                "teacher_vs_student_jsd": jsd(mapped_t.reshape(-1, mapped_t.shape[-1]),
                                              student_rows.reshape(-1, mapped_t.shape[-1])),
                "mapped_t_vs_mapped_n_jsd": jsd(mapped_t.reshape(-1, mapped_t.shape[-1]),
                                                mapped_n.reshape(-1, mapped_n.shape[-1])),
                "mapped_t_vs_mapped_n_l1": float((mapped_t - mapped_n).abs().mean()),
                "mapped_t_vs_mapped_n_maxabs": float((mapped_t - mapped_n).abs().max()),
                "mapped_t_is_normalized": bool(torch.allclose(
                    mapped_t.sum(-1), torch.ones_like(mapped_t.sum(-1)), atol=1e-4)),
            },
            "attention_delta": attn_delta,
            "logits": {
                "max_abs_adapter_minus_base": float((logits_ad - base).abs().max()),
                "max_abs_null_minus_base": float((logits_nu - base).abs().max()),
                "max_abs_alpha0_minus_base": float((logits_a0 - base).abs().max()),
                "max_abs_null_minus_adapter": float((logits_nu - logits_ad).abs().max()),
                "argmax_base": int(torch.argmax(base).item()),
                "argmax_adapter": int(torch.argmax(logits_ad).item()),
                "argmax_null": int(torch.argmax(logits_nu).item()),
                "argmax_alpha0": int(torch.argmax(logits_a0).item()),
                "top1_changed_vs_base_adapter": bool(int(torch.argmax(base)) != int(torch.argmax(logits_ad))),
                "top1_changed_vs_base_null": bool(int(torch.argmax(base)) != int(torch.argmax(logits_nu))),
                "top1_changed_vs_base_alpha0": bool(int(torch.argmax(base)) != int(torch.argmax(logits_a0))),
                "top1_changed_null_vs_adapter": bool(int(torch.argmax(logits_nu)) != int(torch.argmax(logits_ad))),
            },
            "target_token": {
                "id": gold,
                "logit_base": float(base[gold]), "logit_adapter": float(logits_ad[gold]),
                "logit_null": float(logits_nu[gold]), "logit_alpha0": float(logits_a0[gold]),
                "prob_base": prob_of(base, gold), "prob_adapter": prob_of(logits_ad, gold),
                "prob_null": prob_of(logits_nu, gold), "prob_alpha0": prob_of(logits_a0, gold),
                "ce_base": ce_of(base, gold), "ce_adapter": ce_of(logits_ad, gold),
                "ce_null": ce_of(logits_nu, gold), "ce_alpha0": ce_of(logits_a0, gold),
            },
            "intervention_invocations": {
                "prefill_real_invocations": n_prefill_events,
                "decode_real_invocations": decode_events,
                "decode_steps_with_hook_installed": decode_hook_installed,
                "declared_prefill_invocations_field": "len(student_layers)=4 (literal, not measured)",
                "declared_cached_decode_invocations_field": 0,
            },
        }
        per_sample.append(rec)

        # ---------------- gradient separation ----------------
        if has_sequence:
            with torch.enable_grad():
                tr = backend.sequence_treatment_step(target, prior, student_rows)
                token_losses = tr["token_losses"]
                per_position = []
                for t in range(len(token_losses)):
                    adapter.zero_grad(set_to_none=True)
                    token_losses[t].backward(retain_graph=True)
                    per_position.append({"answer_position": t,
                                         "token_loss": float(token_losses[t].detach()),
                                         "grad": summarize_grads(group_grads())})
                adapter.zero_grad(set_to_none=True)
                token_losses[0].backward(retain_graph=True)
                g_first = summarize_grads(group_grads())
                adapter.zero_grad(set_to_none=True)
                if len(token_losses) > 1:
                    token_losses[1:].mean().backward(retain_graph=True)
                g_later = summarize_grads(group_grads())
                adapter.zero_grad(set_to_none=True)
                tr["loss"].backward(retain_graph=True)
                g_total = summarize_grads(group_grads())
                ratio = None
                if g_first["_all"]["norm"] and g_total["_all"]["norm"]:
                    ratio = g_total["_all"]["norm"] / g_first["_all"]["norm"]
                grad_records.append({
                    "sample": idx, "canonical_source_id": row["canonical_source_id"],
                    "answer_token_count": len(token_losses),
                    "objective_reported": "mean over answer-token CE",
                    "objective_effective": "1/T * first-token CE only (later tokens outside steering scope)",
                    "grad_first_token_only": g_first,
                    "grad_later_tokens_only": g_later,
                    "grad_total_loss": g_total,
                    "grad_total_over_first_norm_ratio": ratio,
                    "expected_ratio": 1.0 / len(token_losses),
                    "per_answer_position": per_position,
                })
                adapter.zero_grad(set_to_none=True)

    # ---- regularizer presence -------------------------------------------
    reg_module = importlib.import_module("adapter")
    has_reg_fn = hasattr(reg_module, "adapter_regularization")
    contract = Path(args.source_lock).parent / "frozen_training_config.json"
    frozen_config = json.loads(contract.read_text(encoding="utf-8")) if contract.exists() else None

    path_jsonl = out / "adapter_path_diagnostic.jsonl"
    path_jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in per_sample) + "\n",
                          encoding="utf-8")

    def agg(key, sub, field):
        vals = [r[key][field] for r in per_sample if r[key].get(field) is not None]
        return {"n": len(vals), "mean": sum(vals) / len(vals) if vals else None,
                "min": min(vals) if vals else None, "max": max(vals) if vals else None}

    grad_summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256_file(Path(args.checkpoint)),
        "checkpoint_step": step,
        "source_dir": src,
        "source_dir_has_sequence_treatment_step": has_sequence,
        "gradient_diagnostic_available": bool(grad_records),
        "regularization_term_present_in_training_step": False,
        "regularization_implementation_available_in_source_lock": has_reg_fn,
        "regularization_coefficients_in_frozen_contract": (
            None if not frozen_config else frozen_config),
        "note": ("The trained candidate's optimizer step backpropagates only the answer-token "
                 "cross-entropy; no regularizer term is added, so every non-zero gradient is a "
                 "task-loss gradient. Gradient separation below is therefore task-vs-task."),
        "records": grad_records,
    }
    (out / "adapter_gradient_diagnostic.json").write_text(
        json.dumps(grad_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    trace = {
        "selection_rule": selection_rule,
        "selected": [{"sample": i + 1, "canonical_source_id": r["canonical_source_id"],
                      "source_index": r["source_index"]} for i, r in enumerate(smoke)],
        "checkpoint": args.checkpoint, "checkpoint_step": step,
        "checkpoint_sha256": sha256_file(Path(args.checkpoint)),
        "device": args.device,
        "aggregate": {
            "alpha_std_within_sample_mean": agg("alpha", "alpha", "treatment_std"),
            "alpha_max_abs_treatment_minus_null": agg("alpha", "alpha", "max_abs_treatment_minus_null"),
            "logits_max_abs_adapter_minus_base": agg("logits", "logits", "max_abs_adapter_minus_base"),
            "logits_max_abs_null_minus_adapter": agg("logits", "logits", "max_abs_null_minus_adapter"),
            "prior_mapped_t_vs_n_l1": agg("prior", "prior", "mapped_t_vs_mapped_n_l1"),
            "prior_mapped_t_vs_n_maxabs": agg("prior", "prior", "mapped_t_vs_mapped_n_maxabs"),
        },
        "top1_changed_counts": {
            "adapter_vs_base": sum(1 for r in per_sample if r["logits"]["top1_changed_vs_base_adapter"]),
            "null_vs_base": sum(1 for r in per_sample if r["logits"]["top1_changed_vs_base_null"]),
            "alpha0_vs_base": sum(1 for r in per_sample if r["logits"]["top1_changed_vs_base_alpha0"]),
            "null_vs_adapter": sum(1 for r in per_sample if r["logits"]["top1_changed_null_vs_adapter"]),
            "n": len(per_sample),
        },
    }
    (out / "adapter_path_trace.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "OK", "samples": len(per_sample),
                      "grad_records": len(grad_records),
                      "top1_changed": trace["top1_changed_counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
