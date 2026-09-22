"""Supplementary GPU gates for RESCUE-PRE-EXPERIMENT-CORRECTNESS-GATE-V3-001.

The frozen `gpu_smokes.py` covers §3, §5, §7, §8, §9 and the prefill half of §4.  Two
requirements of the task text are not covered by it:

  §4  cached-decode transparency: STOCK == HOOK_EMPTY for 3-6 decoded tokens
      (logits / selected token / cache length / key length), not prefill only;
  §6  intervention effect: a real TEACHER intervention must change internal numbers
      (attention L1 delta, logits max/mean delta) - `call_count > 0` alone is not a PASS.

This file is gate evidence, NOT part of the frozen manifest: it imports the frozen
modules read-only, patches the in-memory steering namespace only, and writes its
results into the gate output directory.

Memory note: a 1582-token prefill produces logits of shape [1, 1582, 151936], i.e. ~480 MiB
in bf16 and ~960 MiB in fp32.  Tensors are therefore compared in their native dtype and
only the last query row is upcast, which keeps one gate inside an 11 GiB card.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import sys
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def mem_mib() -> float:
    return torch.cuda.memory_allocated() / 2**20


def cache_lengths(cache):
    per_layer_len, per_layer_key = [], []
    for i in range(len(cache)):
        per_layer_len.append(int(cache.get_seq_length(i)))
        try:
            k = cache.layers[i].keys
            per_layer_key.append(None if k is None else int(k.shape[-2]))
        except Exception:
            per_layer_key.append(None)
    return per_layer_len, per_layer_key


def last_row(out_obj):
    """fp32 copy of the final query row only (0.6 MiB instead of 960 MiB)."""
    return out_obj.logits[:, -1, :].detach().float().clone()


def snapshot(out_obj, cache, phase: str, keep_full: bool = False) -> dict:
    per_layer_len, per_layer_key = cache_lengths(cache)
    last = last_row(out_obj)
    snap = {"phase": phase,
            "logits_last": last,
            "next_token": int(torch.argmax(last, dim=-1).item()),
            "cache_len_per_layer": per_layer_len,
            "key_len_per_layer": per_layer_key}
    if keep_full:
        snap["logits_full_native"] = out_obj.logits.detach().clone()
    return snap


def compare_steps(a: dict, b: dict) -> dict:
    out = {
        "phase": a["phase"],
        "logits_exact": bool(torch.equal(a["logits_last"], b["logits_last"])),
        "logits_last_max_abs_delta": float((a["logits_last"] - b["logits_last"]).abs().max()),
        "logits_last_mean_abs_delta": float((a["logits_last"] - b["logits_last"]).abs().mean()),
        "selected_token_equal": a["next_token"] == b["next_token"],
        "stock_selected_token": a["next_token"],
        "cache_len_equal": a["cache_len_per_layer"] == b["cache_len_per_layer"],
        "key_len_equal": a["key_len_per_layer"] == b["key_len_per_layer"],
        "stock_cache_len_per_layer": a["cache_len_per_layer"],
        "stock_key_len_per_layer": a["key_len_per_layer"],
    }
    if "logits_full_native" in a and "logits_full_native" in b:
        out["logits_full_native_exact"] = bool(
            torch.equal(a["logits_full_native"], b["logits_full_native"]))
    return out


def gate_empty_hook_decode(args, H, G, backend_mod, runner_mod, steering, out):
    rows = G.sample_rows(args, runner_mod, args.samples)
    backend = G.build_backend(args, backend_mod, steering.qwen2_adapter_steering)
    layers = set(backend.student_layers)
    mem = {"after_backend_build": mem_mib()}

    def cm_for(hook, maps, gates):
        return (steering.qwen2_adapter_steering(backend.student, maps, gates, backend.eps,
                                                selected_layers=layers,
                                                phase=H.STEERED_PHASE)
                if hook else contextlib.nullcontext())

    records = []
    for row in rows:
        target = backend.prompt_target(row["prompt"], row["answer"])
        with torch.no_grad():
            rs = backend._student_rows(target)
            mask = torch.tensor(target.attention_mask, device=backend.device)
            prior = torch.load(Path(args.validation_prior_cache), map_location="cpu",
                               weights_only=True)[row["canonical_source_id"]]
            mapped, alpha, _ = backend.adapter(prior.to(backend.device), rs, mask)
        lm, la = G.maps_for(backend, mapped, alpha)

        def run(hook: bool, maps, gates):
            steps = []
            with torch.no_grad():
                with cm_for(hook, maps, gates):
                    o = backend.student(**backend._ids(target), use_cache=True,
                                        output_attentions=False)
                cache = o.past_key_values
                steps.append(snapshot(o, cache, "prefill", keep_full=True))
                nxt = steps[-1]["next_token"]
                for t in range(1, args.decode_steps + 1):
                    ids = torch.tensor([[nxt]], device=backend.device)
                    m = torch.ones((1, len(target.input_ids) + t), device=backend.device)
                    with cm_for(hook, maps, gates):
                        o = backend.student(input_ids=ids, attention_mask=m,
                                            past_key_values=cache, use_cache=True,
                                            output_attentions=False)
                    cache = o.past_key_values
                    steps.append(snapshot(o, cache, f"decode_{t}"))
                    nxt = steps[-1]["next_token"]
                del o
            return steps

        stock_steps = run(False, {}, {})
        empty_steps = run(True, {}, {})
        mem["after_stock_and_empty"] = mem_mib()
        steps = [compare_steps(a, b) for a, b in zip(stock_steps, empty_steps)]
        stock_lasts = [s["logits_last"] for s in stock_steps]     # 0.6 MiB each
        for s in stock_steps + empty_steps:
            s.pop("logits_full_native", None)
            s.pop("logits_last", None)
        del stock_steps, empty_steps
        gc.collect()
        torch.cuda.empty_cache()

        steered_steps = run(True, lm, la)
        steered_delta = max(float((a["logits_last"] - b).abs().max())
                            for a, b in zip(steered_steps, stock_lasts))
        rec = {
            "canonical_source_id": row["canonical_source_id"],
            "prompt_len": len(target.input_ids),
            "decode_steps": args.decode_steps,
            "selected_layers": sorted(layers),
            "comparison_scope": ("full-vocab last-query-row logits in fp32 for every step; "
                                 "the complete [1, prompt_len, vocab] prefill tensor is also "
                                 "compared bit-exactly in its native dtype"),
            "steps": steps,
            "all_steps_exact": all(s["logits_exact"] and s["selected_token_equal"]
                                   and s["cache_len_equal"] and s["key_len_equal"]
                                   and s.get("logits_full_native_exact", True)
                                   for s in steps),
            "max_abs_logits_delta_overall": max(s["logits_last_max_abs_delta"]
                                                for s in steps),
            "prefill_full_tensor_exact": steps[0].get("logits_full_native_exact"),
            "steered_decode_last_row_max_abs_delta_vs_stock": steered_delta,
            "cache_len_progression_stock": [s["stock_cache_len_per_layer"][0] for s in steps],
        }
        rec["status"] = "PASS" if rec["all_steps_exact"] else "FAIL"
        records.append(rec)
        del steered_steps, stock_lasts, mapped, alpha, prior, rs, mask
        gc.collect()
        torch.cuda.empty_cache()
    mem["at_end"] = mem_mib()

    payload = {"invariant": "STOCK == HOOK_EMPTY over prefill AND cached decode",
               "decode_steps": args.decode_steps,
               "memory_mib": mem, "records": records,
               "status": "PASS" if all(r["status"] == "PASS" for r in records) else "FAIL"}
    H.atomic_write_text(out / "v3_empty_hook_decode.json",
                        json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return {"gate": "empty_hook_decode", "status": payload["status"],
            "detail": {"max_abs_logits_delta": max(r["max_abs_logits_delta_overall"]
                                                   for r in records),
                       "steps_compared": [len(r["steps"]) for r in records],
                       "prefill_full_tensor_exact": [r["prefill_full_tensor_exact"]
                                                     for r in records],
                       "all_steps_exact": all(r["all_steps_exact"] for r in records)}}


def gate_intervention_effect(args, H, G, backend_mod, runner_mod, steering, out):
    rows = G.sample_rows(args, runner_mod, args.samples)
    backend = G.build_backend(args, backend_mod, steering.qwen2_adapter_steering)
    mem = {"after_backend_build": mem_mib()}
    layers = set(backend.student_layers)
    sel = sorted(layers)

    # in-memory only: wrap the frozen harness mixing function to measure the real effect.
    # NOTE: @contextlib.contextmanager returns a helper defined in contextlib, so the
    # steering namespace is reachable through __wrapped__, not through __globals__.
    cm_func = steering.qwen2_adapter_steering
    ns = getattr(cm_func, "__wrapped__", cm_func).__globals__
    if "_apply_mix" not in ns:
        raise RuntimeError("harness steering namespace has no _apply_mix; refusing to "
                           "report an unmeasured intervention")
    original_mix = ns["_apply_mix"]
    calls: list[dict] = []

    def measured_mix(state, layer_index, attn_probs, key_len):
        pre = attn_probs[:, :, -1, :].detach().float()
        mixed = original_mix(state, layer_index, attn_probs, key_len)
        post = mixed[:, :, -1, :].detach().float()
        alpha = state.layer_to_alpha[layer_index].detach().float()
        calls.append({
            "layer": int(layer_index), "phase": state.phase, "key_len": int(key_len),
            "attn_l1_delta": float((post - pre).abs().sum()),
            "attn_max_abs_delta": float((post - pre).abs().max()),
            "alpha_mean": float(alpha.mean()), "alpha_max": float(alpha.max()),
        })
        return mixed

    ns["_apply_mix"] = measured_mix
    records = []
    try:
        for row in rows:
            target = backend.prompt_target(row["prompt"], row["answer"])
            with torch.no_grad():
                rs = backend._student_rows(target)
                mask = torch.tensor(target.attention_mask, device=backend.device)
                prior = torch.load(Path(args.validation_prior_cache), map_location="cpu",
                                   weights_only=True)[row["canonical_source_id"]]
                mapped, alpha, _ = backend.adapter(prior.to(backend.device), rs, mask)
            lm, la = G.maps_for(backend, mapped, alpha)

            arms = {}
            for arm in ("stock", "hook_empty", "teacher"):
                hook = arm != "stock"
                maps, gates = (({}, {}) if arm in ("stock", "hook_empty") else (lm, la))
                calls.clear()
                with torch.no_grad():
                    cm = (steering.qwen2_adapter_steering(
                        backend.student, maps, gates, backend.eps, selected_layers=layers,
                        phase=H.STEERED_PHASE) if hook else contextlib.nullcontext())
                    with cm as state:
                        o = backend.student(**backend._ids(target), use_cache=True,
                                            output_attentions=False)
                    last = last_row(o)
                    keep_full = arm in ("stock", "hook_empty")
                    full = o.logits.detach().clone() if keep_full else None
                    del o
                hook_mix_log = [] if state is None else list(state.mix_log)
                per_layer = {}
                for c in calls:
                    e = per_layer.setdefault(c["layer"], {"calls": 0, "attn_l1_delta": 0.0,
                                                          "attn_max_abs_delta": 0.0})
                    e["calls"] += 1
                    e["attn_l1_delta"] += c["attn_l1_delta"]
                    e["attn_max_abs_delta"] = max(e["attn_max_abs_delta"],
                                                  c["attn_max_abs_delta"])
                arms[arm] = {"intervention_count": len(calls),
                             "frozen_hook_mix_log_entries": len(hook_mix_log),
                             "wrapper_and_frozen_log_agree": len(calls) == len(hook_mix_log),
                             "per_layer": {str(k): v for k, v in sorted(per_layer.items())},
                             "logits_last": last, "logits_full_native": full,
                             "top1": int(torch.argmax(last, dim=-1).item()),
                             "alpha_mean": (float(alpha.mean()) if arm == "teacher" else None)}
                mem[f"after_{arm}"] = mem_mib()
                gc.collect()
                torch.cuda.empty_cache()

            stock = arms["stock"]
            empty = arms["hook_empty"]
            teacher = arms["teacher"]
            full_exact = (None if stock["logits_full_native"] is None
                          else bool(torch.equal(stock["logits_full_native"],
                                                empty["logits_full_native"])))
            stock["logits_full_native"] = None
            empty["logits_full_native"] = None
            gc.collect()
            torch.cuda.empty_cache()

            delta = (teacher["logits_last"] - stock["logits_last"]).abs()
            teacher_layers_nonzero = {
                str(l): teacher["per_layer"].get(str(l), {}).get("attn_l1_delta", 0.0) > 0
                for l in sel}
            rec = {
                "canonical_source_id": row["canonical_source_id"],
                "prompt_len": len(target.input_ids),
                "selected_layers": sel,
                "arm_intervention_counts": {a: arms[a]["intervention_count"]
                                            for a in ("stock", "hook_empty", "teacher")},
                "hook_empty_logits_last_exact": bool(torch.equal(stock["logits_last"],
                                                                 empty["logits_last"])),
                "hook_empty_prefill_full_tensor_exact": full_exact,
                "hook_empty_max_abs_logits_delta": float(
                    (stock["logits_last"] - empty["logits_last"]).abs().max()),
                "teacher_intervention_count": teacher["intervention_count"],
                "teacher_wrapper_frozen_log_agree": teacher["wrapper_and_frozen_log_agree"],
                "teacher_per_layer_attention": teacher["per_layer"],
                "teacher_attn_l1_delta_total": sum(
                    v["attn_l1_delta"] for v in teacher["per_layer"].values()),
                "teacher_attn_l1_delta_max_over_layers": max(
                    [v["attn_l1_delta"] for v in teacher["per_layer"].values()] or [0.0]),
                "teacher_all_selected_layers_nonzero_attention": all(
                    teacher_layers_nonzero.values()),
                "teacher_logits_max_abs_delta": float(delta.max()),
                "teacher_logits_mean_abs_delta": float(delta.mean()),
                "teacher_top1_flipped": teacher["top1"] != stock["top1"],
                "teacher_alpha_mean": teacher["alpha_mean"],
            }
            rec["status"] = "PASS" if (
                rec["arm_intervention_counts"]["stock"] == 0
                and rec["arm_intervention_counts"]["hook_empty"] == 0
                and rec["hook_empty_logits_last_exact"]
                and rec["hook_empty_prefill_full_tensor_exact"] is True
                and rec["teacher_intervention_count"] > 0
                and rec["teacher_wrapper_frozen_log_agree"]
                and rec["teacher_all_selected_layers_nonzero_attention"]
                and rec["teacher_logits_max_abs_delta"] > 0
                and rec["teacher_logits_mean_abs_delta"] > 0) else "FAIL"
            records.append(rec)
            del arms, mapped, alpha, prior, rs, mask, target
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        ns["_apply_mix"] = original_mix
    mem["at_end"] = mem_mib()

    payload = {
        "invariant": "HOOK_EMPTY == STOCK, and a TEACHER intervention produces a non-zero "
                     "internal numerical effect; `call_count > 0` alone is NOT a PASS",
        "pass_criteria": ["stock intervention count == 0",
                          "hook_empty intervention count == 0",
                          "hook_empty last-row logits bit-identical to stock",
                          "hook_empty full prefill logits bit-identical to stock",
                          "teacher intervention count > 0 (and matches the frozen hook log)",
                          "attention L1 delta > 0 on every selected layer",
                          "logits max |delta| > 0 and mean |delta| > 0"],
        "top1_flip_is_not_a_criterion": True,
        "memory_mib": mem,
        "records": records,
        "status": "PASS" if all(r["status"] == "PASS" for r in records) else "FAIL",
    }
    H.atomic_write_text(out / "v3_intervention_effect.json",
                        json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return {"gate": "intervention_effect", "status": payload["status"],
            "detail": {"teacher_interventions": [r["teacher_intervention_count"]
                                                 for r in records],
                       "attn_l1_delta_total": [r["teacher_attn_l1_delta_total"]
                                               for r in records],
                       "logits_max_abs_delta": [r["teacher_logits_max_abs_delta"]
                                                for r in records],
                       "top1_flipped": [r["teacher_top1_flipped"] for r in records]}}


GATES = {"empty_hook_decode": gate_empty_hook_decode,
         "intervention_effect": gate_intervention_effect}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True, choices=sorted(GATES) + ["all"])
    ap.add_argument("--harness-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--source-lock", required=True)
    ap.add_argument("--teacher-model", required=True)
    ap.add_argument("--student-model", required=True)
    ap.add_argument("--validation-ledger", required=True)
    ap.add_argument("--validation-prior-cache", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--decode-steps", type=int, default=5)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.harness_root).resolve()))
    from rescue_harness import harness as H              # noqa: E402
    from rescue_harness import gpu_smokes as G           # noqa: E402

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    backend_mod, runner_mod, steering = G.setup(args)
    G.make_steered(steering)

    results = []
    for name in (sorted(GATES) if args.gate == "all" else [args.gate]):
        try:
            results.append(GATES[name](args, H, G, backend_mod, runner_mod, steering, out))
        except Exception as exc:                        # fail closed, never silently pass
            results.append({"gate": name, "status": "FAIL",
                            "detail": {"exception": f"{type(exc).__name__}: {exc}",
                                       "traceback": traceback.format_exc()}})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
        gc.collect()
        torch.cuda.empty_cache()                        # each gate starts from a clean card
        print(f"[{name}] cuda_allocated_mib={mem_mib():.0f}", flush=True)
    H.atomic_write_text(out / "v3_ext_summary.json",
                        json.dumps({"results": results}, ensure_ascii=False, indent=2) + "\n")
    return 0 if all(r["status"] == "PASS" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
