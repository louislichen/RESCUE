"""§15 mechanism regression sanity check on a small predetermined subset.

Not another experiment: two predetermined ledger rows, enough to show that
  * a hook installed with EMPTY maps is still exactly stock-equivalent;
  * the Teacher intervention produces a non-zero attention / logit effect;
  * the Null intervention also reaches the intended path (non-zero effect);
  * the full-window arm really intervenes on cached-decode queries (B does not).

Top-1 flipping is explicitly NOT required.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from four_arm_common import (DECODE_PHASE, PREFILL_PHASE, build_backend, cm_for,  # noqa: E402
                            load_config, load_harness, load_modules, maps_for)


def _ctx(backend, steering_cm, maps, alpha, layers, phase, install_hook: bool):
    """nullcontext for STOCK, or the harness CM actually installed (maps may be empty)."""
    if not install_hook:
        return contextlib.nullcontext()
    return steering_cm(backend.student, maps, alpha, backend.eps,
                       selected_layers=layers, phase=phase)


def prefill(backend, steering_cm, target, maps, alpha, layers, phase,
            install_hook: bool = True):
    with torch.no_grad():
        with _ctx(backend, steering_cm, maps, alpha, layers, phase, install_hook) as st:
            o = backend.student(**backend._ids(target), use_cache=True,
                                output_attentions=False)
            n = 0 if st is None else len(st.mix_log)
    return o.logits.detach().clone(), n


def decode_run(backend, steering_cm, target, maps, alpha, layers, steps, steered,
               prefill_hook: bool = True):
    """Prefill then `steps` cached decode steps; returns per-step last-row logits."""
    with torch.no_grad():
        with _ctx(backend, steering_cm, maps, alpha, layers, PREFILL_PHASE,
                  prefill_hook):
            pf = backend.student(**backend._ids(target), use_cache=True,
                                 output_attentions=False)
        past = pf.past_key_values
        nxt = int(torch.argmax(pf.logits[:, -1, :], dim=-1).item())
        lasts = [pf.logits[:, -1, :].detach().float().clone()]
        mix = 0
        for t in range(1, steps + 1):
            ids = torch.tensor([[nxt]], device=backend.device)
            mask = torch.ones((1, len(target.input_ids) + t), device=backend.device)
            if steered:
                with _ctx(backend, steering_cm, maps, alpha, layers, DECODE_PHASE,
                          True) as st:
                    o = backend.student(input_ids=ids, attention_mask=mask,
                                        past_key_values=past, use_cache=True,
                                        output_attentions=False)
                    mix += len(st.mix_log)
            else:
                with _ctx(backend, steering_cm, maps, alpha, layers, DECODE_PHASE,
                          False):
                    o = backend.student(input_ids=ids, attention_mask=mask,
                                        past_key_values=past, use_cache=True,
                                        output_attentions=False)
            past = o.past_key_values
            nxt = int(torch.argmax(o.logits[:, -1, :], dim=-1).item())
            lasts.append(o.logits[:, -1, :].detach().float().clone())
    return lasts, mix


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--source-lock", required=True)
    ap.add_argument("--teacher-model", required=True)
    ap.add_argument("--student-model", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--validation-ledger", required=True)
    ap.add_argument("--validation-prior-cache", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--decode-steps", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    H = load_harness(args.harness_root)
    backend_mod, runner_mod, steering = load_modules(H, args.source_dir, args.source_lock)
    backend = build_backend(backend_mod, args, steering)
    layers = set(backend.student_layers)
    steering_cm = steering.qwen2_adapter_steering
    rows = runner_mod.load_rows(Path(args.validation_ledger), Path(args.dataset_path))
    priors = torch.load(Path(args.validation_prior_cache), map_location="cpu",
                        weights_only=True)

    # in-memory measurement of the frozen harness mixing function
    cm_func = steering_cm
    ns = getattr(cm_func, "__wrapped__", cm_func).__globals__
    if "_apply_mix" not in ns:
        raise RuntimeError("harness steering namespace has no _apply_mix")
    original_mix = ns["_apply_mix"]
    calls: list[dict] = []

    def measured_mix(state, layer_index, attn_probs, key_len):
        pre = attn_probs[:, :, -1, :].detach().float()
        mixed = original_mix(state, layer_index, attn_probs, key_len)
        post = mixed[:, :, -1, :].detach().float()
        calls.append({"layer": int(layer_index), "phase": state.phase,
                      "key_len": int(key_len),
                      "attn_l1_delta": float((post - pre).abs().sum())})
        return mixed

    ns["_apply_mix"] = measured_mix
    records = []
    try:
        for row in rows[:args.rows]:
            sid = row["canonical_source_id"]
            target = backend.prompt_target(row["prompt"], row["answer"])
            prior = priors[sid]
            with torch.no_grad():
                rows_s = backend._student_rows(target)
                mask = torch.tensor(target.attention_mask, device=backend.device)
                mapped, alpha, _ = backend.adapter(prior.to(backend.device), rows_s, mask)
                null_map, null_alpha, _ = backend.adapter(
                    prior.roll(1, dims=-1).to(backend.device), rows_s, mask)
            lm, la = maps_for(backend, mapped, alpha)
            nm, na = maps_for(backend, null_map, null_alpha)

            logits_a, _ = prefill(backend, steering_cm, target, {}, None, layers,
                                  PREFILL_PHASE, install_hook=False)          # STOCK
            logits_empty, _ = prefill(backend, steering_cm, target, {}, {}, layers,
                                      PREFILL_PHASE, install_hook=True)       # HOOK_EMPTY
            calls.clear()
            logits_b, n_b = prefill(backend, steering_cm, target, lm, la, layers,
                                    PREFILL_PHASE)
            l1_b = sum(c["attn_l1_delta"] for c in calls)
            layers_b = sorted({c["layer"] for c in calls})
            calls.clear()
            logits_d, n_d = prefill(backend, steering_cm, target, nm, na, layers,
                                    PREFILL_PHASE)
            l1_d = sum(c["attn_l1_delta"] for c in calls)
            layers_d = sorted({c["layer"] for c in calls})

            dec_a, _ = decode_run(backend, steering_cm, target, {}, None, layers,
                                  args.decode_steps, False, prefill_hook=False)
            dec_empty, _ = decode_run(backend, steering_cm, target, {}, {}, layers,
                                      args.decode_steps, False, prefill_hook=True)
            calls.clear()
            dec_b, mix_b = decode_run(backend, steering_cm, target, lm, la, layers,
                                      args.decode_steps, False)      # prefill-only: no decode hook
            dec_c, mix_c = decode_run(backend, steering_cm, target, lm, la, layers,
                                      args.decode_steps, True)
            l1_c = sum(c["attn_l1_delta"] for c in calls)
            dec_d, mix_d = decode_run(backend, steering_cm, target, nm, na, layers,
                                      args.decode_steps, True)

            rec = {
                "canonical_source_id": sid,
                "source_index": row["source_index"],
                "prompt_len": len(target.input_ids),
                "selected_layers": sorted(layers),
                "empty_hook": {
                    "prefill_full_tensor_exact": bool(torch.equal(logits_a, logits_empty)),
                    "prefill_max_abs_delta": float((logits_a - logits_empty).abs().max()),
                    "decode_last_row_exact": [bool(torch.equal(x, y))
                                              for x, y in zip(dec_a, dec_empty)],
                    "decode_max_abs_delta": max(float((x - y).abs().max())
                                                for x, y in zip(dec_a, dec_empty)),
                },
                "teacher_prefill": {
                    "intervention_calls": n_b, "layers": layers_b,
                    "attn_l1_delta_total": l1_b,
                    "logits_max_abs_delta_vs_A": float((logits_a - logits_b).abs().max()),
                },
                "null_prefill": {
                    "intervention_calls": n_d, "layers": layers_d,
                    "attn_l1_delta_total": l1_d,
                    "logits_max_abs_delta_vs_A": float((logits_a - logits_d).abs().max()),
                },
                "prefill_only_decode": {
                    "arm": "B", "decode_mix_calls": mix_b,
                    "decode_last_row_max_abs_delta_vs_A": max(
                        float((x - y).abs().max()) for x, y in zip(dec_b, dec_a)),
                },
                "full_window_decode": {
                    "arm": "C", "decode_mix_calls": mix_c,
                    "attn_l1_delta_total_decode": l1_c,
                    "decode_last_row_max_abs_delta_vs_A": max(
                        float((x - y).abs().max()) for x, y in zip(dec_c, dec_a)),
                },
                "full_window_null_decode": {
                    "arm": "D", "decode_mix_calls": mix_d,
                    "decode_last_row_max_abs_delta_vs_A": max(
                        float((x - y).abs().max()) for x, y in zip(dec_d, dec_a)),
                },
            }
            rec["status"] = "PASS" if (
                rec["empty_hook"]["prefill_full_tensor_exact"]
                and all(rec["empty_hook"]["decode_last_row_exact"])
                and rec["teacher_prefill"]["intervention_calls"] > 0
                and rec["teacher_prefill"]["attn_l1_delta_total"] > 0
                and rec["teacher_prefill"]["logits_max_abs_delta_vs_A"] > 0
                and rec["null_prefill"]["intervention_calls"] > 0
                and rec["null_prefill"]["attn_l1_delta_total"] > 0
                and rec["prefill_only_decode"]["decode_mix_calls"] == 0
                and rec["full_window_decode"]["decode_mix_calls"] > 0
                and rec["full_window_null_decode"]["decode_mix_calls"] > 0) else "FAIL"
            records.append(rec)
            print(json.dumps({"src": row["source_index"], "status": rec["status"],
                              "empty_exact": rec["empty_hook"]["prefill_full_tensor_exact"],
                              "B_l1": rec["teacher_prefill"]["attn_l1_delta_total"],
                              "C_decode_mix": rec["full_window_decode"]["decode_mix_calls"],
                              "D_decode_mix": rec["full_window_null_decode"]["decode_mix_calls"]},
                             ensure_ascii=False), flush=True)
    finally:
        ns["_apply_mix"] = original_mix

    payload = {
        "task_id": "RESCUE-MINIMAL-FOUR-ARM-VALID-RERUN-001",
        "stage": "mechanism_regression_sanity",
        "subset": f"first {args.rows} ledger rows (predetermined, not selected on results)",
        "decode_steps": args.decode_steps,
        "top1_flip_required": False,
        "config_hash": cfg["config_hash"],
        "checkpoint_sha256": cfg["checkpoint_sha256"],
        "records": records,
        "status": "PASS" if all(r["status"] == "PASS" for r in records) else "FAIL",
    }
    H.atomic_write_text(out / "four_arm_mechanism.json",
                        json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"gate": "mechanism_sanity", "status": payload["status"]},
                     ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
