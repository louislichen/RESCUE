"""TD — Student-side alignment and same-budget injection (Phase C of
RESCUE-DECISION-AWARE-TEACHER-SIGNAL-SPRINT-001).

Consumes the raw signed Teacher attribution `score[4,28,K]` produced by `td_teacher_attribution.py`
and answers sprint §13 questions (2) and (3) on the Student side:

  (2) does the projection preserve the signal   -> the frozen overlap operator's WEIGHTS are reused
      with the non-negativity / renormalisation step dropped (the Supervisor-approved path), then
      the frozen Adapter's learned head-mixing WEIGHTS are reused with `normalize_rows`, the
      residual mix and the final softmax dropped; the result is rescaled to the SAME per-head L1
      budget the frozen Teacher direction uses.  Structure is reported (sign agreement, top-k
      overlap, cancellation, coverage), never asserted.
  (3) can the Student injection exploit it      -> `d_TD` is pushed through the real frozen
      attention path under that same budget and the ACTUAL decision change is measured.

It also re-derives, deterministically, the two things Track A did not persist (`g_S` raw and
`d_O` raw) and refuses to continue unless the recomputation reproduces the frozen evidence:
per-head |g|_1 == the persisted `g_abs_sum_per_head`, per-head L1 budget == the persisted
`budget_l1_per_head`, and V_teacher / V_null / V_oracle == the persisted values.

No training, no optimizer, frozen weights, frozen tree untouched, oracle-style and
NON-DEPLOYABLE (it uses Target decision information).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dr_common import (build_backend, load_config, load_harness, load_modules,  # noqa: E402
                       maps_for, sha256_file)
from td_teacher_attribution import (DELTA_HELPER, DELTA_CALL, DETACH_LINE, GRAPH_LINES,  # noqa
                                    MIX_CALL, TASK_ID, greedy_transfer, prefix_sha,
                                    select_positions)

TOL_G = 1e-5          # persisted |g|_1 is rounded to 8 decimals
TOL_V = 1e-5          # persisted V values are full precision
TOL_MARGIN = 1e-4


def signed_overlap(values: torch.Tensor, teacher_offsets, student_offsets, eps: float = 1e-12):
    """The frozen `project_teacher_prior_to_student` operator WITHOUT its probability steps.

    Same character-overlap weights, same token walk, same denominator; the difference is only that
    the result is NOT clamped to >= 0 and NOT renormalised to sum 1 (which is what a signed input
    forbids).  Returns (projected[K_student], stats).  The frozen file is not modified.
    """
    out = torch.zeros(len(student_offsets), dtype=torch.float32)
    j = 0
    missing = 0
    for i, (t0, t1) in enumerate(teacher_offsets):
        if i >= values.numel():
            break
        if t1 <= t0:
            continue
        while j < len(student_offsets) and student_offsets[j][1] <= t0:
            j += 1
        k = j
        denom = max(1, t1 - t0)
        hit = False
        while k < len(student_offsets):
            s0, s1 = student_offsets[k]
            if s0 >= t1:
                break
            ov = max(0, min(t1, s1) - max(t0, s0))
            if ov > 0:
                out[k] += float(values[i]) * (ov / denom)
                hit = True
            k += 1
        if not hit:
            missing += 1
    stats = {"teacher_tokens": len(teacher_offsets), "student_tokens": len(student_offsets),
             "teacher_tokens_mapping_to_nothing": missing,
             "student_tokens_receiving_mass": int((out != 0).sum()),
             "operator": "frozen char-overlap weights, non-negativity and renormalisation "
                         "dropped (approved projection path); frozen file untouched",
             "eps": eps}
    return out, stats


def coverage_row(values: torch.Tensor, t_off, s_off):
    """Row sums of the overlap matrix: mass each teacher token spreads when values are all ones."""
    ones = torch.ones(len(t_off), dtype=torch.float32)
    row, _ = signed_overlap(ones, t_off, s_off)
    return {"covered_student_tokens": int((row > 0).sum()),
            "total_student_tokens": int(row.numel()),
            "coverage_fraction": float((row > 0).float().mean()),
            "sum_of_operator": float(row.sum()),
            "identity_max_abs_dev": float((row - 1.0).abs().max()),
            "student_tokens_with_mass_gt_1": int((row > 1.0 + 1e-6).sum()),
            "student_tokens_with_mass_lt_1": int((row < 1.0 - 1e-6).sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--teacher-scores", required=True)
    ap.add_argument("--positions-set", required=True)
    ap.add_argument("--set-name", default="all_population")
    ap.add_argument("--k4-dir", required=True)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--source-lock", required=True)
    ap.add_argument("--teacher-model", required=True)
    ap.add_argument("--student-model", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--validation-ledger", required=True)
    ap.add_argument("--validation-prior-cache", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard-lo", type=int, default=0)
    ap.add_argument("--shard-hi", type=int, default=0)
    ap.add_argument("--shard-name", default="s0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    H = load_harness(args.harness_root)
    backend_mod, runner_mod, steering = load_modules(H, args.source_dir, args.source_lock)
    sys.path.insert(0, str(Path(args.k4_dir).resolve()))
    sys.path.insert(0, str(args.source_lock))
    import run_hotpot_attn_transfer as RHT
    if not Path(args.teacher_scores).exists():
        raise SystemExit(f"missing teacher scores: {args.teacher_scores}")
    raw = torch.load(args.teacher_scores, map_location="cpu", weights_only=False)

    # ---- byte-identical diagnostic source diffs to the Teacher script / Track A
    src = H.STEERING_SOURCE
    if src.count(DETACH_LINE) != 1:
        raise RuntimeError("steering source no longer contains the expected capture line")
    diag_src = src.replace(DETACH_LINE, GRAPH_LINES)
    if diag_src.count(MIX_CALL) != 1:
        raise RuntimeError("steering source no longer contains the expected mix call")
    delta_src = diag_src.replace(MIX_CALL, DELTA_CALL) + DELTA_HELPER

    def build(text):
        return H.build_steering_context(text, {"contextlib": __import__("contextlib"),
                                               "math": __import__("math"),
                                               "types": __import__("types"),
                                               "torch": torch, "F": F,
                                               "_STEERING_STATE_CLS": H.SteeringState})
    steering_diag = build(diag_src)
    steering_delta = build(delta_src)

    backend = build_backend(backend_mod, args, steering)
    eps = backend.eps
    vocab = int(backend.student.config.vocab_size)
    layers = sorted(backend.student_layers)
    backbone = backend.student.model
    deep = backbone.layers[len(backbone.layers) - len(layers):]
    apply_mix = getattr(steering_diag, "__wrapped__", steering_diag).__globals__["_apply_mix"]

    # ---- frozen Adapter head-mixing WEIGHTS (reused; probability steps dropped)
    # the deployed adapter wraps the locked AttentionPriorAdapter (entropy-conditioned gate only,
    # the prior path is the locked one), so the head-mixing weights live on `.base`
    adapter_obj = getattr(backend.adapter, "base", backend.adapter)
    mix_logits = adapter_obj.mix_logits.detach().float().cpu()
    scfg = backend.adapter.config
    adapter_provenance = {"adapter_class": type(backend.adapter).__name__,
                          "mixing_class": type(adapter_obj).__name__,
                          "has_base_wrapper": bool(hasattr(backend.adapter, "base")),
                          "note": "the entropy-conditioned wrapper only changes how alpha is "
                                  "computed; the prior path (and therefore the head-mixing "
                                  "weights) is the locked AttentionPriorAdapter"}
    head_w = F.softmax(mix_logits.reshape(scfg.num_student_layers, scfg.num_student_heads, -1),
                       dim=-1)
    head_w_np = head_w.numpy()

    rows_src = runner_mod.load_rows(Path(args.validation_ledger), Path(args.dataset_path))
    by_id = {r["canonical_source_id"]: r for r in rows_src}
    priors = torch.load(Path(args.validation_prior_cache), map_location="cpu",
                        weights_only=True)

    pop = json.loads(Path(args.population).read_text(encoding="utf-8"))
    if args.set_name == "all_population":
        sel = list(pop["rows"])
    else:
        sel, _spec = select_positions(pop["rows"], args.set_name, args.positions_set)
    if args.limit:
        sel = sel[:args.limit]
    if args.shard_hi:
        sel = sel[args.shard_lo:args.shard_hi]
    elif args.shard_lo:
        sel = sel[args.shard_lo:]

    body = out / f"td_alignment_rows_{args.shard_name}.jsonl"
    grads_path = out / f"td_student_grads_{args.shard_name}.pt"
    stored: dict = {}
    records: list[dict] = []
    t_shard = time.time()

    for n, r in enumerate(sel):
        sid = r["sample_id"]
        t = int(r["position"])
        key = f"{sid}|{t}"
        ids = [int(x) for x in r["prefix_token_ids"]]
        L = len(ids)
        pt = L - t
        sha = prefix_sha(ids[:pt], ids[pt:])
        if sha != r["prefix_sha256"]:
            raise SystemExit(f"prefix identity failed for {key}")
        if key not in raw:
            raise SystemExit(f"teacher score missing for {key}")
        entry = raw[key]
        score = entry["score"].float()                      # [4,28,K]
        if list(score.shape[-1:]) != [L]:
            raise SystemExit(f"score key length {score.shape[-1]} != prefix_len {L} for {key}")
        y_T, y_S = int(r["y_T"]), int(r["y_S"])
        if (int(entry["y_T"]), int(entry["y_S"])) != (y_T, y_S):
            raise SystemExit(f"y identity mismatch for {key}")
        froz = r["student_local_oracle_identity"]
        budgets = {int(k): [float(x) for x in v]
                   for k, v in froz["per_layer_budget_l1"].items()}
        g_abs = {int(k): [float(x) for x in v]
                 for k, v in froz["per_layer_g_abs_sum"].items()}

        row = by_id[sid]
        target = backend.prompt_target(row["prompt"], row["answer"])
        prior = priors[sid]
        with torch.no_grad():
            rs = backend._student_rows(target)
            mask = torch.tensor(target.attention_mask, device=backend.device)
            mapped, alpha, _ = backend.adapter(prior.to(backend.device), rs, mask)
            null_map, null_alpha, _ = backend.adapter(prior.roll(1, dims=-1).to(backend.device),
                                                      rs, mask)
        lm, la = maps_for(backend, mapped, alpha)
        nm, na = maps_for(backend, null_map, null_alpha)

        ids_t = torch.tensor([ids], device=backend.device)
        amask = torch.ones_like(ids_t)
        t0 = time.time()

        # ---- pass A: frozen full forward (no graph)
        cap: dict = {}

        def pre_hook(module, hargs, hkwargs):
            cap["h"] = hargs[0].detach().clone()
            for k in ("attention_mask", "position_ids", "cache_position",
                      "position_embeddings"):
                cap[k] = hkwargs.get(k)
            return None

        handle = deep[0].register_forward_pre_hook(pre_hook, with_kwargs=True)
        try:
            with torch.no_grad():
                oA = backend.student(input_ids=ids_t, attention_mask=amask, use_cache=False,
                                     output_attentions=False)
                z_full = oA.logits[0, -1, :vocab].float().cpu()
        finally:
            handle.remove()
        del oA

        # ---- pass B: only the four patched layers carry the graph (Track A mechanism, verbatim)
        h = cap["h"].requires_grad_(True)
        with torch.enable_grad():
            with steering_diag(backend.student, {}, None, eps, selected_layers=set(layers),
                               phase="TD_STUDENT_GRAD", capture_rows={}) as st:
                for layer in deep:
                    h = layer(h, attention_mask=cap["attention_mask"],
                              position_ids=cap["position_ids"], past_key_values=None,
                              use_cache=False, cache_position=cap["cache_position"],
                              position_embeddings=cap["position_embeddings"])
                h = backbone.norm(h)
                logits = backend.student.lm_head(h)[0, -1, :vocab].float()
            z_b = logits.detach().cpu()
            M = logits[y_T] - logits[y_S]
            M.backward()
            full_rows = [st.capture_rows[L_] for L_ in layers]
            grad_connected = [x.grad is not None for x in full_rows]
            p_base = [x[0, :, 0, :].detach().float().cpu() for x in full_rows]
            grads = [torch.zeros_like(p_base[i]) if x.grad is None
                     else x.grad[0, :, 0, :].detach().float().cpu()
                     for i, x in enumerate(full_rows)]
        M_base = float(M.detach())
        recon = float((z_b - z_full).abs().max())
        logp_base = float(F.log_softmax(z_full, dim=-1)[y_T])
        del h, logits, z_b, full_rows, M, st, cap
        torch.cuda.empty_cache()
        if not all(grad_connected):
            raise SystemExit(f"student gradient not connected at {key}")

        # ---- verification hooks against the frozen Track A evidence
        hook = {"g_abs_sum_max_abs_diff": 0.0, "budget_max_abs_diff": 0.0,
                "V_teacher_abs_diff": 0.0, "V_null_abs_diff": 0.0,
                "V_oracle_abs_diff": 0.0, "M_base_abs_diff": abs(M_base - float(r["base_logits_identity"]["M_base_track_a"]))}
        per_layer = {}
        d_T_all, d_O_all, d_N_all = {}, {}, {}
        V_T = V_N = V_O = 0.0
        for i, lay in enumerate(layers):
            p, g = p_base[i], grads[i]
            K = int(p.shape[-1])
            fake = p.unsqueeze(0).unsqueeze(2)

            def mix_with(maps, alphas):
                s = type("S", (), {})()
                s.layer_to_prior, s.layer_to_alpha = maps, alphas
                s.eps, s.mix_log = eps, []
                out_mixed = apply_mix(s, lay, fake.clone(), K)
                return out_mixed[0, :, 0, :].float().cpu()

            d_T = mix_with(lm, la) - p
            d_N = mix_with(nm, na) - p
            recomputed_budget = [float(x) for x in d_T.abs().sum(dim=-1)]
            d_O = torch.stack([greedy_transfer(g[hh], p[hh], recomputed_budget[hh])
                               for hh in range(p.shape[0])])
            V_T += float((g * d_T).sum())
            V_N += float((g * d_N).sum())
            V_O += float((g * d_O).sum())
            d_T_all[lay], d_N_all[lay], d_O_all[lay] = d_T, d_N, d_O
            gsum = [float(g[hh].abs().sum()) for hh in range(p.shape[0])]
            hook["g_abs_sum_max_abs_diff"] = max(
                hook["g_abs_sum_max_abs_diff"],
                max(abs(a - b) for a, b in zip(gsum, g_abs[lay])))
            hook["budget_max_abs_diff"] = max(
                hook["budget_max_abs_diff"],
                max(abs(a - b) for a, b in zip(recomputed_budget, budgets[lay])))
            per_layer[str(lay)] = {"K": K,
                                   "recomputed_budget_l1_per_head": [round(x, 8)
                                                                     for x in recomputed_budget],
                                   "frozen_budget_l1_per_head": [round(x, 8)
                                                                 for x in budgets[lay]],
                                   "g_abs_sum_per_head": [round(x, 6) for x in gsum],
                                   "dO_l1_per_head": [round(float(d_O[hh].abs().sum()), 8)
                                                      for hh in range(p.shape[0])]}
        hook["V_teacher_abs_diff"] = abs(V_T - float(froz["V_teacher"]))
        hook["V_null_abs_diff"] = abs(V_N - float(froz["V_null"]))
        hook["V_oracle_abs_diff"] = abs(V_O - float(froz["V_oracle"]))
        hook["status"] = "PASS" if (hook["g_abs_sum_max_abs_diff"] <= TOL_G
                                    and hook["budget_max_abs_diff"] <= TOL_G
                                    and hook["V_teacher_abs_diff"] <= TOL_V
                                    and hook["V_null_abs_diff"] <= TOL_V
                                    and hook["V_oracle_abs_diff"] <= TOL_V
                                    and hook["M_base_abs_diff"] <= TOL_MARGIN) else "FAIL"
        if hook["status"] != "PASS":
            raise SystemExit(f"frozen-evidence recomputation hook FAILED at {key}: "
                             f"{json.dumps(hook, default=str)}")

        # ---- (2) projection: token-space weights -> head-mixing weights
        proj_stats = {}
        flat = score.reshape(score.shape[0] * score.shape[1], L)
        if score.shape[0] != int(scfg.num_teacher_layers) or \
                flat.shape[0] != int(scfg.num_teacher_layers) * int(scfg.num_teacher_heads):
            raise SystemExit(f"teacher score does not match the frozen Adapter's teacher layout "
                             f"at {key}: {list(score.shape)} vs "
                             f"{(scfg.num_teacher_layers, scfg.num_teacher_heads)}")
        # The teacher is fed the SAME frozen token ids as the student (the parent diagnostic's
        # convention), so teacher and student key coordinates coincide.  That is verified on the
        # frozen operator itself with a real probability row, not assumed.
        ident = [(i, i + 1) for i in range(L)]
        probe_row = p_base[0][0].float().cpu()
        frozen_fallback = None
        try:
            f_out = RHT.project_teacher_prior_to_student(probe_row, ident, ident, eps)
            f_signed = RHT.project_teacher_prior_to_student(score[0][0].float().cpu(),
                                                            ident, ident, eps)
            frozen_fallback = {
                "non_negative_input_max_abs_dev_vs_identity": float((f_out - probe_row).abs().max()),
                "signed_input_total": float(score[0][0].float().sum()),
                "signed_input_renormalised_total": float((score[0][0].float().sum())),
                "signed_input_hit_uniform_fallback": bool(
                    abs(float(f_signed.sum()) - 1.0) < 1e-6
                    and float((f_signed - f_signed.mean()).abs().max()) < 1e-6),
                "note": "the frozen operator divides by the input total; a signed row whose total "
                        "is ~0 therefore lands in its uniform fallback branch. This is the "
                        "concrete reason the Supervisor-approved path reuses the operator's "
                        "WEIGHTS and drops the probability steps.",
            }
        except Exception as exc:                                     # pragma: no cover
            frozen_fallback = {"error": f"{type(exc).__name__}: {exc}"}
        my_out, my_stats = signed_overlap(probe_row, ident, ident, eps)
        proj_stats["token_space"] = coverage_row(torch.ones(L), ident, ident)
        proj_stats["token_space_update"] = {
            "operator_identity_max_abs_dev": float((my_out - probe_row).abs().max()),
            "stats": my_stats,
            "frozen_operator_comparison": frozen_fallback,
        }
        s_mix = torch.einsum("sgx,xk->sgk", head_w, flat)              # [4,12,K]
        w_np = head_w_np.reshape(-1, head_w_np.shape[-1])
        struct = []
        for li in range(s_mix.shape[0]):
            for hh in range(s_mix.shape[1]):
                w = w_np[li * s_mix.shape[1] + hh]
                dom = int(w.argmax())
                trow = flat[dom]
                sp, sn = s_mix[li, hh], trow
                nz = (sp != 0) | (sn != 0)
                agree = float(((sp[nz] > 0) == (sn[nz] > 0)).float().mean()) if bool(nz.any()) else None
                k = min(16, L)
                tp = set(int(x) for x in torch.topk(sp, k).indices)
                tn = set(int(x) for x in torch.topk(sn, k).indices)
                weighted_l1 = float(sum(float(w[x]) * float(flat[x].abs().sum())
                                       for x in range(flat.shape[0])))
                l1 = float(sp.abs().sum())
                struct.append({"layer": int(layers[li]), "student_head": hh,
                               "dominant_teacher_flat_index": dom,
                               "dominant_teacher_weight": round(float(w[dom]), 6),
                               "mix_weight_entropy": round(float(-(torch.tensor(w)
                                                                  * torch.log(torch.tensor(w)
                                                                              + 1e-12)).sum()), 6),
                               "sign_agreement_with_dominant": agree,
                               "topk_jaccard_with_dominant": (len(tp & tn) / len(tp | tn)
                                                              if tp | tn else None),
                               "cancellation": (1.0 - l1 / weighted_l1) if weighted_l1 > 0 else None,
                               "projected_l1": l1, "weighted_teacher_l1": weighted_l1})
        proj_stats["head_mixing"] = {
            "operator": "frozen AttentionPriorAdapter.mix_logits softmax weights; normalize_rows, "
                        "residual mix and final softmax dropped (approved projection path)",
            "adapter_provenance": adapter_provenance,
            "shape": list(head_w.shape), "per_head": struct,
            "mean_cancellation": (sum(x["cancellation"] for x in struct if x["cancellation"]
                                      is not None) / max(1, sum(1 for x in struct
                                                               if x["cancellation"] is not None))),
            "mean_sign_agreement": (sum(x["sign_agreement_with_dominant"] for x in struct
                                        if x["sign_agreement_with_dominant"] is not None)
                                    / max(1, sum(1 for x in struct
                                                 if x["sign_agreement_with_dominant"] is not None))),
        }
        # ---- d_TD: same greedy mass transfer, same per-head L1 budget, signed score
        d_TD = {}
        rescaled = {}
        V_TD = 0.0
        per_layer_td = {}
        for i, lay in enumerate(layers):
            p = p_base[i]
            resc = []
            d_rows = []
            for hh in range(p.shape[0]):
                b = float(budgets[lay][hh])
                l1 = float(s_mix[i, hh].abs().sum())
                sc = (b / l1) if l1 > 1e-12 else 0.0
                resc.append(sc)
                d_rows.append(greedy_transfer(s_mix[i, hh] * sc, p[hh], b))
            d = torch.stack(d_rows)
            d_TD[lay] = d
            rescaled[lay] = resc
            V_TD += float((grads[i] * d).sum())
            per_layer_td[str(lay)] = {
                "rescale_factor_per_head": [round(x, 8) for x in resc],
                "s_mix_l1_per_head": [round(float(s_mix[i, hh].abs().sum()), 8)
                                      for hh in range(p.shape[0])],
                "dTD_l1_per_head": [round(float(d[hh].abs().sum()), 8)
                                    for hh in range(p.shape[0])],
                "frozen_budget_l1_per_head": [round(float(budgets[lay][hh]), 8)
                                              for hh in range(p.shape[0])],
                "V_TD_per_head": [round(float((grads[i][hh] * d[hh]).sum()), 8)
                                  for hh in range(p.shape[0])],
                "V_Oracle_per_head_frozen": [
                    round(float((grads[i][hh] * d_O_all[lay][hh]).sum()), 8)
                    for hh in range(p.shape[0])],
            }

        # ---- (3) actual same-budget injection through the real attention path
        with torch.no_grad():
            with steering_delta(backend.student, {L_: d_TD[L_] for L_ in layers}, None, eps,
                               selected_layers=set(layers), phase="TD_STUDENT_ORACLE") as st2:
                o2 = backend.student(input_ids=ids_t, attention_mask=amask, use_cache=True,
                                     output_attentions=False)
            z_td = o2.logits[0, -1, :vocab].float().cpu()
        M_td = float(z_td[y_T]) - float(z_td[y_S])
        logp_td = float(F.log_softmax(z_td, dim=-1)[y_T])
        top1_td = int(torch.argmax(z_td).item())
        n_inter = len(st2.mix_log)
        del o2, z_td, st2
        torch.cuda.empty_cache()

        rec = {
            "canonical_source_id": sid, "source_index": r["source_index"],
            "condition": "decision_position", "arm": "decision_position",
            "position": t, "group": r["group"], "prefix_len": L, "prompt_len": pt,
            "prefix_sha256": sha,
            "y_target": y_T, "y_base": y_S,
            "M_base": M_base, "M_base_frozen_track_a": float(
                r["base_logits_identity"]["M_base_track_a"]),
            "logp_base_yT": logp_base,
            "V_teacher": V_T, "V_null": V_N, "V_oracle": V_O,
            "V_TD": V_TD,
            "teacher_efficiency": (V_T / V_O) if abs(V_O) > 1e-12 else None,
            "null_efficiency": (V_N / V_O) if abs(V_O) > 1e-12 else None,
            "TD_efficiency": (V_TD / V_O) if abs(V_O) > 1e-12 else None,
            "teacher_l1_total": float(sum(d_T_all[L_].abs().sum() for L_ in layers)),
            "oracle_l1_total": float(sum(d_O_all[L_].abs().sum() for L_ in layers)),
            "TD_l1_total": float(sum(d_TD[L_].abs().sum() for L_ in layers)),
            "M_under_TD_actual": M_td, "dM_TD_actual": M_td - M_base,
            "logp_yT_under_TD_actual": logp_td, "dlogp_yT_TD_actual": logp_td - logp_base,
            "top1_under_TD": top1_td, "top1_flip_under_TD": bool(top1_td != y_S),
            "target_token_is_top1_under_TD": bool(top1_td == y_T),
            "margin_closure_TD": ((M_td - M_base) / (-M_base)) if M_base < 0 else None,
            "TD_interventions": n_inter,
            "frozen_evidence_hook": hook,
            "projection": proj_stats,
            "per_layer_TD": per_layer_td,
            "per_layer_identity": per_layer,
            "reconstruction_vs_full_forward_max_abs_logit_delta": recon,
            "grad_connected_per_layer": dict(zip([str(x) for x in layers], grad_connected)),
            "method": "SIGNED_TEACHER_ATTRIBUTION_INJECTED_AT_MATCHED_BUDGET_NON_DEPLOYABLE",
            "run_config_hash": cfg["config_hash"],
            "checkpoint_sha256": cfg["checkpoint_sha256"],
            "shard": args.shard_name,
        }
        records.append(rec)
        stored[key] = {"g": torch.stack(grads, 0).to(torch.float32),
                       "p_base": torch.stack(p_base, 0).to(torch.float32),
                       "d_TD": torch.stack([d_TD[L_] for L_ in layers], 0).to(torch.float32),
                       "d_O": torch.stack([d_O_all[L_] for L_ in layers], 0).to(torch.float32),
                       "layers": list(layers), "y_T": y_T, "y_S": y_S,
                       "prefix_sha256": sha, "prefix_len": L, "position": t}
        if (n + 1) % 8 == 0 or n == len(sel) - 1:
            torch.save(stored, grads_path)
        del p_base, grads, d_T_all, d_N_all, d_O_all, d_TD, s_mix
        torch.cuda.empty_cache()
        H.write_result_envelope(
            body, task_id=TASK_ID, stage="teacher_to_student_alignment", records=records,
            arms=["decision_position"], sample_ids=sorted({x["canonical_source_id"]
                                                           for x in records}),
            config_hash=cfg["config_hash"], checkpoint_sha256=cfg["checkpoint_sha256"])
        print(json.dumps({"shard": args.shard_name, "n": n, "src": r["source_index"], "pos": t,
                          "K": L, "M": round(M_base, 4), "dM_TD": round(rec["dM_TD_actual"], 5),
                          "V_TD": round(V_TD, 5), "V_O": round(V_O, 5),
                          "eff": (round(V_TD / V_O, 5) if abs(V_O) > 1e-12 else None),
                          "flip": rec["top1_flip_under_TD"],
                          "hook": hook["status"],
                          "sec": round(time.time() - t0, 1)}), flush=True)

    H.write_result_envelope(
        body, task_id=TASK_ID, stage="teacher_to_student_alignment", records=records,
        arms=["decision_position"], sample_ids=sorted({x["canonical_source_id"]
                                                       for x in records}),
        config_hash=cfg["config_hash"], checkpoint_sha256=cfg["checkpoint_sha256"])
    if stored:
        torch.save(stored, grads_path)
    meta = {"shard": args.shard_name, "positions": len(records),
            "layers": layers, "student_heads": int(scfg.num_student_heads),
            "teacher_heads": int(scfg.num_teacher_heads),
            "hook_all_pass": all(x["frozen_evidence_hook"]["status"] == "PASS" for x in records),
            "grads_path": str(grads_path),
            "grads_sha256": sha256_file(grads_path) if grads_path.exists() else None,
            "wall_seconds": round(time.time() - t_shard, 1), "body": str(body),
            "body_sha256": sha256_file(body),
            "verify": H.verify_result(body, expected_config_hash=cfg["config_hash"],
                                      expected_checkpoint_sha256=cfg["checkpoint_sha256"])}
    H.atomic_write_text(out / f"td_alignment_rows_{args.shard_name}_meta.json",
                        json.dumps(meta, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({"shard": args.shard_name, "done": True, "positions": len(records),
                      "verify": meta["verify"]["status"], "hooks": meta["hook_all_pass"],
                      "wall_seconds": meta["wall_seconds"]}, ensure_ascii=False), flush=True)
    return 0 if meta["verify"]["status"] == "PASS" and meta["hook_all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
