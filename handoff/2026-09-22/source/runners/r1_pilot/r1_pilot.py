"""RESCUE-R1-VALUE-NORM-DECISION-ALIGNMENT-PILOT-001.

One method variable changes: the teacher-side token score feeding the frozen projection
    R0:  score_j = attention_j
    R1:  score_j = attention_j × ||V_j||        (normalised per layer/head over keys)
Everything downstream is the frozen mechanism (projection, Adapter, alpha, layers, full-window
intervention, mixing equation, dtype, context, tokenizer pair, checkpoint).

For each of the 67 S≠T decision positions (identity-matched to the mechanism sprint by
sample_id + position + prefix_sha256 + y_T + y_S) this measures, in ONE common framing
(single full-prefix forward, intervention applied at the last query row — the same framing that
produced the Track-A Oracle baseline):

    BASE  no steering
    R0    frozen raw-attention prior
    R1    attention × ||V|| prior
    NULL  frozen Position-Alignment-Null

plus the Track-A local gradient g at the same rows, so V_arm = g · d_arm and the paired
R1 − R0 / R1 − Null contrasts and the Oracle efficiency comparison can be computed.

No training, no hyperparameter change, no QA run, no K4 rerun, no Oracle rerun.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dr_common import (build_backend, cosine, load_config, load_harness, load_modules,  # noqa: E402
                       logprob_of, maps_for, margin_of, sha256_file)

DETACH_LINE = "state.capture_rows[layer_index] = last_probs[0, :, 0, :].detach().float()"
GRAPH_LINES = ("last_probs.retain_grad()\n"
               "                state.capture_rows[layer_index] = last_probs")
MIX_CALL = "last_probs = _apply_mix(state, layer_index, last_probs, key_len)"
DELTA_CALL = "last_probs = _apply_delta(state, layer_index, last_probs, key_len)"

DELTA_HELPER = '''

def _apply_delta(state, layer_index, attn_probs, key_len):
    """Diagnostic-only: apply a FIXED per-layer delta (the same frozen mixing semantics)."""
    delta = state.layer_to_prior[layer_index].to(attn_probs.device, dtype=torch.float32)
    row = attn_probs[:, :, -1, :].float()
    new = (row + delta).clamp_min(0.0)
    new = new / new.sum(dim=-1, keepdim=True).clamp_min(state.eps)
    state.mix_log.append({"layer": layer_index, "key_len": int(key_len), "diagnostic": True})
    return torch.cat((attn_probs[:, :, :-1, :], new.to(attn_probs.dtype).unsqueeze(2)), dim=2)

'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--k4-dir", required=True)
    ap.add_argument("--dr-rows", required=True)
    ap.add_argument("--ta-rows", required=True)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--source-lock", required=True)
    ap.add_argument("--teacher-model", required=True)
    ap.add_argument("--student-model", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--validation-ledger", required=True)
    ap.add_argument("--validation-prior-cache", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--target-max-memory", default="8GiB")
    ap.add_argument("--target-dtype", default="bfloat16")
    ap.add_argument("--shard-lo", type=int, default=0)
    ap.add_argument("--shard-hi", type=int, default=0)
    ap.add_argument("--shard-name", default="r1")
    ap.add_argument("--limit-samples", type=int, default=0)
    ap.add_argument("--save-r1-priors", default="")
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
    from transformers import AutoTokenizer
    from transformers.models.qwen2.modeling_qwen2 import repeat_kv
    from steering import qwen2_teacher_row_observer

    src = H.STEERING_SOURCE
    if src.count(DETACH_LINE) != 1 or src.count(MIX_CALL) != 1:
        raise RuntimeError("frozen steering source changed shape; refusing to instrument")
    diag_src = src.replace(DETACH_LINE, GRAPH_LINES)
    delta_src = diag_src.replace(MIX_CALL, DELTA_CALL) + DELTA_HELPER

    def build(text):
        return H.build_steering_context(text, {"contextlib": __import__("contextlib"),
                                               "math": __import__("math"),
                                               "types": __import__("types"),
                                               "torch": torch, "F": F,
                                               "_STEERING_STATE_CLS": H.SteeringState})
    steering_diag = build(diag_src)
    steering_delta = build(delta_src)
    apply_mix = getattr(steering_diag, "__wrapped__", steering_diag).__globals__["_apply_mix"]
    fit_prior = getattr(steering.qwen2_adapter_steering, "__wrapped__",
                        steering.qwen2_adapter_steering).__globals__["_fit_prior"]

    backend = build_backend(backend_mod, args, steering)
    layers = sorted(backend.student_layers)
    eps = backend.eps
    vocab = int(backend.student.config.vocab_size)
    backbone = backend.student.model
    deep = backbone.layers[len(backbone.layers) - len(layers):]

    model, device, dmap = __import__("real_k4_runner").build_target(args)
    teacher_layers = list(RHT.selected_layer_indices(int(model.config.num_hidden_layers),
                                                     "last_4_layers"))
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_model, local_files_only=True,
                                                use_fast=True)
    student_tok = backend.student_tokenizer
    prior_cache = torch.load(Path(args.validation_prior_cache), map_location="cpu",
                             weights_only=True)

    def v_hook(store):
        def hook(module, inputs, kwargs, output):
            hs = kwargs.get("hidden_states")
            if hs is None:
                hs = inputs[0]
            hidden_shape = (*hs.shape[:-1], -1, module.head_dim)
            v = module.v_proj(hs).view(hidden_shape).transpose(1, 2)
            v = repeat_kv(v, module.num_key_value_groups)
            store[int(module.layer_idx)] = v[0].float().norm(dim=-1).detach().cpu()
            return None
        return hook

    # ---- identity-matched population -------------------------------------
    dr = {}
    for line in Path(args.dr_rows).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if "__result_header__" not in r:
                dr[(r["canonical_source_id"], int(r["position"]))] = r
    ta = {}
    for line in Path(args.ta_rows).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if "__result_header__" not in r and "M_base" in r:
                ta[(r["canonical_source_id"], int(r["position"]))] = r
    pop = [k for k, r in dr.items() if r["group"] == "S!=T"]
    pop.sort(key=lambda k: (int(dr[k]["source_index"]), k[1]))
    identity = []
    for k in pop:
        d, t = dr[k], ta.get(k)

        ok = t is not None
        if ok:
            ok = (t["prefix_sha256"] == d["prefix_sha256"] and
                  int(t["y_target"]) == int(d["y_target"]) and
                  int(t["y_base"]) == int(d["y_base"]))
        identity.append({"local": k, "matched": bool(ok)})
    mismatched = [x for x in identity if not x["matched"]]
    if mismatched:
        raise RuntimeError(f"Track-A identity mismatches: {mismatched[:5]}")
    if args.limit_samples > 0:
        pop = pop[:args.limit_samples]

    # build teacher attn  ||V|| priors and instrument the Student
    r1_priors = {}
    records = []
    t0 = time.time()
    for n, k in enumerate(pop):
        d = dr[k]
        sid, t = k
        row = {r ["canonical_source_id"]: r for r in runner_mod.load_rows(
            Path(args.validation_ledger), Path(args.dataset_path))}[sid]
        prompt_ids = backend.student_tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        prefix = prompt_ids + [int(x) for x in d["prefix_generated_tokens"]]
        ids = torch.tensor([prefix], device=backend.device, dtype=torch.long)
        amask = torch.ones_like(ids)
        y_T, y_S = int(d["y_target"]), int(d["y_base"])
        key = f"{sid}|{t}"

        # frozen student rows, R0 and null priors
        t_target = backend.prompt_target(row["prompt"], row["answer"])
        with torch.no_grad():
            rows_s = backend._student_rows(t_target)
            p = prior_cache[sid].to(backend.device)
            mapped, alpha, _ = backend.adapter(p, rows_s,
                                         torch.tensor(t_target.attention_mask, device=backend.device))
            nm, na, _ = backend.adapter(p.roll(1, dims=-1), rows_s,
                                torch.tensor(t_target.attention_mask, device=backend.device))

        # R1 prior from Teacher attention × ||V||, then the unchanged frozen projection + Adapter
        vnorms = {}
        hooks = []
        for lyr in model.model.layers:
            if int(lyr.self_attn.layer_idx) in teacher_layers:
                hooks.append(lyr.self_attn.register_forward_hook(v_hook(vnorms), with_kwargs=True))
        t_text = backend.student_tokenizer.decode(prefix, skip_special_tokens=False)
        with qwen2_teacher_row_observer(model, teacher_layers) as t_cap:
            with torch.inference_mode():
                enc = teacher_tok(t_text, return_tensors="pt", add_special_tokens=False)
                enc = {kk: vv.to(device) for kk, vv in enc.items()}
                _out = model(**enc, use_cache=False, output_attentions=False)
        for h in hooks:
            h.remove()
        if set(v_horms) != set(teacher_layers):
            raise RuntimeError(f"missing vnorm layers {set(teacher_layers) - set(v_horms)}")
        teacher_explicit = []
        student_offsets = RHT._tokenize_with_offsets(student_tok, t_text)[2]
        student_len = len(student_offsets)
        for L in teacher_layers:
            pre_t = t_cap["attn"][L].to(torch.float32)
            vm = vnorms[L].to(torch.float32)
            if pre_t.shape != vm.shape:
                raise RuntimeError(f"vnorm shape {vm.shape} != attn shape {pre_t.shape}")
            raw_r1 = pre_t * vm
            # The Teacher observer row is already projected onto Student keys; vnorm
            # is on the Teacher support, so project the reweighted row explicitly.
            # No new projection rule is introduced — this uses the same frozen code.
            # The observer does not expose the Teacher offsets, so compute them once as the
            # frozen projection does.
            _, t_offsets, _ = RHT._tokenize_with_offsets(teacher_tok, t_text)
            projected = RHT.project_teacher_heads_to_student(
                raw_r1, t_offsets, student_offsets, backend.eps[form())
            projected = fit_prior(projected, student_len, backend.eps)
            teacher_explicit.append(projected)
        projected_r1 = torch.stack(teacher_explicit, dim=0).to(backend.device)
        with torch.no_grad():
            m1, a1, _ = backend.adapter(projected_r1, rows_s,
                                       torch.tensor(t_target.attention_mask, device=backend.device))
        save_key = f"{sid}|{t}"
        r1_priors[save_key] = m1.detach().cpu()

        # single Student base forward + captured deep layer inputs
        captured = {}
        hooks = []
        for idx in range(len(deep)):
            def mk(i):
                return lambda _m, inps: captured.setdefault(i, inps)
            hooks.append(deep[idx].register_forward_pre_hook(mkh(idx))
        with torch.no_grad():
            obj = backend.student(input_ids=ids, attention_mask=amask, use_cache=False,
                                  output_attentions=False)
            z = obj.logits[0, -1, :vocab].float().cpu()
        for h in hooks:
            h.remove()
        del obj, z

        grad_state = steering_diag.__globals_["_STEERING_STATE_CLS"](
            layer_to_prior={l: maps_for(backend, mapped.detach())[l].to(backend.device)
                         for l in layers},
            layer_to_alpha={l: alpha[i].to(backend.device) for i, l in enumerate(layers)},
            eps=eps, phase="R1_GRAD_LOCAL", selected_layers=set(layers))
        grad_state.capture_rows = {}
        getattr(steering_diag, mname).__set__(backend.student, grad_state)
        x = captured[0][0].detach()
        args_dest = captured[0][1:]
        hidden = deep[0].input_layernorm(x)
        for i, layer in enumerate(deep):
            hidden = layer(hidden, *args_dest)
            idx = len(backbone.layers) - len(layers) + i
            grad_state.capture_rows[idx].retain_grad()
        hidden = backbone.norm(hidden)
        logits_grad = backend.student.lm_head(hidden)[0, -1, :vocab].float()
        M = logits_grad[y_T] - logits_grad[y_S]
        M_keep = float(M.detach())
        M.backward()
        grads = [grad_state.capture_rows[L].grad.detach().cpu() for L in layers]
        grad_ok = [grad_state.capture_rows[L].grad is not None for L in layers]
        del grad_state, hidden, logits_grad, M
        torch.cuda.empty_cache()

        # Reconstruction check vs the base logits is the same diagnostic structure as the mechanism sprint.
        stock_was_training = backend.student.training
        backend.student.eval()
        with torch.no_grad():
            x0 = captured[0][0].detach()
            h1 = deep[0].input_layernorm(x0)
            for layer in deep:
                h1 = layer(h1, *args_dest)
            h1 = backbone.norm(h1)
            z_rec = backend.student.lm_head(h1)[0, -1, :vocab].float().cpu()
        recon = float(1z_rec - z_base).abs().max())
        backend.student.train(stock_was_training)

        lm0 = maps_for(backend, mapped.detach())
        la0 = {L: alpha[i].detach() for i, L in enumerate(layerr)}
        lm1 = maps_for(backend, m1.detach())
        la1 = {L: a1[i].detach() for i, L in enumerate(layers)}
        nm = maps_for(backend, nm.detach())
        na = {L: na[i].detach() for i, L in enumerate(layerr)}

        per_layer = {}
        dirs = {}
        for L, g in zip(layers, grads):
            p = diag["pre_rows"][L]
            def mix_with(maps, the_alpha):
                st = steering_diag.__globals_["_STEERING_STATE_CLS"](
                    layer_to_prior={L: maps[L].to(backend.device)},
                    layer_to_alpha={L: the_alpha[L].to(backend.device)}, eps=eps,
                    phase="R1_MIX_PROBE", selected_layers={L}, mix_log=[])
                out_mixed = apply_mix(st, L, p.to(backend.device).unsqueeze(0), int(p.shape[-1]))
                return out_mixed[0, :, 0, :].float().cpu()

                d0 = mix_with(lm0, la0) - p
                d1 = mix_with(lm1, la1) - p
                dn = mix_with(nm, na) - p
                dirs[L] = {"R1": d1, "R0": d0, "NULL": dn}
                per_layer[str(L)] = {
                    "d1_R0_l1": float(d0.abs().sum()), "d1_R1_l1": float(d1.abs().sum()),
                    "d1_Null_l1": float(dn.abs().sum()),
                    "V_R0": float((g * d0).sum()), "V_R1": float((g * d1).sum()),
                    "V_Null": float((g * dn).sum()),
                    "g_abs_sum": float(g.abs().sum()),
                }

            arm_out = {}
            for arm in ("R0", "R1", "NULL"):
                delta = {L: dirs[L][arm] for L in layers}
                with torch.no_grad():
                    with steering_delta(backend.student, delta, None, eps,
                                       selected_layers=set(layers), phase=f"R1_{arm}") as st2:
                        o2 = backend.student(input_ids=ids, attention_mask=amask,
                                             use_cache=True, output_attentions=False)
                    z_a = o2.logits[0, -1, :vocab].float().cpu()
                    n_int = len(st2.mix_log)
                del o2
                torch.cuda.empty_cache()
                y_a = int(torch.argmax(z_a).item())
                dz = z_a - z_base
                arm_out[arm] = {
                    "M": float(z_a[y_T] - z_a[y_S]), "logp_yT": float(F.log_softmax(
                        z_a, dim=-1)[y_T]),
                    "top1": y_a, "flip": bool(y_a != y_S), "flip_to_target": bool(y_a == y_T),
                    "flip_away_from_target": bool(y_a != y_S and y_a != y_T),
                    "dz_l2": float(dz.norm()), "dz_max": float(dz.abs().max()),
                    "cos_to_dtarget": cosine(dz, d_target),
                    "l1_total": sum(float(dirs[L][arm].abs().sum()) for L in layers),
                    "interventions": n_int,
                }

            rec = {
                "canonical_source_id": sid, "source_index": int(d["source_index"]),
                "condition": "decision_position", "arm": "decision_position",
                "position": t, "group": "S!=T", "prefix_len": len(prefix),
                "prefix_sha256": d["prefix_sha256"], "y_target": y_T, "y_base": y_S,
                "M_base": M, "logp_base_yT": logp_base,
                "grad_connected": dict(zip([str(L) for L in layers], grad_ok)),
                "reconstruction_max_abs_logit_delta": recon,
                "target_top1_recomputed_matches_parent": bool(y_T_recomputed == y_T),
                "d_target_definition": "z_Target - z_Base, full vocabulary (parent definition)",
                "framework": "single full-prefix forward, intervention at the last query row "
                             "(same framing as the Track-A Oracle baseline)",
                "arms": arm_out,
                "V_first_order": {a: sum(per_layer[str(L)][f"V_{a}"] for L in layers)for a in ("R0", "R1", "Null")},
                "l1_first_order": {a: sum(per_layer[str(L)][f"d1_{a}_l1"] for L in layers) for a in ("R0", "R1", "Null")},
                "layers": per_layer,
                "track_a": {"present": bool(k in ta),
                            "M_base": None if k not in ta else ta[k]["M_base"],
                           "V_teacher": None if k not in ta else ta[k]["V_teacher"],
                            "V_null": None if k not in ta else ta[k]["V_null"],
                           "V_oracle": None if k not in ta else ta[k]["V_oracle"],
                            "teacher_efficiency": None if k not in ta else ta[k]["teacher_efficiency"],
                            "null_efficiency": None if k not in ta else ta[k]["N_efficiency"],
                           "oracle_l1_total": None if k not in ta else ta[k]["oracle_l1_total"],
                            "prefix_sha256": None if k not in ta else ta[k]["prefix_sha256"]},
                "dr_reference": {"dM_teacher_R0_incremental": d["hdM"],
                                 "dlogp_T_teacher_R0_incremental": d["dlogp_teacher"],
                                 "cos_teacher_R0_incremental": d["cos_dt"],
                                   "dM_null_incremental": d["dM_null"],
                                   "cos_null_incremental": d["cos_null"],
                                 "note": "parent diagnostic framing (prefill + steered decode); reused as an external cross-check, not as the paired baseline"},
                "alpha_R0_mean": float(alpha.mean()), "alpha_R1_mean": float(a1_mean),
                "run_config_hash": cfg["config_hash"], "checkpoint_sha256": cfg["checkpoint_sha256"], "shard": args.shard_name}
            records.append(rec)
            H.write_result_envelope(body, task_id="RESCUE-R1-VALUE-NORM-DECISION-ALIGNMENT-PILOT-001", stage="r1_decision_alignment", records=records, arms=["decision_position"], sample_ids=sorted({r["canonical_source_id"] for r in records}), config_hash=cfg["config_hash"], checkpoint_sha256=cfg["checkpoint_sha256"])
            print(json.dumps({"shard": args.shard_name, "pos": len(records), "src": d["source_index"], "t": t, "dM_R0": round(arm_out["R1"]["M"] - M, 5), "dM_R1": round(arm_out["R1"]["M"] - M, 5), "dM_Null": round(arm_out["NULL"]["M"] - M, 5), "V_R1": round(rec["V_first_order"]["R1"], 5), "flip1": arm_out("R1")["flip_to_target"]}), flush=True)
            del p_base, grads, dirs, st

    H.write_result_envelope(body, task_id="RESCUE-R1-VALUE-NORM-DECISION-ALIGNMENT-PILOT-001", stage="r1_decision_alignment", records=records, arms=["decision_position"], sample_ids=sorted({r["canonical_source_id"] for r in records}), config_hash=cfg["config_hash"], checkpoint_sha256=cfg["checkpoint_sha256"])
    if args.save_r1_priors and r1_priors:
        torch.save(r1_priors, args.save_r1_priors)
    meta = {"shard": args.shard_name, "planned_positions": len(pop), "positions": len(records), "samples": len(samples), "identity_mismatches": len(mismatched), "identity": identity, "teacher_layers": teacher_layers, "target_device_map": dmap, "wall_seconds": round(time.time() - t0, 1), "body": str(body), "body_sha256": sha256_file(body), "verify": H.verify_result(body, expected_config_hash=cfg["config_hash"], expected_checkpoint_sha256=cfg["checkpoint_sha256"])}
    H.atomic_write_text(out / f"r1_shard_{args.shard_name}_meta.json", json.dumps(meta, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({"shard": args.shard_name, "done": True, "positions": len(records), "verify": meta["verify"]["status"], "wall_seconds": meta["wall_seconds"]}, ensure_ascii=False), flush=True)
    return 0 if meta["verify"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
