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
        identity.append({"sample_id": k[0], "position": k[1],
                         "prefix_sha256": d["prefix_sha256"],
                         "y_target": d["y_target"], "y_base": d["y_base"],
                         "track_a_present": t is not None,
                         "track_a_prefix_sha256": None if t is None else t["prefix_sha256"],
                         "identity_match": bool(t is not None
                                                and t["prefix_sha256"] == d["prefix_sha256"]
                                                and int(t["y_target"]) == int(d["y_target"])
                                                and int(t["y_base"]) == int(d["y_base"]))})
    mismatched = [i for i in identity if not i["identity_match"]]
    shard = pop[args.shard_lo:args.shard_hi] if args.shard_hi else pop
    samples = []
    for k in shard:
        if k[0] not in samples:
            samples.append(k[0])
    if args.limit_samples:
        samples = samples[:args.limit_samples]
        shard = [k for k in shard if k[0] in samples]
    print(json.dumps({"shard": args.shard_name, "planned_positions": len(pop),
                      "shard_positions": len(shard), "samples": len(samples),
                      "identity_mismatches": len(mismatched)}), flush=True)

    rows_src = runner_mod.load_rows(Path(args.validation_ledger), Path(args.dataset_path))
    by_id = {r["canonical_source_id"]: r for r in rows_src}

    body = out / f"r1_shard_{args.shard_name}.jsonl"
    records: list[dict] = []
    r1_priors: dict = {}
    t0 = time.time()
    for si, sid in enumerate(samples):
        row = by_id[sid]
        prompt = row["prompt"]
        target = backend.prompt_target(prompt, row["answer"])
        # ---- build the R1 mapped prior for this sample (one frozen teacher pass) ----
        t_enc, t_off = RHT.encode_with_offsets(teacher_tok, prompt, device=None)
        _, s_off = RHT.encode_with_offsets(student_tok, prompt, device=None)
        acaps, vstore = {}, {}
        handles = [model.model.layers[L].self_attn.register_forward_hook(v_hook(vstore),
                                                                        with_kwargs=True)
                   for L in teacher_layers]
        try:
            t_ids = t_enc["input_ids"].to(device)
            with torch.no_grad():
                with qwen2_teacher_row_observer(model.model, teacher_layers, acaps):
                    _ = model(input_ids=t_ids, attention_mask=torch.ones_like(t_ids),
                              use_cache=False)
            del _
        finally:
            for h in handles:
                h.remove()
        torch.cuda.empty_cache()
        r1_layers = []
        for L in teacher_layers:
            a, v = acaps.get(L), vstore.get(L)
            if a is None or v is None:
                continue
            a = a.float()
            n = min(a.shape[-1], v.shape[-1])
            r1 = a[:, :n] * v[:, :n]
            r1 = r1 / r1.sum(dim=-1, keepdim=True).clamp_min(eps)
            r1_layers.append(r1)
        R1 = torch.stack(r1_layers, 0)
        proj1 = torch.stack([RHT.project_teacher_heads_to_student(R1[li], t_off, s_off, eps)
                             for li in range(R1.shape[0])], 0)
        with torch.no_grad():
            rs_rows = backend._student_rows(target)
            smask = torch.tensor(target.attention_mask, device=backend.device)
            prior = prior_cache[sid]
            mapped_R0, alpha, _ = backend.adapter(prior.to(backend.device), rs_rows, smask)
            null_R0, null_alpha, _ = backend.adapter(prior.roll(1, dims=-1).to(backend.device),
                                                     rs_rows, smask)
            mapped_R1, alpha_R1, _ = backend.adapter(proj1.to(backend.device), rs_rows, smask)
        lm0, la0 = maps_for(backend, mapped_R0, alpha)
        lm1, la1 = maps_for(backend, mapped_R1, alpha_R1)
        nm, na = maps_for(backend, null_R0, null_alpha)
        if args.save_r1_priors:
            r1_priors[sid] = {str(L): mapped_R1[i].detach().float().cpu()
                              for i, L in enumerate(layers)}
        print(json.dumps({"shard": args.shard_name, "sample": si, "src": row["source_index"],
                          "alpha_equal": bool(torch.allclose(alpha, alpha_R1, atol=1e-6))}),
              flush=True)

        for k in [x for x in shard if x[0] == sid]:
            d = dr[k]
            t = int(k[1])
            base_tokens = list(map(int, d["base_tokens"]))
            prefix = list(map(int, target.input_ids)) + base_tokens[:t]
            y_T, y_S = int(d["y_target"]), int(d["y_base"])
            ids = torch.tensor([prefix], device=backend.device)
            amask = torch.ones_like(ids)

            # pass A: frozen full forward (no graph)
            cap: dict = {}

            def pre_hook(module, hargs, hkwargs):
                cap["h"] = hargs[0].detach().clone()
                for kk in ("attention_mask", "position_ids", "cache_position",
                           "position_embeddings"):
                    cap[kk] = hkwargs.get(kk)
                return None

            handle = deep[0].register_forward_pre_hook(pre_hook, with_kwargs=True)
            try:
                with torch.no_grad():
                    oA = backend.student(input_ids=ids, attention_mask=amask, use_cache=False,
                                         output_attentions=False)
                    z_full = oA.logits[0, -1, :vocab].float().cpu()
            finally:
                handle.remove()
            del oA

            # pass B: patched layers only, with graph
            h = cap["h"].requires_grad_(True)
            with torch.enable_grad():
                with steering_diag(backend.student, {}, None, eps,
                                   selected_layers=set(layers), phase="R1_GRAD",
                                   capture_rows={}) as st:
                    for layer in deep:
                        h = layer(h, attention_mask=cap["attention_mask"],
                                  position_ids=cap["position_ids"], past_key_values=None,
                                  use_cache=False, cache_position=cap["cache_position"],
                                  position_embeddings=cap["position_embeddings"])
                    h = backbone.norm(h)
                    logits = backend.student.lm_head(h)[0, -1, :vocab].float()
                z_base = logits.detach().cpu()
                M = float(logits[y_T] - logits[y_S])
                logp_base = logprob_of(logits.detach(), y_T)
                (logits[y_T] - logits[y_S]).backward()
                full_rows = [st.capture_rows[L] for L in layers]
                grad_ok = [r.grad is not None for r in full_rows]
                p_base = [r[0, :, 0, :].detach().float().cpu() for r in full_rows]
                grads = [torch.zeros_like(p_base[i]) if r.grad is None
                         else r.grad[0, :, 0, :].detach().float().cpu()
                         for i, r in enumerate(full_rows)]
            recon = float((z_base - z_full).abs().max())
            del h, logits, full_rows, cap
            torch.cuda.empty_cache()

            # the frozen parent definition of the Target correction direction
            with torch.no_grad():
                oT = model(input_ids=ids, attention_mask=amask, use_cache=False)
                z_target = oT.logits[0, -1, :vocab].float().cpu()
            del oT
            torch.cuda.empty_cache()
            d_target = z_target - z_base
            y_T_recomputed = int(torch.argmax(z_target).item())
            del z_target

            per_layer, dirs = {}, {}
            for i, L in enumerate(layers):
                p, g = p_base[i], grads[i]
                K = p.shape[-1]
                fake = p.unsqueeze(0).unsqueeze(2)

                def mix_with(maps, alphas):
                    s = type("S", (), {})()
                    s.layer_to_prior, s.layer_to_alpha = maps, alphas
                    s.eps, s.mix_log = eps, []
                    out_mixed = apply_mix(s, L, fake.clone(), K)
                    return out_mixed[0, :, 0, :].float().cpu()

                d0 = mix_with(lm0, la0) - p
                d1 = mix_with(lm1, la1) - p
                dn = mix_with(nm, na) - p
                dirs[L] = {"R0": d0, "R1": d1, "NULL": dn}
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
                "V_first_order": {a: sum(per_layer[str(L)][f"V_{a}"] for L in layers)
                                  for a in ("R0", "R1", "Null")},
                "l1_first_order": {a: sum(per_layer[str(L)][f"d1_{a}_l1"] for L in layers)
                                   for a in ("R0", "R1", "Null")},
                "layers": per_layer,
                "track_a": {"present": bool(k in ta),
                            "M_base": None if k not in ta else ta[k]["M_base"],
                            "V_teacher": None if k not in ta else ta[k]["V_teacher"],
                            "V_null": None if k not in ta else ta[k]["V_null"],
                            "V_oracle": None if k not in ta else ta[k]["V_oracle"],
                            "teacher_efficiency": None if k not in ta
                            else ta[k]["teacher_efficiency"],
                            "null_efficiency": None if k not in ta else ta[k]["null_efficiency"],
                            "oracle_l1_total": None if k not in ta else ta[k]["oracle_l1_total"],
                            "prefix_sha256": None if k not in ta else ta[k]["prefix_sha256"]},
                "dr_reference": {"dM_teacher_R0_incremental": d["dM_teacher"],
                                 "dlogp_T_teacher_R0_incremental": d["dlogp_T_teacher"],
                                 "cos_teacher_R0_incremental": d["cos_teacher"],
                                 "dM_null_incremental": d["dM_null"],
                                 "cos_null_incremental": d["cos_null"],
                                 "note": "parent diagnostic framing (prefill + steered decode); "
                                         "reused as an external cross-check, not as the paired "
                                         "baseline"},
                "alpha_R0_mean": float(alpha.mean()), "alpha_R1_mean": float(alpha_R1.mean()),
                "run_config_hash": cfg["config_hash"],
                "checkpoint_sha256": cfg["checkpoint_sha256"], "shard": args.shard_name,
            }
            records.append(rec)
            H.write_result_envelope(
                body, task_id="RESCUE-R1-VALUE-NORM-DECISION-ALIGNMENT-PILOT-001",
                stage="r1_decision_alignment", records=records, arms=["decision_position"],
                sample_ids=sorted({r["canonical_source_id"] for r in records}),
                config_hash=cfg["config_hash"], checkpoint_sha256=cfg["checkpoint_sha256"])
            print(json.dumps({"shard": args.shard_name, "pos": len(records), "src": d["source_index"],
                              "t": t, "dM_R0": round(arm_out["R0"]["M"] - M, 5),
                              "dM_R1": round(arm_out["R1"]["M"] - M, 5),
                              "dM_Null": round(arm_out["NULL"]["M"] - M, 5),
                              "V_R1": round(rec["V_first_order"]["R1"], 5),
                              "flip1": arm_out["R1"]["flip_to_target"]}), flush=True)
            del p_base, grads, dirs, st

    H.write_result_envelope(
        body, task_id="RESCUE-R1-VALUE-NORM-DECISION-ALIGNMENT-PILOT-001",
        stage="r1_decision_alignment", records=records, arms=["decision_position"],
        sample_ids=sorted({r["canonical_source_id"] for r in records}),
        config_hash=cfg["config_hash"], checkpoint_sha256=cfg["checkpoint_sha256"])
    if args.save_r1_priors and r1_priors:
        torch.save(r1_priors, args.save_r1_priors)
    meta = {"shard": args.shard_name, "planned_positions": len(pop),
            "positions": len(records), "samples": len(samples),
            "identity_mismatches": len(mismatched), "identity": identity,
            "teacher_layers": teacher_layers, "target_device_map": dmap,
            "wall_seconds": round(time.time() - t0, 1), "body": str(body),
            "body_sha256": sha256_file(body),
            "verify": H.verify_result(body, expected_config_hash=cfg["config_hash"],
                                      expected_checkpoint_sha256=cfg["checkpoint_sha256"])}
    H.atomic_write_text(out / f"r1_shard_{args.shard_name}_meta.json",
                        json.dumps(meta, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({"shard": args.shard_name, "done": True, "positions": len(records),
                      "verify": meta["verify"]["status"],
                      "wall_seconds": meta["wall_seconds"]}, ensure_ascii=False), flush=True)
    return 0 if meta["verify"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
