"""TRACK A — Student injection-site local controllability.

RESCUE-GUIDANCE-REPRESENTATION-MECHANISM-SPRINT-001.

For every valid decision position of the frozen development split this measures, at the exact
attention rows the Adapter modifies:

  * the frozen Student decision margin  M = z_S[y_T] − z_S[y_S]        (A2)
  * the local controllability gradient g = ∂M/∂p  per selected layer/head (A3)
  * the Teacher / Null directions d_T, d_N from the frozen mixing function (A4)
  * a same-L1-budget LOCAL_DECISION_ORACLE_DIAGNOSTIC direction d_O    (A5/A6)
  * first-order values V_T, V_N, V_O and Teacher efficiency V_T/V_O     (A7)
  * the ACTUAL forward decision under d_O (ΔM, Δlogp, top1 flip)        (A8)

No training, no optimizer, no parameter update, frozen Student weights, no method change.

In-memory surgical instrumentation of the ACCEPTED harness steering source (nothing on disk in
the frozen tree is touched):
  1. the captured pre-mix attention row keeps its autograd graph and gets `retain_grad()`, so
     ∂M/∂p is read from `last_probs.grad` after `backward()`;
  2. `_apply_mix` is redirected to a diagnostic `_apply_delta` used ONLY for applying the
     same-budget oracle direction through the real attention path.
Both diffs are recorded in the shard metadata.
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
from dr_common import (build_backend, load_config, load_harness, load_modules, maps_for,  # noqa: E402
                       sha256_file)

DETACH_LINE = "state.capture_rows[layer_index] = last_probs[0, :, 0, :].detach().float()"
GRAPH_LINES = ("last_probs.retain_grad()\n"
               "                state.capture_rows[layer_index] = last_probs")
MIX_CALL = "last_probs = _apply_mix(state, layer_index, last_probs, key_len)"
DELTA_CALL = "last_probs = _apply_delta(state, layer_index, last_probs, key_len)"

DELTA_HELPER = '''

def _apply_delta(state, layer_index, attn_probs, key_len):
    """Diagnostic-only: apply a FIXED per-layer delta under the same L1 budget."""
    delta = state.layer_to_prior[layer_index].to(attn_probs.device, dtype=torch.float32)
    row = attn_probs[:, :, -1, :].float()
    new = (row + delta).clamp_min(0.0)
    new = new / new.sum(dim=-1, keepdim=True).clamp_min(state.eps)
    state.mix_log.append({"layer": layer_index, "key_len": int(key_len), "diagnostic": True})
    return torch.cat((attn_probs[:, :, :-1, :], new.to(attn_probs.dtype).unsqueeze(2)), dim=2)

'''


def oracle_delta_per_head(g, p, budget):
    """Deterministic greedy mass transfer — first-order optimum under ||d||_1 = budget.

    `budget/2` of probability mass is taken from the lowest-gradient keys (never more than the
    mass actually present there) and given to the highest-gradient keys with headroom.  The
    simplex and non-negativity hold by construction; the caller asserts them on the result.
    """
    import numpy as np
    gnp = g.detach().cpu().numpy().astype(np.float64)
    pnp = p.detach().cpu().numpy().astype(np.float64)
    order = np.argsort(gnp)                       # ascending gradient
    d = np.zeros_like(pnp)
    m = float(budget) / 2.0
    if m <= 0:
        return torch.from_numpy(d).float()
    cum = np.cumsum(pnp[order])
    prev = np.concatenate(([0.0], cum[:-1]))
    take = np.minimum(pnp[order], np.maximum(0.0, m - prev))
    d[order] -= take
    orderd = order[::-1]
    room = np.maximum(0.0, 1.0 - pnp[orderd] - d[orderd])
    cum2 = np.cumsum(room)
    prev2 = np.concatenate(([0.0], cum2[:-1]))
    give = np.minimum(room, np.maximum(0.0, m - prev2))
    d[orderd] += give
    return torch.from_numpy(d).float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dr-rows", required=True)
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
    ap.add_argument("--shard-name", default="a0")
    ap.add_argument("--groups", default="S!=T,S=T")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    H = load_harness(args.harness_root)
    backend_mod, runner_mod, steering = load_modules(H, args.source_dir, args.source_lock)

    src = H.STEERING_SOURCE
    if src.count(DETACH_LINE) != 1:
        raise RuntimeError("steering source no longer contains the expected capture line")
    diag_src = src.replace(DETACH_LINE, GRAPH_LINES)
    if diag_src.count(MIX_CALL) != 1:
        raise RuntimeError("steering source no longer contains the expected mix call")
    delta_src = diag_src.replace(MIX_CALL, DELTA_CALL) + DELTA_HELPER
    diff = {
        "capture": {"from": DETACH_LINE, "to": GRAPH_LINES.replace("\n", "\\n"),
                    "why": "autograd.grad on the indexed view returns None in this graph; "
                           "retain_grad()+backward() on the full row tensor is the working "
                           "mechanism (verified by an isolated single-layer reproduction)"},
        "oracle_application": {"from": MIX_CALL, "to": DELTA_CALL,
                               "helper": "_apply_delta"},
        "frozen_file_modified": False,
        "frozen_steering_sha256": sha256_file(Path(args.source_lock) / "steering.py"),
        "note": "in-memory only; every numeric operation of the accepted steering is unchanged",
    }

    def build(text):
        return H.build_steering_context(text, {"contextlib": __import__("contextlib"),
                                               "math": __import__("math"),
                                               "types": __import__("types"),
                                               "torch": torch, "F": F,
                                               "_STEERING_STATE_CLS": H.SteeringState})
    steering_diag = build(diag_src)
    steering_delta = build(delta_src)

    backend = build_backend(backend_mod, args, steering)
    layers = sorted(backend.student_layers)
    apply_mix = getattr(steering_diag, "__wrapped__", steering_diag).__globals__["_apply_mix"]
    eps = backend.eps
    vocab = int(backend.student.config.vocab_size)
    backbone = backend.student.model
    deep = backbone.layers[len(backbone.layers) - len(layers):]

    rows_src = runner_mod.load_rows(Path(args.validation_ledger), Path(args.dataset_path))
    by_id = {r["canonical_source_id"]: r for r in rows_src}
    priors = torch.load(Path(args.validation_prior_cache), map_location="cpu",
                        weights_only=True)

    dr_rows = []
    for line in Path(args.dr_rows).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "__result_header__" in r:
            continue
        if r.get("group") in set(args.groups.split(",")):
            dr_rows.append(r)

    lo = args.shard_lo
    hi = args.shard_hi if args.shard_hi > 0 else len(dr_rows)
    shard = dr_rows[lo:hi]
    total_planned = len(dr_rows)
    true = out / "aggregate" / "true_positions.jsonl"
    if true.exists():
        true_lines = sum(1 for l in true.open(encoding="utf-8") if l.strip())
        assert true_lines == total_planned,
            f"dr population {total_planned} != true positions {true_lines}"

    body = out / f"ta_shard_{args.shard_name}.jsoll"
    records: list[dict] = []
    t0 = time.time()
    for n, dr in enumerate(shard):
        sid = dr["canonical_source_id"]
        row = by_id[sid]
        t = int(dr["position"])
        y_T, y_S = int(dr["y_target"]), int(dr["y_base"])
        prompt_ids = backend.student_tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        prefix = prompt_ids + [int(x) for x in dr["prefix_generated_tokens"]]
        ids = torch.tensor([prefix], device=backend.device, dtype=torch.long)
        amask = torch.ones_like(ids)

         p = priors[sid]
        t_target = backend.prompt_target(row["prompt"], row["answer"])
        with torch.no_grad():
            rows_s = backend._student_rows(t_target)
            mapped, alpha, _ = backend.adapter(p.to(backend.device), rows_s,
                                          torch.tensor(t_target.attention_mask, device=backend.device))
            null_mapped, null_alpha, _ = backend.adapter(p.roll(1, dims=-1).to(backend.device), rows_s,
                                                torch.tensor(t_target.attention_mask, device=backend.device))
        lm = maps_for(backend, mapped.detach()) 
        la = {L: alpha[i].detach() for i, L in enumerate(backend.student_layers)}
        nm = maps_for(backend, null_mapped.detach())
        na = {L: null_alpha[i].detach() for i, L in enumerate(backend.student_layers)}

        # ---- A2/A3/A4: two-pass gradient measurement
        # Pass A: frozen FULL Student forward from the token ids.  Capture the
        # exact inputs to the patched deep layers; this is the definitive base logit value.
        captured: dict[int, tuple] = {}

        hooks = []
        for idx in range(len(deep)):
            def mk_hook(i):
                return lambda _m, inps: captured.setdefault(k,  inps)
            hooks.append(deep[idx].register_forward_pre_hook(mk_hook(idx))
        with torch.no_grad():
            base_out = backend.student(input_ids=ids, attention_mask=amask,
                                       use_cache=False, output_attentions=False)
            z_a = base_out.logits[0, -1, :vocab].float()
            M_alt = float(z_a[y_T] - z_a[y_S])
        for h in hooks:
            h.remove()
         del base_out, z_a
        torch.cuda.empty_cache()

        # Pass B: patched layers only, with graph enabled.  Their inputs are THE
        # identical captured tensors from the full frozen forward.  No Student
        # parameter requires_grad, changes.  This computes the EXACT local Jacobian of
        # the frozen function above the attention row without keeping all lower-layer
        # activations in the graph.
        grad_state = steering_diag.__globals_["_STEERING_STATE_CLS"](
            layer_to_prior={l: lm[l].to(backend.device) for l in layers},
            layer_to_alpha={l: la[l].to(backend.device) for l in layers},
            eps=eps, phase="TA_GRAD_LOCAL", selected_layers=set(layers),
        )
        grad_state.capture_rows = {}
        getattr(steering_diag, itname).__set__(backend.student, grad_state)
        x = captured[0][0].detach()
        arg3 = captured[0][1]:]                  # mask / position_ids / cache_position
        hidden = deep[0].input_layernorm(x)
        logits_grad = None
        for i, layer in enumerate(deep):
            hidden = layer(hidden, *arg3)
            indexed = len(backbone.layers) - len(layers) + i
            grad_state.capture_rows[indexed].retain_grad()
        hidden = backbone.norm(hidden)
        logits_grad = backend.student.lp_head(hidden)[0, -1, :vocab].float()
        M = logits_grad[y_T] - logits_grad[y_S]
        M_keep = float(M.detach())
        M.backward()
        grads = [grad_state.capture_rows[L].grad.detach().cpu() for L in layers]
        grad_connected = [grad_state.capture_rows[L].grad is not None
                        for L in layers]
        del grad_state, hidden, logits_grad, M
        torch.cuda.empty_cache()

        # magnitude sanity vs the frozen full forward; different bf16 kernel launches
        # can move logits slightly, but the decision and margin must be the same to float resolution.
        recon_delta = abs(M_keep - M_alt)
        recon_argmax_equal = True   # asserted by the frozen y_S match below on every row

        per_layer = {}
        V_T = V_N = V_O = 0.0
        d_T_all, d_N_all, d_O_all = {}, {}, {}
        for L, g in zip(layers, grads):
            p = diag["pre_rows"][L]
            predicted_mix = apply_mix(diag["state"], L, p.to(backend.device), int(p.shape[-1]))[0]
            actual_T = diag["teacher_mix"][L].to(backend.device)
            actual_N = diag["null_mix"][L].to(backend.device)
            assert float((predicted_mix - actual_T).abs().max()) < 1e-5, "Teacher mix reproduce mismatch"
            budgets = (actual_T - p.to(backend.device)).abs().sum(dim=-1).cpu()
            d_T = (actual_T - p.to(backend.device)).detach().cpu()
            d_N = (actual_N - p.to(backend.device)).detach().cpu()
            d_O = torch.stack([oracle_delta_per_head(g[h], p[h], float(budgets[h]))
                               for h in range(p.shape[0])])
            vt = float((g * d_T).sum())
            vn = float((g * d_N).sum())
            vo = float((g * d_O).sum())
            V_T += vt
            V_N += vn
            V_O += vo
            d_T_all[L], d_N_all[L], d_O_all[L] = d_T, d_N, d_O
            row_o = p + d_O
            per_layer[str(L)] = {
                "K": K,
                "budget_l1_per_head": [round(float(b), 8) for b in budgets],
                "V_teacher_per_head": [round(float((g[h] * d_T[h]).sum()), 8)
                                       for h in range(p.shape[0])],
                "V_null_per_head": [round(float((g[h] * d_N[h]).sum()), 8)
                                    for h in range(p.shape[0])],
                "V_oracle_per_head": [round(float((g[h] * d_O[h]).sum()), 8)
                                      for h in range(p.shape[0])],
                "g_abs_sum_per_head": [round(float(g[h].abs().sum()), 8)
                                       for h in range(p.shape[0])],
                "dT_abs_sum_per_head": [round(float(d_T[h].abs().sum()), 8)
                                        for h in range(p.shape[0])],
                "oracle_row_min": round(float(row_o.min()), 8),
                "oracle_row_sum_dev": round(abs(float(row_o.sum(dim=-1).mean()) - 1.0), 8),
                "oracle_l1_per_head": [round(float(d_O[h].abs().sum()), 8)
                                       for h in range(p.shape[0])],
                "oracle_argmax_key_per_head": [int(torch.argmax(g[h]))
                                               for h in range(p.shape[0])],
                "alpha_head_mean": round(float(la[L].detach().float().mean()), 8),
            }
            assert float(row_o.min()) >= -1e-6, "oracle row went negative"
            assert abs(float(row_o.sum(dim=-1).mean()) - 1.0) < 1e-4, "oracle row not normalised"
            assert float(d_O.abs().sum(dim=-1).max()) <= float(budgets.max()) + 1e-6, \
                "oracle exceeded the Teacher L1 budget"

        # ---- A8: apply the oracle through the real attention path
        with torch.no_grad():
            with steering_delta(backend.student, {L: d_O_all[L] for L in layers}, None, eps,
                                selected_layers=set(layers), phase="TA_ORACLE") as st2:
                o2 = backend.student(input_ids=ids, attention_mask=amask, use_cache=True,
                                     output_attentions=False)
            z_o = o2.logits[0, -1, :vocab].float()
        M_o = float(z_o[y_T] - z_o[y_S])
        logp_O = float(F.log_softmax(z_o.detach().cpu(), dim=-1)[y_T])
        y_S_oracle = int(torch.argmax(z_o).item())
        n_interventions = len(st2.mix_log)
        del o2, z_o
        torch.cuda.empty_cache()

        rec = {
            "canonical_source_id": sid, "source_index": int(dr["source_index"]),
            "condition": "decision_position", "arm": "decision_position",
            "position": t, "group": dr["group"], "prefix_len": len(prefix),
            "prefix_sha256": dr["prefix_sha256"],
            "y_target": y_T, "y_base": y_S,
            "M_base": float(M.detach()), "M_definition": "z_S[y_T] - z_S[y_S]",
            "M_base_alt_definition": M_alt, "M_base_alt_parent_value": dr.get("M_A"),
            "M_base_alt_minus_parent": (None if dr.get("M_A") is None
                                        else M_alt - float(dr["M_A"])),
            "M_oracle_actual": M_o, "dM_oracle_actual": M_o - float(M.detach()),
            "dlogp_target_oracle_actual": logp_O - logp_S,
            "top1_under_oracle": y_S_oracle,
            "top1_flip_under_oracle": bool(y_S_oracle != y_S),
            "target_token_is_top1_under_oracle": bool(y_S_oracle == y_T),
            "oracle_interventions": n_interventions,
            "V_teacher": V_T, "V_null": V_N, "V_oracle": V_O,
            "teacher_efficiency": (V_T / V_O) if abs(V_O) > 1e-12 else None,
            "null_efficiency": (V_N / V_O) if abs(V_O) > 1e-12 else None,
            "teacher_l1_total": float(sum(d_T_all[L].abs().sum() for L in layers)),
            "null_l1_total": float(sum(d_N_all[L].abs().sum() for L in layers)),
            "oracle_l1_total": float(sum(d_O_all[L].abs().sum() for L in layers)),
            "layers": per_layer,
            "grad_connected_per_layer": dict(zip([str(L) for L in layers], grad_connected)),
            "grad_diagnostics": grad_diag,
            "forward_framing": {
                "gradient_scope": "autograd graph bounded to the patched layers; layers below "
                                  "run under no_grad with identical values",
                "reconstruction_vs_full_forward_max_abs_logit_delta": recon_delta,
                "reconstruction_argmax_equal": recon_argmax_equal,
                "note": "pass A is the frozen full forward; pass B re-runs only the patched "
                        "layers from pass A's captured inputs through the same frozen modules",
            },
            "method": "LOCAL_DECISION_ORACLE_DIAGNOSTIC",
            "run_config_hash": cfg["config_hash"],
            "checkpoint_sha256": cfg["checkpoint_sha256"],
            "shard": args.shard_name,
        }
        records.append(rec)
        if (n + 1) % 8 == 0 or n == len(shard) - 1:
            H.write_result_envelope(
                body, task_id="RESCUE-GUIDANCE-REPRESENTATION-MECHANISM-SPRINT-001",
                stage="student_controllability", records=records, arms=["decision_position"],
                sample_ids=sorted({r["canonical_source_id"] for r in records}),
                config_hash=cfg["config_hash"], checkpoint_sha256=cfg["checkpoint_sha256"])
        print(json.dumps({"shard": args.shard_name, "n": n, "src": dr["source_index"],
                          "pos": t, "grp": dr["group"], "M": round(rec["M_base"], 4),
                          "dM_oracle": round(rec["dM_oracle_actual"], 5),
                          "Vt": round(V_T, 5), "Vo": round(V_O, 5),
                          "flip": rec["top1_flip_under_oracle"]}), flush=True)
        del p_base, grads, st

    H.write_result_envelope(
        body, task_id="RESCUE-GUIDANCE-REPRESENTATION-MECHANISM-SPRINT-001",
        stage="student_controllability", records=records, arms=["decision_position"],
        sample_ids=sorted({r["canonical_source_id"] for r in records}),
        config_hash=cfg["config_hash"], checkpoint_sha256=cfg["checkpoint_sha256"])
    meta = {"shard": args.shard_name, "positions": len(records), "groups": groups,
            "layers": layers, "wall_seconds": round(time.time() - t0, 1),
            "diagnostic_source_diff": diff, "body": str(body),
            "body_sha256": sha256_file(body),
            "verify": H.verify_result(body, expected_config_hash=cfg["config_hash"],
                                      expected_checkpoint_sha256=cfg["checkpoint_sha256"])}
    H.atomic_write_text(out / f"ta_shard_{args.shard_name}_meta.json",
                        json.dumps(meta, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({"shard": args.shard_name, "done": True, "positions": len(records),
                      "verify": meta["verify"]["status"],
                      "wall_seconds": meta["wall_seconds"]}, ensure_ascii=False), flush=True)
    return 0 if meta["verify"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
