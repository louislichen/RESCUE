"""TD — Teacher decision attribution (RESCUE-DECISION-AWARE-TEACHER-SIGNAL-SPRINT-001, Phase C).

Canonical object, frozen by TEACHER_DECISION_ATTRIBUTION_IMPLEMENTATION_REVIEW:

    M_T          = z_T[y_T] - z_T[y_S]
    score[l,h,j] = d M_T / d a[l,h,j]

`a` is the LAST-QUERY-ROW attention probability at the four frozen teacher layers (24-27),
post-mask, post-softmax (fp32), pre-dropout, post-repeat_kv, pre-AV -- exactly the tensor the
frozen `steering.qwen2_teacher_row_observer` produces, kept attached to the autograd graph.

Extraction mechanism (same one the accepted Track A used on the Student): the harness V3
STEERING_SOURCE is installed on the four frozen teacher layers with an EMPTY prior map (capture
only; `_apply_mix` is never reached), the capture line is replaced in memory by
`last_probs.retain_grad()`, and the gradient is read from `last_probs.grad[0, :, 0, :]` after
`backward()`.  `autograd.grad` on the sliced view returns None in this graph (established earlier
in this project by an isolated reproduction).  Nothing on disk in the frozen tree is touched.

Autograd scope: layers 0-23 run once under no_grad through the frozen model forward (pass A) and
their captured layer-24 inputs seed pass B, which re-runs exactly the frozen modules of layers
24-27.  No parameter requires grad, so no weight gradients are ever allocated.

Teacher-space question 1 of the sprint (does the Teacher possess the decision-aware signal): the
signed score is turned into a direction by the SAME deterministic greedy mass transfer the accepted
Track A used for the Student oracle direction, at the same per-head L1 budget (both rows are
probability distributions over the same keys, so the budget is directly comparable), then applied
through the real frozen attention path and the ACTUAL dM_T of the Teacher's own decision is
measured.  Oracle-style, non-deployable.

Usage:
  td_teacher_attribution.py --harness-root H --config CFG --population POP --positions-set SET \\
      --k4-dir KD --source-dir S --source-lock SL --teacher-model TM --student-model SM \\
      --dataset-path DS --validation-ledger V --validation-prior-cache P --checkpoint CK \\
      --set-name short|long|pilot|all_population --shard-lo 0 --shard-hi 0 --shard-name t0 \\
      --out OUTDIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dr_common import (build_backend, load_config, load_harness, load_modules,  # noqa: E402
                       sha256_file)

TASK_ID = "RESCUE-DECISION-AWARE-TEACHER-SIGNAL-SPRINT-001"
TEACHER_LAYER_SCOPE = "last_4_layers"

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


def prefix_sha(prompt_ids, toks) -> str:
    """The frozen parent's prefix identity hash (dr_stream.prefix_sha)."""
    return hashlib.sha256(json.dumps([list(map(int, prompt_ids)), list(map(int, toks))],
                                     separators=(",", ":")).encode()).hexdigest()


def greedy_transfer(score, p, budget):
    """Deterministic greedy mass transfer, ||d||_1 = budget, ordered by `score` ascending.

    Byte-for-byte the construction of the accepted Track A oracle direction
    (`ta_controllability.oracle_delta_per_head`), with the Student controllability gradient
    replaced by the signed Teacher attribution.  Taking from the most negative and giving to the
    most positive attributed keys is the signed-input analogue of the same operator; the simplex
    and non-negativity hold by construction.
    """
    import numpy as np
    snp = np.asarray(score, dtype=np.float64)
    pnp = np.asarray(p, dtype=np.float64)
    d = np.zeros_like(pnp)
    m = float(budget) / 2.0
    if m <= 0.0:
        return torch.from_numpy(d).float()
    order = np.argsort(snp)
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


def select_positions(pop_rows, set_name, set_file):
    """Frozen selection: never re-selected after seeing attribution values."""
    if set_name == "all_population":
        return list(pop_rows), None
    spec = json.loads(Path(set_file).read_text(encoding="utf-8"))
    if set_name == "short":
        wanted = [spec["one_short_position"]]
    elif set_name == "long":
        wanted = [spec["one_long_position"]]
    elif set_name == "smoke":
        wanted = [spec["one_short_position"], spec["one_long_position"]]
    elif set_name == "pilot":
        wanted = list(spec["eight_position_pilot"])
    else:
        raise SystemExit(f"unknown --set-name {set_name}")
    order = {(w["sample_id"], int(w["position"])): i for i, w in enumerate(wanted)}
    keys = set(order)
    sel = [r for r in pop_rows if (r["sample_id"], int(r["position"])) in keys]
    sel.sort(key=lambda r: order[(r["sample_id"], int(r["position"]))])
    if len(sel) != len(wanted):
        raise SystemExit(f"selection incomplete: {len(sel)} of {len(wanted)}")
    for r in sel:
        w = wanted[order[(r["sample_id"], int(r["position"]))]]
        if int(w["prefix_len"]) != int(r["prefix_len"]):
            raise SystemExit(f"prefix_len mismatch for {r['sample_id']} pos {r['position']}")
        if w["prefix_sha256"] != r["prefix_sha256"]:
            raise SystemExit(f"prefix_sha256 mismatch for {r['sample_id']} pos {r['position']}")
    return sel, spec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--population", required=True)
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
    ap.add_argument("--target-max-memory", default="8GiB")
    ap.add_argument("--target-dtype", default="bfloat16")
    ap.add_argument("--shard-lo", type=int, default=0)
    ap.add_argument("--shard-hi", type=int, default=0)
    ap.add_argument("--shard-name", default="t0")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--framing-check", action="store_true")
    ap.add_argument("--observer-check", action="store_true")
    ap.add_argument("--abort-vram-fraction", type=float, default=0.90)
    ap.add_argument("--gpu-total-mib", type=int, default=11264)
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
    from steering import qwen2_teacher_row_observer

    backend = build_backend(backend_mod, args, steering)
    eps = backend.eps
    vocab = int(backend.student.config.vocab_size)
    model, tdev, dmap = __import__("real_k4_runner").build_target(args)
    vocab = min(vocab, int(model.config.vocab_size))
    tlayers = list(RHT.selected_layer_indices(int(model.config.num_hidden_layers),
                                              TEACHER_LAYER_SCOPE))
    tblocks = model.model.layers
    deep = tblocks[len(tblocks) - len(tlayers):]
    if [int(i) for i in range(len(tblocks) - len(tlayers), len(tblocks))] != tlayers:
        raise SystemExit("teacher layer scope is not the frozen last_4_layers")

    pop_all = json.loads(Path(args.population).read_text(encoding="utf-8"))
    pop_rows = pop_all["rows"]
    sel, set_spec = select_positions(pop_rows, args.set_name, args.positions_set)
    if args.shard_hi:
        sel = sel[args.shard_lo:args.shard_hi]
    elif args.shard_lo:
        sel = sel[args.shard_lo:]

    # ---- in-memory diagnostic source diffs (frozen tree untouched)
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
                           "mechanism (verified earlier in this project)"},
        "teacher_oracle_application": {"from": MIX_CALL, "to": DELTA_CALL,
                                       "helper": "_apply_delta"},
        "capture_mode_prior_map": "empty ({}) -> _apply_mix is never reached; the attention "
                                  "value path is the frozen one, only the row is captured",
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

    body = out / f"td_teacher_rows_{args.shard_name}.jsonl"
    scores_path = out / f"td_scores_{args.shard_name}.pt"
    raw: dict = {}
    if scores_path.exists():
        raw = torch.load(scores_path, map_location="cpu", weights_only=False)
    records: list[dict] = []
    t_shard = time.time()

    def peak_mib():
        return {i: round(torch.cuda.max_memory_allocated(i) / (1024 * 1024), 1)
                for i in range(torch.cuda.device_count())}

    for n, r in enumerate(sel):
        sid = r["sample_id"]
        t = int(r["position"])
        ids = [int(x) for x in r["prefix_token_ids"]]
        L = len(ids)
        pt = L - t
        sha = prefix_sha(ids[:pt], ids[pt:])
        if sha != r["prefix_sha256"]:
            raise SystemExit(f"prefix identity failed for {sid} pos {t}: {sha}")
        y_T, y_S = int(r["y_T"]), int(r["y_S"])
        froz = r["student_local_oracle_identity"]
        budgets = {int(k): [float(x) for x in v]
                   for k, v in froz["per_layer_budget_l1"].items()}
        budget_T = {lay: (sum(budgets[lay]) / len(budgets[lay])) for lay in tlayers}

        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
        free_before = {i: round(torch.cuda.mem_get_info(i)[0] / (1024 * 1024), 1)
                       for i in range(torch.cuda.device_count())}
        ids_t = torch.tensor([ids], device=tdev)
        amask = torch.ones_like(ids_t)
        t0 = time.time()

        # ---------- pass A: frozen full forward under no_grad, capture layer-24 inputs
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
                oA = model(input_ids=ids_t, attention_mask=amask, use_cache=False)
                z_full = oA.logits[0, -1, :vocab].float().cpu()
        finally:
            handle.remove()
        del oA
        M_T_full = float(z_full[y_T]) - float(z_full[y_S])
        t_top1 = int(torch.argmax(z_full).item())
        t_pass_a = time.time() - t0

        # ---------- pass B: only the four frozen layers carry the graph
        t1 = time.time()
        h = cap["h"].requires_grad_(True)
        with torch.enable_grad():
            with steering_diag(model, {}, None, eps, selected_layers=set(tlayers),
                               phase="TD_GRAD_CAPTURE", capture_rows={}) as st:
                for layer in deep:
                    h = layer(h, attention_mask=cap["attention_mask"],
                              position_ids=cap["position_ids"], past_key_values=None,
                              use_cache=False, cache_position=cap["cache_position"],
                              position_embeddings=cap["position_embeddings"])
                h = model.model.norm(h)
                logits = model.lm_head(h)[0, -1, :vocab].float()
            z_b = logits.detach().cpu()
            M_T = logits[y_T] - logits[y_S]
            M_T.backward()
            full_rows = [st.capture_rows[lay] for lay in tlayers]
            grad_connected = [row.grad is not None for row in full_rows]
            p_teacher = [row[0, :, 0, :].detach().float().cpu() for row in full_rows]
            score_rows = [torch.zeros_like(p_teacher[i]) if row.grad is None
                          else row.grad[0, :, 0, :].detach().float().cpu()
                          for i, row in enumerate(full_rows)]
        recon = float((z_b - z_full).abs().max())
        recon_argmax_equal = bool(int(torch.argmax(z_b)) == int(torch.argmax(z_full)))
        t_pass_b = time.time() - t1
        no_param_grad = all(p.grad is None for p in model.parameters())
        grad_diag = {
            "p_requires_grad": [bool(row.requires_grad) for row in full_rows],
            "p_grad_present": grad_connected,
            "p_shape": [list(row.shape) for row in full_rows],
            "p_device": [str(row.device) for row in full_rows],
            "M_T_requires_grad": bool(M_T.requires_grad),
            "h_entry_requires_grad": bool(cap["h"].requires_grad),
            "h_entry_device": str(cap["h"].device),
            "mask_device": str(cap["attention_mask"].device),
            "position_ids_device": str(getattr(cap["position_ids"], "device", None)),
            "position_embeddings_devices": [str(x.device) for x in cap["position_embeddings"]],
            "captured_layers": sorted(int(x) for x in st.capture_rows.keys()),
            "layers_run": tlayers,
            "logits_device": str(z_b.device),
            "no_parameter_grad": bool(no_param_grad),
        }
        del h, logits, z_b, full_rows
        torch.cuda.empty_cache()
        if not all(grad_connected) or not no_param_grad:
            raise SystemExit(f"gradient contract violated at {sid} pos {t}: "
                             f"connected={grad_connected} no_param_grad={no_param_grad}")

        # ---------- teacher-space check (question 1): does the Teacher have the signal?
        t2 = time.time()
        d_teacher = {}
        per_layer = {}
        v_td_teacher = 0.0
        for i, lay in enumerate(tlayers):
            sc, p = score_rows[i], p_teacher[i]
            b = budget_T[lay]
            d = torch.stack([greedy_transfer(sc[hh], p[hh], b) for hh in range(sc.shape[0])])
            d_teacher[lay] = d
            v_td_teacher += float((sc * d).sum())
            per_layer[str(lay)] = {
                "K": int(sc.shape[-1]),
                "budget_l1_per_head": [round(b, 8) for _ in range(sc.shape[0])],
                "student_side_budget_mean_l1": round(b, 8),
                "score_l1_per_head": [round(float(sc[hh].abs().sum()), 6)
                                      for hh in range(sc.shape[0])],
                "score_absmax_per_head": [round(float(sc[hh].abs().max()), 8)
                                          for hh in range(sc.shape[0])],
                "score_pos_frac": round(float((sc > 0).float().mean()), 8),
                "score_neg_frac": round(float((sc < 0).float().mean()), 8),
                "score_zero_frac": round(float((sc == 0).float().mean()), 8),
                "score_top_pos_keys": [int(x) for x in torch.topk(sc.flatten(), 5).indices],
                "score_top_neg_keys": [int(x) for x in torch.topk(-sc.flatten(), 5).indices],
                "dTD_teacher_l1_per_head": [round(float(d[hh].abs().sum()), 8)
                                            for hh in range(d.shape[0])],
                "V_TD_teacher_per_head": [round(float((sc[hh] * d[hh]).sum()), 8)
                                          for hh in range(d.shape[0])],
                "p_teacher_row_sum_dev": round(
                    abs(float(p.sum(dim=-1).mean()) - 1.0), 8),
            }
        with torch.no_grad():
            with steering_delta(model, {lay: d_teacher[lay] for lay in tlayers}, None, eps,
                               selected_layers=set(tlayers), phase="TD_TEACHER_ORACLE") as st2:
                o2 = model(input_ids=ids_t, attention_mask=amask, use_cache=True)
            z_o = o2.logits[0, -1, :vocab].float().cpu()
        M_T_act = float(z_o[y_T]) - float(z_o[y_S])
        logp_base = float(F.log_softmax(z_full, dim=-1)[y_T])
        logp_act = float(F.log_softmax(z_o, dim=-1)[y_T])
        top1_act = int(torch.argmax(z_o).item())
        n_inter = len(st2.mix_log)
        del o2, z_o
        torch.cuda.empty_cache()
        t_pass_c = time.time() - t2

        # ---------- optional smoke-only cross-checks
        observer = None
        if args.observer_check:
            caps: dict = {}
            with torch.no_grad():
                with qwen2_teacher_row_observer(model.model, tlayers, caps):
                    _ = model(input_ids=ids_t, attention_mask=amask, use_cache=False)
            del _
            torch.cuda.empty_cache()
            obs = {}
            for i, lay in enumerate(tlayers):
                if lay in caps:
                    a = caps[lay].float()
                    b = p_teacher[i][:, :a.shape[-1]]
                    obs[str(lay)] = {"max_abs_diff": float((a - b).abs().max()),
                                     "shape": list(a.shape)}
            observer = {"max_abs_diff": max(v["max_abs_diff"] for v in obs.values()),
                        "per_layer": obs,
                        "note": "captured pass-B row vs the frozen qwen2_teacher_row_observer "
                                "row on the same prefix"}
        framing = None
        if args.framing_check:
            with torch.no_grad():
                pre = torch.tensor([ids[:pt]], device=tdev)
                o = model(input_ids=pre,
                          attention_mask=torch.ones((1, pt), device=tdev, dtype=torch.long),
                          use_cache=True)
                cache = o.past_key_values
                for step in range(t):
                    o = model(input_ids=torch.tensor([[ids[pt + step]]], device=tdev),
                              attention_mask=torch.ones((1, pt + step + 1), device=tdev,
                                                        dtype=torch.long),
                              past_key_values=cache, use_cache=True)
                    cache = o.past_key_values
                z_inc = o.logits[0, -1, :vocab].float().cpu()
            del o, cache, pre
            torch.cuda.empty_cache()
            framing = {"M_T_full_prefix": M_T_full,
                       "M_T_incremental": float(z_inc[y_T]) - float(z_inc[y_S]),
                       "prompt_len": pt, "decode_steps": t,
                       "M_T_diff": (float(z_inc[y_T]) - float(z_inc[y_S])) - M_T_full,
                       "max_abs_logit_diff": float((z_inc - z_full).abs().max()),
                       "incremental_top1": int(torch.argmax(z_inc).item()),
                       "top1_agrees_across_framings": bool(
                           int(torch.argmax(z_inc).item()) == t_top1),
                       "framing_provenance": "the parent decision-relevance diagnostic produced "
                                             "the frozen y_T through this prefill + cached-decode "
                                             "path; TD captures the attribution in the single "
                                             "full-prefix framing (mechanism sprint framing)"}
            del z_inc

        pk = peak_mib()
        rec = {
            "canonical_source_id": sid, "source_index": r["source_index"],
            "condition": "decision_position", "arm": "decision_position",
            "position": t, "group": r["group"], "prefix_len": L, "prompt_len": pt,
            "prefix_sha256": sha, "prefix_identity_ok": True,
            "y_target": y_T, "y_base": y_S,
            "M_T_base": M_T_full, "M_T_definition": "z_T[y_T] - z_T[y_S]",
            "M_T_framing": "single full-prefix forward (frozen teacher, no intervention)",
            "M_T_reconstruction": float(M_T.detach()),
            "teacher_top1_full_prefix": t_top1,
            "teacher_top1_matches_frozen_y_target": bool(t_top1 == y_T),
            "teacher_top1_matches_frozen_y_base": bool(t_top1 == y_S),
            "M_T_alt_margin_of_yT": float(z_full[y_T])
            - float(z_full.clone().index_fill(
                0, torch.tensor([y_T], device=z_full.device), float("-inf")).max()),
            "score_shape": [len(tlayers), int(score_rows[0].shape[0]), L],
            "score_layer_scope": tlayers,
            "score_finite": bool(all(torch.isfinite(x).all() for x in score_rows)),
            "score_nonzero_fraction": round(
                float(sum((x != 0).float().mean() for x in score_rows) / len(score_rows)), 8),
            "teacher_space_budget_rule": "mean over the 12 Student heads of the frozen "
                                         "per-head L1 budget of the same layer",
            "V_TD_teacher_first_order": v_td_teacher,
            "M_T_under_TD_teacher": M_T_act,
            "dM_T_actual": M_T_act - M_T_full,
            "logp_yT_base": logp_base, "logp_yT_under_TD_teacher": logp_act,
            "dlogp_yT_actual": logp_act - logp_base,
            "top1_under_TD_teacher": top1_act,
            "top1_flip_teacher": bool(top1_act != t_top1),
            "target_token_is_top1_under_TD_teacher": bool(top1_act == y_T),
            "teacher_interventions": n_inter,
            "per_layer": per_layer,
            "grad_diagnostics": grad_diag,
            "reconstruction_vs_full_forward_max_abs_logit_delta": recon,
            "reconstruction_argmax_equal": recon_argmax_equal,
            "observer_cross_check": observer,
            "framing_check": framing,
            "vram": {"peak_allocated_mib": pk, "free_mib_before": free_before,
                     "gpu_total_mib": args.gpu_total_mib,
                     "peak_fraction": {str(k): round(v / args.gpu_total_mib, 4)
                                       for k, v in pk.items()}},
            "wall_seconds": {"pass_a_frozen_forward": round(t_pass_a, 2),
                             "pass_b_grad_capture": round(t_pass_b, 2),
                             "teacher_space_probe": round(t_pass_c, 2),
                             "total": round(time.time() - t0, 2)},
            "device_map": {str(k): str(v) for k, v in dmap.items()},
            "method": "TEACHER_DECISION_ATTRIBUTION_ORACLE_DIAGNOSTIC_NON_DEPLOYABLE",
            "run_config_hash": cfg["config_hash"],
            "checkpoint_sha256": cfg["checkpoint_sha256"],
            "shard": args.shard_name,
        }
        records.append(rec)
        key = f"{sid}|{t}"
        raw[key] = {"score": torch.stack(score_rows, 0).to(torch.float32),
                    "p_teacher": torch.stack(p_teacher, 0).to(torch.float32),
                    "layers": list(tlayers), "y_T": y_T, "y_S": y_S,
                    "prefix_sha256": sha, "prefix_len": L, "position": t,
                    "source_index": int(r["source_index"])}
        if (n + 1) % 8 == 0 or n == len(sel) - 1:
            torch.save(raw, scores_path)
        del p_teacher, score_rows, d_teacher, st, cap, M_T
        torch.cuda.empty_cache()
        H.write_result_envelope(
            body, task_id=TASK_ID, stage="teacher_decision_attribution", records=records,
            arms=["decision_position"], sample_ids=sorted({x["canonical_source_id"]
                                                           for x in records}),
            config_hash=cfg["config_hash"], checkpoint_sha256=cfg["checkpoint_sha256"])
        print(json.dumps({"shard": args.shard_name, "n": n, "src": r["source_index"], "pos": t,
                          "K": L, "M_T": round(M_T_full, 4),
                          "dM_T_actual": round(rec["dM_T_actual"], 5),
                          "V_TD_teacher": round(v_td_teacher, 5),
                          "score_absmax": max(round(float(x.abs().max()), 5)
                                              for x in raw[key]["score"]),
                          "recon": round(recon, 5),
                          "peak_mib": max(pk.values()),
                          "sec": round(rec["wall_seconds"]["total"], 1)}), flush=True)

    H.write_result_envelope(
        body, task_id=TASK_ID, stage="teacher_decision_attribution", records=records,
        arms=["decision_position"], sample_ids=sorted({x["canonical_source_id"]
                                                       for x in records}),
        config_hash=cfg["config_hash"], checkpoint_sha256=cfg["checkpoint_sha256"])
    if raw:
        torch.save(raw, scores_path)
    peaks = {str(i): round(torch.cuda.max_memory_allocated(i) / (1024 * 1024), 1)
             for i in range(torch.cuda.device_count())}
    risk = max(peaks.values()) / args.gpu_total_mib if peaks else 0.0
    meta = {"shard": args.shard_name, "set_name": args.set_name, "positions": len(records),
            "selection": {"set_name": args.set_name,
                          "selection_sha256": (set_spec or {}).get("selection_sha256"),
                          "population": args.population,
                          "selected": [{"sample_id": x["canonical_source_id"],
                                        "position": x["position"],
                                        "prefix_len": x["prefix_len"],
                                        "prefix_sha256": x["prefix_sha256"]}
                                       for x in records]},
            "teacher_layers": tlayers, "device_map": {str(k): str(v) for k, v in dmap.items()},
            "diagnostic_source_diff": diff, "scores_path": str(scores_path),
            "scores_sha256": (sha256_file(scores_path) if scores_path.exists() else None),
            "peak_allocated_mib": peaks, "peak_fraction_of_total": round(risk, 4),
            "vram_risk": bool(risk > args.abort_vram_fraction),
            "wall_seconds": round(time.time() - t_shard, 1), "body": str(body),
            "body_sha256": sha256_file(body),
            "verify": H.verify_result(body, expected_config_hash=cfg["config_hash"],
                                      expected_checkpoint_sha256=cfg["checkpoint_sha256"])}
    H.atomic_write_text(out / f"td_teacher_rows_{args.shard_name}_meta.json",
                        json.dumps(meta, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({"shard": args.shard_name, "done": True, "positions": len(records),
                      "verify": meta["verify"]["status"], "peak_mib": max(peaks.values()),
                      "vram_risk": meta["vram_risk"],
                      "wall_seconds": meta["wall_seconds"]}, ensure_ascii=False), flush=True)
    if meta["verify"]["status"] != "PASS":
        return 1
    return 3 if meta["vram_risk"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
