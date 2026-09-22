"""TRACK B — Teacher representation decomposition (prompt-level, matching the frozen prior).

RESCUE-GUIDANCE-REPRESENTATION-MECHANISM-SPRINT-001.

The frozen method's Teacher prior is a PROMPT-level object (shape [4,28,K_prompt] in Student
token coordinates) that is padded + renormalised to each decision position's key length by the
frozen `_fit_prior` at injection time.  This track therefore extracts, per sample and on the same
prompt:

  R0  TEACHER_ATTENTION           the frozen cached prior (current method baseline)
  R1  TEACHER_ATTN_VALUE_NORM     attention_j × ||V_j||, normalised per layer/head
  R2  TEACHER_CONTEXT_CONTRIBUTION  ||attention_j · V_j|| as an importance proxy

Teacher attention rows come from the frozen `qwen2_teacher_row_observer`; ||V_j|| comes from a
read-only forward hook on `v_proj` (RoPE does not touch V, so the norm is exact).  R1 is projected
to Student coordinates with the UNCHANGED frozen `project_teacher_heads_to_student`, and both R0
and R1 are pushed through the UNCHANGED frozen Adapter with the real Student attention rows.

§B4 (Teacher decision attribution) is optional and gated: it runs as a bounded one-position
feasibility probe at the end and is classified FEASIBLE or BLOCKED_RESOURCE_OPTIONAL without
changing dtype, model or context.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dr_common import (build_backend, jsd_rows, l1_rows, load_config, load_harness,  # noqa: E402
                       load_modules, sha256_file)

TOP_K = 16
TEACHER_LAYER_SCOPE = "last_4_layers"


def entropy_rows(x: torch.Tensor, eps: float = 1e-12) -> float:
    p = x.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    return float((-(p * p.log()).sum(dim=-1)).mean())


def prep(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x)).float().clamp_min(0.0)
    return x / x.sum(dim=-1, keepdim=True).clamp_min(eps)


def topk_sets(x: torch.Tensor, k: int):
    """Top-k index sets per head, over any leading (layer, head) shape."""
    p = prep(x).reshape(-1, x.shape[-1])
    kk = min(k, p.shape[-1])
    mass, sets = [], []
    for h in range(p.shape[0]):
        v, i = torch.topk(p[h], kk)
        mass.append(float(v.sum()))
        sets.append(set(int(t) for t in i.tolist()))
    return sum(mass) / len(mass), sets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-root", required=True)
    ap.add_argument("--config", required=True)
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
    ap.add_argument("--shard-name", default="b0")
    ap.add_argument("--attribution-probe", action="store_true")
    ap.add_argument("--limit-samples", type=int, default=0)
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

    backend = build_backend(backend_mod, args, steering)
    eps = backend.eps
    vocab = int(backend.student.config.vocab_size)
    fit_prior = getattr(steering.qwen2_adapter_steering, "__wrapped__",
                        steering.qwen2_adapter_steering).__globals__["_fit_prior"]
    prior_cache = torch.load(Path(args.validation_prior_cache), map_location="cpu",
                             weights_only=True)

    model, device, dmap = __import__("real_k4_runner").build_target(args)
    teacher_layers = list(RHT.selected_layer_indices(int(model.config.num_hidden_layers),
                                                     TEACHER_LAYER_SCOPE))
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_model, local_files_only=True,
                                                use_fast=True)
    student_tok = backend.student_tokenizer

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

    rows_src = runner_mod.load_rows(Path(args.validation_ledger), Path(args.dataset_path))
    by_id = {r["canonical_source_id"]: r for r in rows_src}
    sids = sorted(prior_cache.keys())
    shard = sids[args.shard_lo:args.shard_hi] if args.shard_hi else sids
    if args.limit_samples:
        shard = shard[:args.limit_samples]

    body = out / f"tb_shard_{args.shard_name}.jsonl"
    records: list[dict] = []
    t_shard = time.time()
    for n, sid in enumerate(shard):
        row = by_id[sid]
        prompt = row["prompt"]
        t_enc, t_off = RHT.encode_with_offsets(teacher_tok, prompt, device=None)
        _, s_off = RHT.encode_with_offsets(student_tok, prompt, device=None)
        target = backend.prompt_target(prompt, row["answer"])
        with torch.no_grad():
            rs_rows = backend._student_rows(target)
            smask = torch.tensor(target.attention_mask, device=backend.device)
        K_student = int(rs_rows.shape[-1])

        attn_caps: dict = {}
        vstore: dict = {}
        handles = [model.model.layers[L].self_attn.register_forward_hook(v_hook(vstore),
                                                                        with_kwargs=True)
                   for L in teacher_layers]
        try:
            t_ids = t_enc["input_ids"].to(device)
            with torch.no_grad():
                with qwen2_teacher_row_observer(model.model, teacher_layers, attn_caps):
                    _ = model(input_ids=t_ids, attention_mask=torch.ones_like(t_ids),
                              use_cache=False)
            del _
            torch.cuda.empty_cache()
        finally:
            for h in handles:
                h.remove()

        r0 = prep(prior_cache[sid].float())                     # [4,28,K_student] frozen prior
        r1_layers, r2_stats = [], {}
        for li, L in enumerate(teacher_layers):
            a = attn_caps.get(L)
            v = vstore.get(L)
            if a is None or v is None:
                continue
            a = a.float()
            n = min(a.shape[-1], v.shape[-1])
            a, v = a[:, :n], v[:, :n]
            score = a * v                                       # R2 = ||attn_j * V_j||
            r1 = prep(score)
            r1_layers.append(r1)
            r2_stats[str(L)] = {"score_sum": float(score.sum()),
                                "score_mean": float(score.mean()),
                                "attn_entropy": entropy_rows(prep(a)),
                                "vnorm_mean": float(v.mean()),
                                "vnorm_std": float(v.std())}
        R1_teacher = torch.stack(r1_layers, 0) if r1_layers else None
        # the frozen projection takes one [heads, K] layer row at a time
        proj1 = (torch.stack([RHT.project_teacher_heads_to_student(R1_teacher[li], t_off,
                                                                  s_off, eps)
                              for li in range(R1_teacher.shape[0])], 0)
                 if R1_teacher is not None else None)

        def through_adapter(P):
            P_fit = torch.stack([fit_prior(P[i], K_student, "cpu", eps)
                                 for i in range(P.shape[0])])
            m, a, _e = backend.adapter(P_fit.to(backend.device), rs_rows, smask)
            return m.detach().float().cpu(), a.detach().float().cpu()

        rec = {"canonical_source_id": sid, "condition": "decision_position",
               "arm": "decision_position", "source_index": row.get("source_index"),
               "prompt_len": K_student, "representations": {}}
        m0, a0 = through_adapter(r0)
        rec["representations"]["R0_TEACHER_ATTENTION"] = {
            "shape": list(r0.shape), "alpha_mean": float(a0.mean()),
            "entropy": entropy_rows(r0), "topK_mass": topk_sets(r0, TOP_K)[0],
            "l1_vs_student_attention": l1_rows(m0, rs_rows.float().cpu()),
            "jsd_vs_student_attention": jsd_rows(m0, rs_rows.float().cpu()),
            "finite": bool(torch.isfinite(m0).all()), "non_negative": bool((m0 >= 0).all()),
        }
        if proj1 is not None:
            _, set1 = topk_sets(proj1, TOP_K)
            _, set0 = topk_sets(r0, TOP_K)
            inter = sum(len(set1[h] & set0[h]) for h in range(len(set1)))
            union = sum(len(set1[h] | set0[h]) for h in range(len(set1)))
            m1, a1 = through_adapter(proj1)
            rec["representations"]["R1_TEACHER_ATTN_VALUE_NORM"] = {
                "teacher_space_shape": list(R1_teacher.shape),
                "projected_shape": list(proj1.shape),
                "alpha_mean": float(a1.mean()),
                "entropy": entropy_rows(proj1), "topK_mass": topk_sets(proj1, TOP_K)[0],
                "topK_jaccard_vs_R0": (inter / union) if union else None,
                "l1_vs_R0": l1_rows(proj1, r0), "jsd_vs_R0": jsd_rows(proj1, r0),
                "l1_vs_student_attention": l1_rows(m1, rs_rows.float().cpu()),
                "jsd_vs_student_attention": jsd_rows(m1, rs_rows.float().cpu()),
                "finite": bool(torch.isfinite(m1).all()), "non_negative": bool((m1 >= 0).all()),
            }
            rec["representations"]["R2_TEACHER_CONTEXT_CONTRIBUTION"] = {
                "status": "IDENTICAL_TO_R1_BEFORE_NORMALISATION",
                "identity": "||attention_j * V_j|| == attention_j * ||V_j||",
                "per_layer": r2_stats,
            }
            rec["R1_minus_R0_entropy"] = (rec["representations"][
                "R1_TEACHER_ATTN_VALUE_NORM"]["entropy"]
                - rec["representations"]["R0_TEACHER_ATTENTION"]["entropy"])
        records.append(rec)
        H.write_result_envelope(
            body, task_id="RESCUE-GUIDANCE-REPRESENTATION-MECHANISM-SPRINT-001",
            stage="teacher_representation", records=records, arms=["decision_position"],
            sample_ids=sorted({r["canonical_source_id"] for r in records}),
            config_hash=cfg["config_hash"], checkpoint_sha256=cfg["checkpoint_sha256"])
        print(json.dumps({"shard": args.shard_name, "n": n, "src": row.get("source_index"),
                          "prompt_len": K_student}), flush=True)

    attribution = {"attempted": False}
    if args.attribution_probe and shard:
        sid = shard[0]
        row = by_id[sid]
        try:
            t_enc, _ = RHT.encode_with_offsets(teacher_tok, row["prompt"], device=device)
            emb = model.model.embed_tokens(t_enc["input_ids"]).detach().requires_grad_(True)
            with torch.enable_grad():
                o = model(inputs_embeds=emb,
                          attention_mask=torch.ones_like(t_enc["input_ids"]), use_cache=False)
                z = o.logits[0, -1, :vocab].float()
                y_T = int(torch.argmax(z).item())
                M_T = z[y_T] - float(z.clone().index_fill(
                    0, torch.tensor([y_T]), float("-inf")).max())
                M_T.backward()
            attribution = {"attempted": True, "status": "FEASIBLE",
                           "M_T_teacher_side": float(M_T.detach()),
                           "emb_grad_abs_sum": float(emb.grad.abs().sum()),
                           "note": "one-position feasibility probe only; full attribution is "
                                   "out of scope for this sprint"}
            del o, z, emb
        except Exception as exc:
            attribution = {"attempted": True, "status": "BLOCKED_RESOURCE_OPTIONAL",
                           "error": f"{type(exc).__name__}: {exc}",
                           "note": "7B backward does not fit under the frozen setup; dtype / "
                                   "model / context were NOT changed"}
        torch.cuda.empty_cache()

    meta = {"shard": args.shard_name, "samples": len(shard), "records": len(records),
            "teacher_layers": teacher_layers, "top_k": TOP_K,
            "granularity": "prompt-level (the frozen prior's own granularity); each candidate is "
                           "fitted to every decision position with the unchanged _fit_prior",
            "target_device_map": dmap, "attribution_probe": attribution,
            "wall_seconds": round(time.time() - t_shard, 1), "body": str(body),
            "body_sha256": sha256_file(body),
            "verify": H.verify_result(body, expected_config_hash=cfg["config_hash"],
                                      expected_checkpoint_sha256=cfg["checkpoint_sha256"])}
    H.atomic_write_text(out / f"tb_shard_{args.shard_name}_meta.json",
                        json.dumps(meta, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({"shard": args.shard_name, "done": True, "records": len(records),
                      "verify": meta["verify"]["status"], "attribution": attribution,
                      "wall_seconds": meta["wall_seconds"]}, ensure_ascii=False), flush=True)
    return 0 if meta["verify"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
