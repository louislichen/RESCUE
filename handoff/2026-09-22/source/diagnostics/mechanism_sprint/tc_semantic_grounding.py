"""TRACK C — semantic evidence grounding.

RESCUE-GUIDANCE-REPRESENTATION-MECHANISM-SPRINT-001.

Maps HotpotQA supporting facts onto token spans by character offsets and asks, for every valid
decision position, how much probability mass each guidance representation puts on the supporting
evidence:

  R0  TEACHER_ATTENTION            frozen cached prior        (Student token space)
  R1  TEACHER_ATTN_VALUE_NORM      attention_j × ||V_j||      (Teacher token space)
  R2  TEACHER_CONTEXT_CONTRIBUTION identical to R1 before normalisation (reported once)
  N   Position-Alignment-Null      roll(1, -1) of the same row in each space
  S   Student attention row        the Student's own row      (Student token space)

Each representation is measured in its own token space, so the evidence mask is built twice: once
with the Student tokenizer offsets and once with the Teacher tokenizer offsets.  A supporting fact
that cannot be located verbatim in the prompt is excluded with an explicit reason and only from
this track.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dr_common import build_backend, load_config, load_harness, load_modules, sha256_file  # noqa: E402

TOP_K = 16


def prep(x, eps=1e-12):
    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x)).float().clamp_min(0.0)
    return x / x.sum(dim=-1, keepdim=True).clamp_min(eps)


def evidence_metrics(row2d: torch.Tensor, mask: torch.Tensor, k: int = TOP_K):
    """row2d [heads, K], mask [K] bool -> grounding metrics."""
    p = prep(row2d)
    K = p.shape[-1]
    m = mask[:K]
    n_ev = int(m.sum())
    if n_ev == 0 or n_ev == K:
        return None
    ev = p[:, m]
    ne = p[:, ~m]
    mass_ev = float(ev.sum(dim=-1).mean())
    mass_ne = float(ne.sum(dim=-1).mean())
    kk = min(k, K)
    recall, frac_ev_topk = [], []
    for h in range(p.shape[0]):
        _, idx = torch.topk(p[h], kk)
        hit = int(m[idx].sum())
        recall.append(hit / max(1, min(kk, n_ev)))
        frac_ev_topk.append(hit / kk)
    ent_ev = float((-(prep(ev) * prep(ev).log()).sum(dim=-1)).mean()) if n_ev > 1 else 0.0
    return {"evidence_mass": mass_ev, "non_evidence_mass": mass_ne,
            "evidence_to_non_evidence_ratio": mass_ev / mass_ne if mass_ne > 0 else None,
            "topk_evidence_recall": sum(recall) / len(recall),
            "topk_evidence_fraction": sum(frac_ev_topk) / len(frac_ev_topk),
            "entropy_within_evidence": ent_ev,
            "evidence_token_count": n_ev, "key_len": K}


def spans_to_mask(spans, offsets, K):
    mask = torch.zeros(K, dtype=torch.bool)
    for (a, b) in spans:
        for i, (s, e) in enumerate(offsets[:K]):
            if e > a and s < b:
                mask[i] = True
    return mask


def locate(text: str, needle: str):
    i = text.find(needle)
    if i < 0:
        return None
    return (i, i + len(needle))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--k4-dir", required=True)
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
    ap.add_argument("--target-max-memory", default="8GiB")
    ap.add_argument("--target-dtype", default="bfloat16")
    ap.add_argument("--shard-lo", type=int, default=0)
    ap.add_argument("--shard-hi", type=int, default=0)
    ap.add_argument("--shard-name", default="c0")
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
    from datasets import load_from_disk

    backend = build_backend(backend_mod, args, steering)
    prior_cache = torch.load(Path(args.validation_prior_cache), map_location="cpu",
                             weights_only=True)
    model, device, dmap = __import__("real_k4_runner").build_target(args)
    teacher_layers = list(RHT.selected_layer_indices(int(model.config.num_hidden_layers),
                                                     "last_4_layers"))
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_model, local_files_only=True,
                                                use_fast=True)
    student_tok = backend.student_tokenizer
    ds = load_from_disk(args.dataset_path)
    all_rows = runner_mod.load_rows(Path(args.validation_ledger), Path(args.dataset_path))
    by_id = {r["canonical_source_id"]: r for r in all_rows}

    import csv as _csv
    with Path(args.validation_ledger).open(encoding="utf-8-sig", newline="") as f:
        ledger = list(_csv.DictReader(f))
    dr_rows = []
    for line in Path(args.dr_rows).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if "__result_header__" not in r:
                dr_rows.append(r)
    groups_by_sample = {}
    for r in dr_rows:
        groups_by_sample.setdefault(r["canonical_source_id"], set()).add(r["group"])
    sids = [row["canonical_source_id"] for row in ledger]
    order = {s: i for i, s in enumerate(sids)}
    sids.sort(key=lambda s: order[s])
    shard = sids[args.shard_lo:args.shard_hi] if args.shard_hi else sids
    if args.limit_samples:
        shard = shard[:args.limit_samples]

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

    body = out / f"tc_shard_{args.shard_name}.jsonl"
    records: list[dict] = []
    excluded: list[dict] = []
    t0 = time.time()
    for n, sid in enumerate(shard):
        li = order[sid]
        lrow = ledger[li]
        srow = ds[int(lrow["source_index"])]
        row = by_id[sid]
        prompt = row["prompt"]
        sf = srow.get("supporting_facts") or {}
        ctx = srow.get("context") or {}
        titles = list(ctx.get("title") or [])
        sentences = list(ctx.get("sentences") or [])
        spans, reasons = [], []
        for t, sent_id in zip(list(sf.get("title") or []), list(sf.get("sent_id") or [])):
            if t not in titles:
                reasons.append(f"title_not_in_context:{t}")
                continue
            sents = sentences[titles.index(t)]
            if int(sent_id) >= len(sents):
                reasons.append(f"sent_id_out_of_range:{t}:{sent_id}")
                continue
            sp = locate(prompt, sents[int(sent_id)])
            if sp is None:
                reasons.append(f"supporting_sentence_not_found_verbatim:{t}:{sent_id}")
                continue
            spans.append(sp)
        if not spans:
            excluded.append({"canonical_source_id": sid, "reasons": reasons[:5],
                             "reason": "no supporting fact could be mapped to the prompt"})
            continue

        t_enc, t_off = RHT.encode_with_offsets(teacher_tok, prompt, device=None)
        s_enc, s_off = RHT.encode_with_offsets(student_tok, prompt, device=None)
        K_t = int(t_enc["input_ids"].shape[-1])
        K_s = len(s_off)
        mask_t = spans_to_mask(spans, t_off, K_t)
        mask_s = spans_to_mask(spans, s_off, K_s)

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
        finally:
            for h in handles:
                h.remove()
        torch.cuda.empty_cache()

        target = backend.prompt_target(prompt, row["answer"])
        with torch.no_grad():
            student_rows = backend._student_rows(target).detach().float().cpu()
            rs_mask_t = torch.tensor(target.attention_mask)

        r0 = prep(prior_cache[sid].float())                       # [4,28,K_s]
        r0_null = prep(prior_cache[sid].float().roll(1, dims=-1))
        rec = {"canonical_source_id": sid, "condition": "decision_position",
               "arm": "decision_position", "source_index": lrow.get("source_index"),
               "prompt_len_student": K_s, "prompt_len_teacher": K_t,
               "evidence_spans_chars": spans, "evidence_token_count_student": int(mask_s.sum()),
               "evidence_token_count_teacher": int(mask_t.sum()),
               "excluded_facts": reasons[:10], "representations": {}}
        m = evidence_metrics(r0.reshape(-1, K_s), mask_s)
        if m:
            rec["representations"]["R0_TEACHER_ATTENTION_student_space"] = m
        m = evidence_metrics(r0_null.reshape(-1, K_s), mask_s)
        if m:
            rec["representations"]["NULL_student_space"] = m
        m = evidence_metrics(student_rows.reshape(-1, K_s), mask_s)
        if m:
            rec["representations"]["STUDENT_ATTENTION_student_space"] = m

        r1_layers = []
        for L in teacher_layers:
            a, v = attn_caps.get(L), vstore.get(L)
            if a is None or v is None:
                continue
            a = a.float()
            nn = min(a.shape[-1], v.shape[-1], K_t)
            r1_layers.append(prep(a[:, :nn] * v[:, :nn]))
        if r1_layers:
            R1 = torch.stack(r1_layers, 0)
            m = evidence_metrics(R1.reshape(-1, R1.shape[-1]), mask_t)
            if m:
                rec["representations"]["R1_TEACHER_ATTN_VALUE_NORM_teacher_space"] = m
                rec["representations"]["R2_TEACHER_CONTEXT_CONTRIBUTION"] = {
                    "status": "IDENTICAL_TO_R1_BEFORE_NORMALISATION",
                    "identity": "||attention_j * V_j|| == attention_j * ||V_j||"}
            trow = torch.stack([prep(attn_caps[L][:, :min(attn_caps[L].shape[-1], K_t)].float())
                                for L in teacher_layers if L in attn_caps], 0)
            m = evidence_metrics(trow.reshape(-1, trow.shape[-1]), mask_t)
            if m:
                rec["representations"]["R0_TEACHER_ATTENTION_teacher_space"] = m
            tnull = torch.stack([prep(attn_caps[L][:, :min(attn_caps[L].shape[-1], K_t)]
                                      .float().roll(1, dims=-1))
                                 for L in teacher_layers if L in attn_caps], 0)
            m = evidence_metrics(tnull.reshape(-1, tnull.shape[-1]), mask_t)
            if m:
                rec["representations"]["NULL_teacher_space"] = m
        rec["decision_positions"] = sorted(groups_by_sample.get(sid, []))
        records.append(rec)
        H.write_result_envelope(
            body, task_id="RESCUE-GUIDANCE-REPRESENTATION-MECHANISM-SPRINT-001",
            stage="semantic_grounding", records=records, arms=["decision_position"],
            sample_ids=sorted({r["canonical_source_id"] for r in records}),
            config_hash=cfg["config_hash"], checkpoint_sha256=cfg["checkpoint_sha256"])
        print(json.dumps({"shard": args.shard_name, "n": n, "src": lrow.get("source_index"),
                          "facts": len(spans), "excluded_facts": len(reasons)}), flush=True)

    H.write_result_envelope(
        body, task_id="RESCUE-GUIDANCE-REPRESENTATION-MECHANISM-SPRINT-001",
        stage="semantic_grounding", records=records, arms=["decision_position"],
        sample_ids=sorted({r["canonical_source_id"] for r in records}),
        config_hash=cfg["config_hash"], checkpoint_sha256=cfg["checkpoint_sha256"])
    meta = {"shard": args.shard_name, "samples": len(shard), "records": len(records),
            "excluded_samples": len(excluded), "exclusion_examples": excluded[:10],
            "teacher_layers": teacher_layers, "top_k": TOP_K,
            "spaces": "each representation is measured in its own token space with a mask built "
                      "from that tokenizer's character offsets",
            "wall_seconds": round(time.time() - t0, 1), "body": str(body),
            "body_sha256": sha256_file(body),
            "verify": H.verify_result(body, expected_config_hash=cfg["config_hash"],
                                      expected_checkpoint_sha256=cfg["checkpoint_sha256"])}
    H.atomic_write_text(out / f"tc_shard_{args.shard_name}_meta.json",
                        json.dumps(meta, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({"shard": args.shard_name, "done": True, "records": len(records),
                      "excluded": len(excluded), "verify": meta["verify"]["status"],
                      "wall_seconds": meta["wall_seconds"]}, ensure_ascii=False), flush=True)
    return 0 if meta["verify"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
