"""Decision-position measurement pass for
RESCUE-TEACHER-PRIOR-DECISION-RELEVANCE-DIAGNOSTIC-001.

Nothing is trained, no method variable changes.  For every sample of the frozen 128-row
development split this walks the BASE greedy trajectory and, at each decision position,
evaluates the same prefix through A (stock Student), C (full-window Teacher), D
(full-window Null) and T (Target Qwen2-7B), then records the decision-space metrics of
task §4-§12.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dr_common import (DECODE_PHASE, PREFILL_PHASE, build_backend, generate_greedy,
                       jsd_rows, l1_rows, load_config, load_harness, load_modules,
                       logprob_of, maps_for, margin_of, cosine, sha256_file,
                       student_positions, target_positions)

M_FIXED = 4096


def prefix_sha(prompt_ids, toks) -> str:
    return hashlib.sha256(json.dumps([list(map(int, prompt_ids)), list(map(int, toks))],
                                     separators=(",", ":")).encode()).hexdigest()


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
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--shard-lo", type=int, default=0)
    ap.add_argument("--shard-hi", type=int, default=128)
    ap.add_argument("--shard-name", default="d0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    H = load_harness(args.harness_root)
    backend_mod, runner_mod, steering = load_modules(H, args.source_dir, args.source_lock)
    sys.path.insert(0, str(Path(args.k4_dir).resolve()))
    k4 = __import__("real_k4_runner")

    backend = build_backend(backend_mod, args, steering)
    layers = set(backend.student_layers)
    steering_cm = steering.qwen2_adapter_steering
    ns = getattr(steering_cm, "__wrapped__", steering_cm).__globals__
    if "_fit_prior" not in ns:
        raise RuntimeError("harness steering namespace has no _fit_prior")
    fit_prior = ns["_fit_prior"]

    model, device, dmap = k4.build_target(args)
    vocab = min(int(backend.student.config.vocab_size), int(model.config.vocab_size))
    rows = runner_mod.load_rows(Path(args.validation_ledger), Path(args.dataset_path))
    priors = torch.load(Path(args.validation_prior_cache), map_location="cpu",
                        weights_only=True)
    shard_rows = rows[args.shard_lo:args.shard_hi]

    body = out / f"dr_shard_{args.shard_name}.jsonl"
    records: list[dict] = []
    t_shard = time.time()
    for n, row in enumerate(shard_rows):
        sid = row["canonical_source_id"]
        target = backend.prompt_target(row["prompt"], row["answer"])
        prior = priors[sid]
        with torch.no_grad():
            rows_s = backend._student_rows(target)
            mask = torch.tensor(target.attention_mask, device=backend.device)
            mapped, alpha, _ = backend.adapter(prior.to(backend.device), rows_s, mask)
            null_map, null_alpha, _ = backend.adapter(prior.roll(1, dims=-1).to(backend.device),
                                                      rows_s, mask)
        lm, la = maps_for(backend, mapped, alpha)
        nm, na = maps_for(backend, null_map, null_alpha)
        prompt_ids = target.input_ids

        base_tokens = generate_greedy(backend, target, args.max_new_tokens)
        if not base_tokens:
            continue

        zA, natA, _ = student_positions(backend, steering_cm, target, base_tokens, {}, None,
                                        "none", layers, vocab)
        zC, natC, caps = student_positions(backend, steering_cm, target, base_tokens, lm, la,
                                           "full", layers, vocab, capture_rows=True)
        zD, natD, _ = student_positions(backend, steering_cm, target, base_tokens, nm, na,
                                        "full", layers, vocab)
        zT = target_positions(model, device, vocab, prompt_ids, base_tokens)

        gold_ids = backend.student_tokenizer(row["answer"],
                                             add_special_tokens=False)["input_ids"]
        for t in range(len(base_tokens)):
            z_a, z_c, z_d, z_t = zA[t], zC[t], zD[t], zT[t]
            y_T = int(torch.argmax(z_t).item())
            y_S = int(torch.argmax(z_a).item())
            d_teacher = z_c - z_a
            d_null = z_d - z_a
            d_target = z_t - z_a
            M_A = margin_of(z_a, y_T)
            M_C = margin_of(z_c, y_T)
            M_D = margin_of(z_d, y_T)
            logp_A, logp_C = logprob_of(z_a, y_T), logprob_of(z_c, y_T)
            logp_D = logprob_of(z_d, y_T)
            group = "S=T" if y_S == y_T else "S!=T"

            prior_dist = None
            cap = caps[t] if t < len(caps) else None
            if cap:
                tl1, tjsd, nl1, njsd = [], [], [], []
                per_layer = {}
                for li in sorted(layers):
                    row_student = cap.get(li)
                    if row_student is None:
                        continue
                    key = int(row_student.shape[-1])
                    st_row = row_student.float().cpu()
                    p_t = fit_prior(lm[li], key, "cpu", backend.eps).float().cpu()
                    p_n = fit_prior(nm[li], key, "cpu", backend.eps).float().cpu()
                    tl1.append(l1_rows(st_row, p_t))
                    tjsd.append(jsd_rows(st_row, p_t))
                    nl1.append(l1_rows(st_row, p_n))
                    njsd.append(jsd_rows(st_row, p_n))
                    per_layer[str(li)] = {"teacher_l1": tl1[-1], "teacher_jsd": tjsd[-1],
                                          "null_l1": nl1[-1], "null_jsd": njsd[-1],
                                          "key_len": key}
                if tl1:
                    prior_dist = {"teacher_l1": sum(tl1) / len(tl1),
                                  "teacher_jsd": sum(tjsd) / len(tjsd),
                                  "null_l1": sum(nl1) / len(nl1),
                                  "null_jsd": sum(njsd) / len(njsd),
                                  "layers": per_layer,
                                  "space": "harness pre-mix last-query-row vs _fit_prior(row) at the same key length"}

            gold = {"available": False, "reason": "generated prefix is not a token-prefix of the gold answer"}
            if t < len(gold_ids) and list(base_tokens[:t]) == list(gold_ids[:t]):
                y_g = int(gold_ids[t])
                gold = {"available": True, "token": y_g,
                        "logp_A": logprob_of(z_a, y_g), "logp_C": logprob_of(z_c, y_g),
                        "logp_D": logprob_of(z_d, y_g),
                        "M_A": margin_of(z_a, y_g), "M_C": margin_of(z_c, y_g),
                        "M_D": margin_of(z_d, y_g)}
                gold["dlogp_teacher"] = gold["logp_C"] - gold["logp_A"]
                gold["dlogp_null"] = gold["logp_D"] - gold["logp_A"]
                gold["dM_teacher"] = gold["M_C"] - gold["M_A"]
                gold["dM_null"] = gold["M_D"] - gold["M_A"]

            rec = {
                "canonical_source_id": sid, "source_index": row["source_index"],
                "condition": "decision_position", "arm": "decision_position",
                "position": t, "prefix_len": len(prompt_ids) + t,
                "prefix_sha256": prefix_sha(prompt_ids, base_tokens[:t]),
                "base_tokens": list(map(int, base_tokens)),
                "y_base": y_S, "y_target": y_T, "group": group,
                "natural_top1_C": natC[t], "natural_top1_D": natD[t],
                "natural_C_follows_base_up_to_here": bool(list(natC[:t]) == list(base_tokens[:t])),
                "natural_D_follows_base_up_to_here": bool(list(natD[:t]) == list(base_tokens[:t])),
                "logp_A_yT": logp_A, "logp_C_yT": logp_C, "logp_D_yT": logp_D,
                "dlogp_T_teacher": logp_C - logp_A, "dlogp_T_null": logp_D - logp_A,
                "M_A": M_A, "M_C": M_C, "M_D": M_D,
                "dM_teacher": M_C - M_A, "dM_null": M_D - M_A,
                "dz_teacher_l2": float(d_teacher.norm()), "dz_null_l2": float(d_null.norm()),
                "dz_teacher_max_abs": float(d_teacher.abs().max()),
                "dz_null_max_abs": float(d_null.abs().max()),
                "dz_target_l2": float(d_target.norm()),
                "cos_teacher": cosine(d_teacher, d_target),
                "cos_null": cosine(d_null, d_target),
                "cos_teacher_topM": cosine(d_teacher[:M_FIXED], d_target[:M_FIXED]),
                "cos_null_topM": cosine(d_null[:M_FIXED], d_target[:M_FIXED]),
                "linearized_scale_to_boundary_teacher": (
                    (-M_A) / (M_C - M_A) if (group == "S!=T" and (M_C - M_A) > 0) else None),
                "linearized_label": "LINEARIZED_DIAGNOSTIC_ONLY",
                "prior_distance": prior_dist,
                "gold": gold,
                "topM": M_FIXED,
                "run_config_hash": cfg["config_hash"],
                "checkpoint_sha256": cfg["checkpoint_sha256"],
                "shard": args.shard_name,
            }
            records.append(rec)
        H.write_result_envelope(
            body, task_id="RESCUE-TEACHER-PRIOR-DECISION-RELEVANCE-DIAGNOSTIC-001",
            stage="decision_relevance", records=records,
            arms=["A_base", "C_full_window_teacher", "D_full_window_null", "T_target"],
            sample_ids=sorted({r["canonical_source_id"] for r in records}),
            config_hash=cfg["config_hash"], checkpoint_sha256=cfg["checkpoint_sha256"])

    meta = {"shard": args.shard_name, "rows": len(shard_rows), "records": len(records),
            "samples": len({r["canonical_source_id"] for r in records}),
            "target_device_map": dmap, "target_dtype": args.target_dtype,
            "vocab_limit": vocab, "topM": M_FIXED,
            "harness_manifest_sha256": cfg["harness_manifest_sha256"],
            "config_hash": cfg["config_hash"], "checkpoint_sha256": cfg["checkpoint_sha256"],
            "shard_lo": args.shard_lo, "shard_hi": args.shard_hi,
            "wall_seconds": round(time.time() - t_shard, 1),
            "body": str(body), "body_sha256": sha256_file(body),
            "verify": H.verify_result(body, expected_config_hash=cfg["config_hash"],
                                      expected_checkpoint_sha256=cfg["checkpoint_sha256"])}
    H.atomic_write_text(out / f"dr_shard_{args.shard_name}_meta.json",
                        json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    return 0 if meta["verify"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
