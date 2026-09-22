"""Four-arm QA evaluation — one shard per GPU process.

Greedy decoding, identical prompts / tokenizer / max_new_tokens for every arm.
Each shard writes a frozen-V3 result envelope (atomic write + body-bound completion
marker); the aggregator refuses anything without a verified marker.

Usage:
  four_arm_qa.py --harness-root H --config CFG --source-dir S --source-lock SL \
      --teacher-model TM --student-model SM --dataset-path DS --validation-ledger V \
      --validation-prior-cache P --checkpoint CK --device cuda:0 \
      --shard-lo 0 --shard-hi 32 --shard-name q0 --out OUTDIR
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from four_arm_common import (ARMS, ARM_POLICY, DECODE_PHASE, PREFILL_PHASE,  # noqa: E402
                            build_backend, cm_for, em_score, f1_score, load_config,
                            load_harness, load_modules, maps_for)


def generate(backend, steering_cm, target, prior, alpha, window, layers, max_new_tokens):
    """Greedy generation under one arm policy; returns tokens, text, intervention counts."""
    lm, la = maps_for(backend, prior, alpha)
    counts = {"prefill_mix_calls": 0, "decode_mix_calls": 0, "decode_steps": 0,
              "decode_steered_steps": 0}
    with torch.no_grad():
        with cm_for(steering_cm, backend, lm, la, layers, PREFILL_PHASE) as st:
            pf = backend.student(**backend._ids(target), use_cache=True,
                                 output_attentions=False)
            counts["prefill_mix_calls"] = 0 if st is None else len(st.mix_log)
        past = pf.past_key_values
        nxt = int(torch.argmax(pf.logits[:, -1, :], dim=-1).item())
        toks = []
        for _ in range(max_new_tokens):
            if backend.student_tokenizer.eos_token_id is not None and \
                    nxt == backend.student_tokenizer.eos_token_id:
                break
            toks.append(nxt)
            ids = torch.tensor([[nxt]], device=backend.device, dtype=torch.long)
            mask = torch.ones((1, len(target.input_ids) + len(toks)),
                              device=backend.device, dtype=torch.long)
            if window == "full" and lm:
                with cm_for(steering_cm, backend, lm, la, layers, DECODE_PHASE) as st:
                    o = backend.student(input_ids=ids, attention_mask=mask,
                                        past_key_values=past, use_cache=True,
                                        output_attentions=False)
                    counts["decode_mix_calls"] += len(st.mix_log)
                counts["decode_steered_steps"] += 1
            else:
                o = backend.student(input_ids=ids, attention_mask=mask,
                                    past_key_values=past, use_cache=True,
                                    output_attentions=False)
            past = o.past_key_values
            nxt = int(torch.argmax(o.logits[:, -1, :], dim=-1).item())
            counts["decode_steps"] += 1
    text = backend.student_tokenizer.decode(toks, skip_special_tokens=True).strip()
    return toks, text, counts


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
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--shard-lo", type=int, default=0)
    ap.add_argument("--shard-hi", type=int, default=128)
    ap.add_argument("--shard-name", default="q0")
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
    shard_rows = rows[args.shard_lo:args.shard_hi]

    body = out / f"qa_shard_{args.shard_name}.jsonl"
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
        priors_by_arm = {"teacher": (mapped, alpha), "null": (null_map, null_alpha)}
        base_tokens = None
        for arm in ARMS:
            pol = ARM_POLICY[arm]
            pr, al = (None, None) if pol["prior"] is None else priors_by_arm[pol["prior"]]
            t0 = time.time()
            toks, text, counts = generate(backend, steering_cm, target, pr, al,
                                          pol["window"], layers, args.max_new_tokens)
            if arm == "A_base":
                base_tokens = toks
            records.append({
                "canonical_source_id": sid,
                "source_index": row["source_index"],
                "condition": arm,
                "arm": arm,
                "tokens": toks,
                "text": text,
                "gold": row["answer"],
                "em": em_score(text, row["answer"]),
                "f1": f1_score(text, row["answer"]),
                "n_tokens": len(toks),
                "first_divergence_from_base": (
                    None if arm == "A_base" or base_tokens is None
                    else next((i for i, (a, b) in enumerate(zip(toks, base_tokens)) if a != b),
                              None if len(toks) == len(base_tokens) else
                              min(len(toks), len(base_tokens)))),
                "identical_to_base": (None if arm == "A_base" or base_tokens is None
                                      else bool(toks == base_tokens)),
                "prompt_len": len(target.input_ids),
                "window": pol["window"],
                "prior_kind": pol["prior"],
                "interventions": counts,
                "run_config_hash": cfg["config_hash"],
                "checkpoint_sha256": cfg["checkpoint_sha256"],
                "shard": args.shard_name,
                "wall_seconds": round(time.time() - t0, 3),
            })
            print(json.dumps({"shard": args.shard_name, "row": n, "src": row["source_index"],
                              "arm": arm, "em": records[-1]["em"],
                              "n_tok": len(toks),
                              "mix": counts["prefill_mix_calls"] + counts["decode_mix_calls"]},
                             ensure_ascii=False), flush=True)
        H.write_result_envelope(
            body, task_id="RESCUE-MINIMAL-FOUR-ARM-VALID-RERUN-001", stage="qa_four_arm",
            records=records, arms=list(ARMS),
            sample_ids=sorted({r["canonical_source_id"] for r in records}),
            config_hash=cfg["config_hash"], checkpoint_sha256=cfg["checkpoint_sha256"])
    meta = {"shard": args.shard_name, "rows": len(shard_rows), "records": len(records),
            "arms": list(ARMS), "device": args.device, "max_new_tokens": args.max_new_tokens,
            "harness_manifest_sha256": cfg["harness_manifest_sha256"],
            "config_hash": cfg["config_hash"], "checkpoint_sha256": cfg["checkpoint_sha256"],
            "shard_lo": args.shard_lo, "shard_hi": args.shard_hi,
            "wall_seconds": round(time.time() - t_shard, 1),
            "body": str(body), "body_sha256": H.sha256_file(body),
            "verify": H.verify_result(body, expected_config_hash=cfg["config_hash"],
                                      expected_checkpoint_sha256=cfg["checkpoint_sha256"])}
    H.atomic_write_text(out / f"qa_shard_{args.shard_name}_meta.json",
                        json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"shard": args.shard_name, "done": True, "records": len(records),
                      "verify": meta["verify"]["status"],
                      "wall_seconds": meta["wall_seconds"]}, ensure_ascii=False), flush=True)
    return 0 if meta["verify"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
