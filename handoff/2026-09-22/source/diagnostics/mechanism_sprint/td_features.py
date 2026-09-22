"""TRACK D feature completion — Student output entropy and margin geometry per decision position.

The parent diagnostic recorded the margins and log-probabilities but not the Student output
entropy.  This short pass adds it (and re-checks the margins) with the frozen arm-A framing
(no hook at all), one forward per decision position.  No new experiment: same prefixes, same
tokens, same model.
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
from dr_common import build_backend, load_config, load_harness, load_modules, sha256_file  # noqa: E402


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
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard-lo", type=int, default=0)
    ap.add_argument("--shard-hi", type=int, default=0)
    ap.add_argument("--shard-name", default="d0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    H = load_harness(args.harness_root)
    backend_mod, runner_mod, steering = load_modules(H, args.source_dir, args.source_lock)
    backend = build_backend(backend_mod, args, steering)
    vocab = int(backend.student.config.vocab_size)
    rows_src = runner_mod.load_rows(Path(args.validation_ledger), Path(args.dataset_path))
    by_id = {r["canonical_source_id"]: r for r in rows_src}

    dr = []
    for line in Path(args.dr_rows).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if "__result_header__" not in r:
                dr.append(r)
    dr.sort(key=lambda r: (int(r["source_index"]), r["position"]))
    shard = dr[args.shard_lo:args.shard_hi] if args.shard_hi else dr

    body = out / f"td_shard_{args.shard_name}.jsonl"
    records: list[dict] = []
    t0 = time.time()
    for n, r in enumerate(shard):
        row = by_id[r["canonical_source_id"]]
        target = backend.prompt_target(row["prompt"], row["answer"])
        base = list(map(int, r["base_tokens"]))
        t = int(r["position"])
        prefix = list(map(int, target.input_ids)) + base[:t]
        ids = torch.tensor([prefix], device=backend.device)
        with torch.no_grad():
            o = backend.student(input_ids=ids, attention_mask=torch.ones_like(ids),
                                use_cache=False, output_attentions=False)
            z = o.logits[0, -1, :vocab].float()
            del o
        torch.cuda.empty_cache()
        logp = F.log_softmax(z, dim=-1)
        p = logp.exp()
        y_T, y_S = int(r["y_target"]), int(r["y_base"])
        zc = z.clone()
        top2 = torch.topk(zc, 2).values
        zc[y_T] = float("-inf")
        margin_vs_rest = float(z[y_T] - zc.max())
        records.append({
            "canonical_source_id": r["canonical_source_id"],
            "source_index": int(r["source_index"]), "condition": "decision_position",
            "arm": "decision_position", "position": t, "group": r["group"],
            "prefix_len": len(prefix),
            "student_entropy": float(-(p * logp).sum()),
            "student_top1_prob": float(p.max()),
            "student_top2_gap": float(top2[0] - top2[1]),
            "margin_vs_yS": float(z[y_T] - z[y_S]),
            "margin_vs_rest": margin_vs_rest,
            "logp_yT": float(logp[y_T]),
            "logp_yS": float(logp[y_S]),
            "parent_M_A": float(r["M_A"]), "parent_M_A_recomputed_delta": margin_vs_rest - float(r["M_A"]),
            "run_config_hash": cfg["config_hash"],
            "checkpoint_sha256": cfg["checkpoint_sha256"], "shard": args.shard_name,
        })
        if (n + 1) % 32 == 0 or n == len(shard) - 1:
            H.write_result_envelope(
                body, task_id="RESCUE-GUIDANCE-REPRESENTATION-MECHANISM-SPRINT-001",
                stage="track_d_features", records=records, arms=["decision_position"],
                sample_ids=sorted({x["canonical_source_id"] for x in records}),
                config_hash=cfg["config_hash"], checkpoint_sha256=cfg["checkpoint_sha256"])
    H.write_result_envelope(
        body, task_id="RESCUE-GUIDANCE-REPRESENTATION-MECHANISM-SPRINT-001",
        stage="track_d_features", records=records, arms=["decision_position"],
        sample_ids=sorted({x["canonical_source_id"] for x in records}),
        config_hash=cfg["config_hash"], checkpoint_sha256=cfg["checkpoint_sha256"])
    meta = {"shard": args.shard_name, "records": len(records),
            "wall_seconds": round(time.time() - t0, 1), "body": str(body),
            "body_sha256": sha256_file(body),
            "verify": H.verify_result(body, expected_config_hash=cfg["config_hash"],
                                      expected_checkpoint_sha256=cfg["checkpoint_sha256"])}
    H.atomic_write_text(out / f"td_shard_{args.shard_name}_meta.json",
                        json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"shard": args.shard_name, "done": True, "records": len(records),
                      "verify": meta["verify"]["status"],
                      "wall_seconds": meta["wall_seconds"]}, ensure_ascii=False), flush=True)
    return 0 if meta["verify"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
