"""RESCUE 002 / P3 runner: per-prefix sequence-loss candidate (single candidate).

Modes:
  gradsmoke : prove per-answer-position task gradient reaches the Adapter
  train     : 128 optimizer steps, frozen config, then validation-only selection
  validate  : shardable validation re-evaluation over existing checkpoints
  qa        : 3-arm QA on the development ledger for one selected checkpoint
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefix_seqloss import PrefixSequenceBackend, load_backend_module  # noqa: E402

CHECKPOINTS = (16, 32, 48, 64, 80, 96, 112, 128)
ARMS = ("StudentBaseline", "StudentAdapter", "PositionAlignmentNull")
SEED = 20260830


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def peak_vram():
    if not torch.cuda.is_available():
        return None
    return {"peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1)}


def norm_txt(t):
    return " ".join(str(t).lower().strip().split())


def f1_score(pred, gold):
    p, g = norm_txt(pred).split(), norm_txt(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum(min(p.count(x), g.count(x)) for x in set(p))
    if not common:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def build(args, load_teacher=False):
    backend_mod, runner_mod = load_backend_module(args.source_dir, args.source_lock)
    backend = backend_mod.QwenAdapterBackend(
        args.teacher_model, args.student_model, args.device,
        source_lock=args.source_lock, load_teacher=load_teacher)
    return backend_mod, runner_mod, backend


def grad_smoke(args):
    _, runner_mod, backend = build(args)
    wrap = PrefixSequenceBackend(backend)
    rows = runner_mod.load_rows(Path(args.train_ledger), Path(args.dataset_path))[: args.samples]
    priors = torch.load(Path(args.prior_cache), map_location="cpu", weights_only=True)
    out = {"mode": "grad_smoke", "implementation": "per_prefix_prefill",
           "supervision": "each answer position t via prefill(prompt+gold[:t])",
           "frozen_aspects": ["Adapter prior/alpha computed from prompt only",
                              "intervention prefill-only and last-query-row",
                              "Student frozen", "single optimizer step per sample"],
           "records": []}
    for i, row in enumerate(rows, 1):
        target = backend.prompt_target(row["prompt"], row["answer"])
        prior = priors[row["canonical_source_id"]]
        rows_s = backend._student_rows(target)
        per_pos, groups = [], ["base.mix_logits", "base.residual_logits", "base.gate_logits",
                               "base.log_temperature", "entropy_gate_weight"]
        for t, gold in enumerate(target.answer_ids):
            wrap.optimizer.zero_grad(set_to_none=True)
            maps, gates, _, _, _ = wrap._maps(target, prior, rows_s)
            ce = torch.nn.functional.cross_entropy(
                wrap.prefix_logits(target, t, maps, gates),
                torch.tensor([int(gold)], device=wrap.device))
            ce.backward()
            g = {n: p.grad for n, p in wrap.adapter.named_parameters()}
            summary = {}
            for n in groups:
                t_ = g.get(n)
                summary[n] = None if t_ is None else {
                    "norm": float(t_.detach().float().norm()),
                    "nonzero": bool(torch.any(t_ != 0).item())}
            total = sum(x.float().pow(2).sum() for x in g.values() if x is not None)
            summary["_all"] = {"norm": float(torch.sqrt(total))}
            per_pos.append({"answer_position": t, "gold_id": int(gold),
                            "ce": float(ce.detach()), "grad": summary})
        base_leak = any(p.grad is not None and torch.any(p.grad != 0)
                        for p in backend.student.parameters() if p.requires_grad)
        out["records"].append({
            "sample": i, "canonical_source_id": row["canonical_source_id"],
            "answer_token_count": len(target.answer_ids), "positions": per_pos,
            "all_positions_nonzero_task_gradient": all(
                p["grad"]["_all"]["norm"] > 0 for p in per_pos),
            "student_base_grad_leak": bool(base_leak),
            "student_base_state": {"any_base_param_requires_grad":
                                   any(p.requires_grad for p in backend.student.parameters())},
        })
        wrap.optimizer.zero_grad(set_to_none=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "prefix_gradient_smoke.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "OK", "samples": len(out["records"]),
                      "all_ok": all(r["all_positions_nonzero_task_gradient"] for r in out["records"])}))
    return 0


def train(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    _, runner_mod, backend = build(args)
    wrap = PrefixSequenceBackend(backend)
    train_rows = runner_mod.load_rows(Path(args.train_ledger), Path(args.dataset_path))
    priors = torch.load(Path(args.prior_cache), map_location="cpu", weights_only=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    log = root / "training_curve.jsonl"
    log.unlink(missing_ok=True)
    t0 = time.time()
    for step, row in enumerate(train_rows[: args.steps], 1):
        target = backend.prompt_target(row["prompt"], row["answer"])
        ts = time.time()
        r = wrap.train_step(target, priors[row["canonical_source_id"]])
        rec = {"step": step, "canonical_source_id": row["canonical_source_id"],
               "loss": r["loss"], "first_token_loss": r["first_loss"],
               "later_token_loss": r["later_loss"],
               "answer_token_count": r["answer_token_count"],
               "supervised_tokens": r["supervised_tokens"],
               "grads_finite": r["grads_ok"], "step_seconds": round(time.time() - ts, 3)}
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if step in CHECKPOINTS:
            torch.save({"step": step, "adapter": wrap.adapter.state_dict(),
                        "optimizer": wrap.optimizer.state_dict(),
                        "supervision": "per_prefix_prefill",
                        "teacher_identity": backend.teacher_identity,
                        "student_identity": backend.student_identity},
                       root / f"checkpoint_{step:03d}.pt")
    result = {"status": "SEQUENCE_PREFIX_TRAINING_COMPLETE", "steps": args.steps,
              "checkpoints": list(CHECKPOINTS),
              "supervised_token_count": wrap.supervised_token_count,
              "forward_count": wrap.supervision_forward_count,
              "backward_count": wrap.supervision_backward_count,
              "wall_seconds": round(time.time() - t0, 1),
              "peak_vram": peak_vram()}
    (root / "training_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0


def validate(args):
    root = Path(args.root)
    _, runner_mod, backend = build(args)
    wrap = PrefixSequenceBackend(backend)
    validation_rows = runner_mod.load_rows(Path(args.validation_ledger), Path(args.dataset_path))
    vpriors = torch.load(Path(args.validation_prior_cache), map_location="cpu", weights_only=True)
    selections = []
    for step in args.steps_list:
        ck = root / f"checkpoint_{step:03d}.pt"
        payload = torch.load(ck, map_location=backend.device, weights_only=False)
        wrap.adapter.load_state_dict(payload["adapter"])
        first_t, first_n, seq_t = [], [], []
        for row in validation_rows:
            target = backend.prompt_target(row["prompt"], row["answer"])
            p = vpriors[row["canonical_source_id"]]
            rows_s = backend._student_rows(target)
            first_t.append(wrap.first_gold_ce(target, p, rows_s, null=False))
            first_n.append(wrap.first_gold_ce(target, p, rows_s, null=True))
            seq_t.append(wrap.sequence_ce(target, p, rows_s))
        n = len(validation_rows)
        paired = sum((a + b) / 2.0 for a, b in zip(first_t, first_n)) / n
        selections.append({
            "step": step, "checkpoint_sha256": sha256(ck),
            "paired_first_gold_ce": paired,
            "treatment_first_gold_ce": sum(first_t) / n,
            "null_first_gold_ce": sum(first_n) / n,
            "treatment_sequence_ce": sum(seq_t) / n,
            "valid_rate": n / len(validation_rows), "valid": True,
        })
        (root / f"validation_partial_{step:03d}.json").write_text(
            json.dumps(selections[-1], indent=2) + "\n", encoding="utf-8")
    (root / f"validation_shard_{args.shard_name}.json").write_text(
        json.dumps(selections, indent=2) + "\n", encoding="utf-8")
    return 0


def qa(args):
    root = Path(args.root)
    _, runner_mod, backend = build(args)
    wrap = PrefixSequenceBackend(backend)
    payload = torch.load(root / f"checkpoint_{args.step:03d}.pt",
                         map_location=backend.device, weights_only=False)
    wrap.adapter.load_state_dict(payload["adapter"])
    validation_rows = runner_mod.load_rows(Path(args.validation_ledger), Path(args.dataset_path))
    vpriors = torch.load(Path(args.validation_prior_cache), map_location="cpu", weights_only=True)
    preds, alpha_rec = [], []
    for row in validation_rows[args.shard_lo:args.shard_hi]:
        target = backend.prompt_target(row["prompt"], row["answer"])
        p = vpriors[row["canonical_source_id"]]
        with torch.no_grad():
            rows_s = backend._student_rows(target)
            mask = torch.tensor(target.attention_mask, device=backend.device)
            mapped, alpha, entropy = wrap.adapter(p.to(backend.device), rows_s, mask)
            nmapped, nalpha, nentropy = wrap.adapter(p.roll(1, dims=-1).to(backend.device), rows_s, mask)
        arms = {"StudentBaseline": backend.generate(target),
                "StudentAdapter": backend.generate(target, mapped.detach(), alpha.detach()),
                "PositionAlignmentNull": backend.generate(target, nmapped.detach(), nalpha.detach())}
        preds.append({"canonical_source_id": row["canonical_source_id"],
                      "source_index": row["source_index"], "gold": row["answer"],
                      "predictions": arms,
                      "em": {k: int(norm_txt(v) == norm_txt(row["answer"])) for k, v in arms.items()},
                      "f1": {k: f1_score(v, row["answer"]) for k, v in arms.items()}})
        alpha_rec.append({"canonical_source_id": row["canonical_source_id"],
                          "alpha_treatment": alpha.detach().float().cpu().tolist(),
                          "alpha_null": nalpha.detach().float().cpu().tolist()})
    (root / f"qa_shard_{args.shard_name}.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in preds) + "\n", encoding="utf-8")
    (root / f"qa_alpha_shard_{args.shard_name}.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in alpha_rec) + "\n", encoding="utf-8")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=("gradsmoke", "train", "validate", "qa"))
    p.add_argument("--root", default="/tmp/unused")
    p.add_argument("--out", default="/tmp/unused")
    p.add_argument("--source-dir", required=True)
    p.add_argument("--source-lock", required=True)
    p.add_argument("--teacher-model", required=True)
    p.add_argument("--student-model", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--train-ledger", required=True)
    p.add_argument("--validation-ledger", required=True)
    p.add_argument("--prior-cache", required=True)
    p.add_argument("--validation-prior-cache", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--steps-list", type=int, nargs="+", default=list(CHECKPOINTS))
    p.add_argument("--shard-name", default="0")
    p.add_argument("--shard-lo", type=int, default=0)
    p.add_argument("--shard-hi", type=int, default=100000)
    p.add_argument("--step", type=int, default=64)
    args = p.parse_args()
    return {"gradsmoke": grad_smoke, "train": train,
            "validate": validate, "qa": qa}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
