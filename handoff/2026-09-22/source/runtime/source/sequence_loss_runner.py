"""Fail-closed real-Qwen smoke and DEVELOPMENT_BASELINE_V1 runner."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

from qwen_adapter_backend import QwenAdapterBackend, RealQwenRuntimeError

ARMS = ("StudentBaseline", "StudentAdapter", "PositionAlignmentNull")
CHECKPOINTS = (16, 32, 48, 64, 80, 96, 112, 128)
BASELINE_PRIOR_SHA256 = {
    "train": "af74b6c1c8ed275ac8e3fc6dc262f99c03763f11ba8a1e193194b9a0d1ae3f59",
    "validation": "b820317774791c48439f8caa52d1410c9c744018524e92bb5ce02d033e0256c7",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_row(ledger: Path, dataset_path: Path) -> dict[str, str]:
    with ledger.open(encoding="utf-8-sig", newline="") as f:
        row = next(csv.DictReader(f))
    from datasets import load_from_disk
    ds = load_from_disk(str(dataset_path))
    return make_row(row, ds[int(row["source_index"])])


def make_row(row: dict[str, str], item: dict) -> dict[str, str]:
    docs = []
    for title, sentences in zip(item.get("context", {}).get("title", []), item.get("context", {}).get("sentences", [])):
        text = (str(title).strip() + "\n" + " ".join(map(str, sentences))).strip()
        if text:
            docs.append(text)
    question = f"Question: {str(item.get('question', '')).strip()}\nAnswer:"
    prompt = ("\n\n".join([f"[Doc {i + 1}]\n{x}" for i, x in enumerate(docs)] + [question])).strip()
    if hashlib.sha256(prompt.encode()).hexdigest() != row["prompt_sha256"]:
        raise RealQwenRuntimeError("ledger prompt_sha256 does not match locked dataset prompt")
    if hashlib.sha256(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != row["content_sha256"]:
        raise RealQwenRuntimeError("ledger content_sha256 does not match locked dataset row")
    return {"canonical_source_id": row["canonical_source_id"], "source_index": row["source_index"],
            "prompt": prompt, "answer": str(item.get("answer", "")).strip(),
            "prompt_sha256": row["prompt_sha256"], "content_sha256": row["content_sha256"]}


def load_rows(ledger: Path, dataset_path: Path) -> list[dict[str, str]]:
    with ledger.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    from datasets import load_from_disk
    ds = load_from_disk(str(dataset_path))
    return [make_row(row, ds[int(row["source_index"])]) for row in rows]


def run_capture(cmd: list[str], stdout: Path, stderr: Path, cwd: Path) -> int:
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=cwd, stdout=out, stderr=err, text=True, check=False)
    stdout.with_suffix(".exit").write_text(str(proc.returncode) + "\n", encoding="utf-8")
    return proc.returncode


def nvidia(path: Path, label: str) -> None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader"],
                           text=True, capture_output=True, check=False)
        path.write_text(f"{label}\nexit={r.returncode}\n{r.stdout}{r.stderr}", encoding="utf-8")
    except FileNotFoundError:
        path.write_text(f"{label}\nexit=127\nnvidia-smi unavailable\n", encoding="utf-8")


def validate_base_path(path: str) -> None:
    lower = Path(path).name.lower()
    if "instruct" in lower:
        raise RealQwenRuntimeError("Instruct snapshot is forbidden; explicit base Qwen2 snapshot required")


def load_reused_prior_cache(root: Path, name: str, rows: list[dict[str, str]]) -> dict[str, torch.Tensor] | None:
    prior_path = root / f"{name}_teacher_priors.pt"
    manifest_path = root / f"{name}_teacher_prior_manifest.json"
    if not prior_path.exists() or not manifest_path.exists():
        return None
    actual_sha = sha256(prior_path)
    expected_sha = BASELINE_PRIOR_SHA256[name]
    if actual_sha != expected_sha:
        raise RealQwenRuntimeError(f"reused {name} prior cache SHA mismatch")
    payload = torch.load(prior_path, map_location="cpu", weights_only=True)
    expected_keys = {str(row["canonical_source_id"]) for row in rows}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise RealQwenRuntimeError(f"reused {name} prior cache key set mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "DONE" or int(manifest.get("record_count_requested", -1)) != len(rows):
        raise RealQwenRuntimeError(f"reused {name} prior manifest is not complete")
    (root / f"{name}_teacher_reuse.json").write_text(json.dumps({
        "status": "REUSED_VERIFIED_BASELINE_CACHE",
        "sha256": actual_sha,
        "records": len(rows),
        "manifest": str(manifest_path),
    }, indent=2) + "\n", encoding="utf-8")
    return payload


def smoke(args: argparse.Namespace, root: Path) -> dict:
    validate_base_path(args.teacher_model)
    validate_base_path(args.student_model)
    row = load_row(Path(args.train_ledger), Path(args.dataset_path))
    input_path = root / "smoke_input.json"
    input_path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cache_path = root / "train_teacher_priors.pt"
    cache_manifest = root / "train_teacher_prior_manifest.json"
    if not cache_path.exists() or not cache_manifest.exists():
        raise RealQwenRuntimeError("verified train Teacher prior cache is required for smoke")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    prior_payload = cache.get(str(row["canonical_source_id"])) if isinstance(cache, dict) else None
    if not isinstance(prior_payload, torch.Tensor):
        raise RealQwenRuntimeError("reused smoke Teacher prior key is missing")
    prior_sha = sha256(cache_path)
    backend = QwenAdapterBackend(args.teacher_model, args.student_model, args.student_device,
                                 source_lock=args.source_lock, load_teacher=False)
    target = backend.prompt_target(row["prompt"], row["answer"])
    if target.target_position != len(target.input_ids) - 1 or target.target_id is None:
        raise RealQwenRuntimeError("prompt boundary/first gold token target contract failed")
    before = {n: p.detach().clone() for n, p in backend.student.named_parameters()}
    smoke_rows = backend._student_rows(target)
    smoke_mask = torch.tensor(target.attention_mask, device=backend.device)
    with torch.no_grad():
        _, smoke_alpha_zero, smoke_entropy = backend.adapter(prior_payload.to(backend.device), smoke_rows, smoke_mask)
        smoke_static_alpha = torch.sigmoid(backend.adapter.gate_logits) * backend.adapter.config.max_alpha
    w_init_zero = bool(torch.equal(backend.adapter.w.detach(), torch.zeros_like(backend.adapter.w.detach())))
    w_zero_equivalent = bool(torch.allclose(smoke_alpha_zero, smoke_static_alpha, atol=1e-7, rtol=0.0))
    result = backend.step(target, prior_payload)
    after = {n: p.detach() for n, p in backend.student.named_parameters()}
    base_unchanged = all(torch.equal(before[n], after[n]) for n in before)
    checkpoint = root / "smoke_checkpoint.pt"
    backend.checkpoint(checkpoint, 1, json.loads(cache_manifest.read_text(encoding="utf-8")))
    resumed = backend.load_checkpoint(checkpoint)
    decoded = backend.cached_decode(target, int(target.target_id))
    nvidia(root / "nvidia_smi_student_end.txt", "student_end")
    if not base_unchanged or resumed != 1 or not decoded["decode_logits_finite"]:
        raise RealQwenRuntimeError("smoke base freeze/checkpoint/decode gate failed")
    return {"status": "PASS", "real_qwen": True, "teacher_prior_reused": True, "source_index": row["source_index"],
            "target_id": target.target_id, "target_position": target.target_position,
            "loss": float(result["loss"].detach().cpu()), "loss_finite": True,
            "first_token_loss": float(result["first_loss"].detach().cpu()),
            "later_token_loss": float(result["later_loss"].detach().cpu()) if result["later_loss"] is not None else None,
            "answer_token_count": result["answer_token_count"],
            "prefill_adapter_invocations": result["prefill_adapter_invocations"],
            "cached_decode_adapter_invocations": result["cached_decode_adapter_invocations"],
            "adapter_changed": result["adapter_changed"], "grad_norm": result["grad_norm"],
            "w_grad_norm": result["w_grad_norm"], "w_grad_nonzero": result["w_grad_nonzero"],
            "w_init_zero": w_init_zero, "w_zero_equivalent": w_zero_equivalent,
            "entropy_finite": bool(torch.isfinite(smoke_entropy).all().item()),
            "base_weights_unchanged": base_unchanged, "prior_sha256": prior_sha,
            "checkpoint_resume_step": resumed, "cached_decode": decoded}


def train_and_validate(args: argparse.Namespace, root: Path) -> dict:
    train_rows = load_rows(Path(args.train_ledger), Path(args.dataset_path))
    validation_rows = load_rows(Path(args.validation_ledger), Path(args.dataset_path))
    prior_cache = {}
    for rowset, name in ((train_rows, "train"), (validation_rows, "validation")):
        input_path = root / f"{name}_prompts.json"
        input_path.write_text(json.dumps(rowset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        prior_path = root / f"{name}_teacher_priors.pt"
        manifest_path = root / f"{name}_teacher_prior_manifest.json"
        reused = load_reused_prior_cache(root, name, rowset)
        if reused is not None:
            prior_cache[name] = reused
            continue
        code = run_capture([
            sys.executable, str(Path(__file__).with_name("q3_teacher_prior_worker.py")),
            "--teacher-model", args.teacher_model, "--student-tokenizer", args.student_model,
            "--input", str(input_path), "--out", str(prior_path), "--manifest", str(manifest_path),
            "--source-lock", args.source_lock, "--device", args.teacher_device,
        ], root / f"{name}_teacher.stdout", root / f"{name}_teacher.stderr", Path.cwd())
        if code != 0:
            raise RealQwenRuntimeError(f"{name} Teacher prior worker failed exit={code}")
        prior_cache[name] = torch.load(prior_path, map_location="cpu", weights_only=True)
    train_priors = prior_cache["train"]
    validation_priors = prior_cache["validation"]
    random.seed(20260830)
    torch.manual_seed(20260830)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260830)
    backend = QwenAdapterBackend(args.teacher_model, args.student_model, args.student_device,
                                 source_lock=args.source_lock, load_teacher=False)
    checkpoints = {}
    train_log = root / "training_curve.jsonl"
    if train_log.exists():
        train_log.unlink()
    for step, row in enumerate(train_rows, 1):
        target = backend.prompt_target(row["prompt"], row["answer"])
        result = backend.step(target, train_priors[row["canonical_source_id"]])
        record = {"step": step, "canonical_source_id": row["canonical_source_id"],
                  "source_index": row["source_index"], "loss": float(result["loss"].detach().cpu()),
                  "first_token_loss": float(result["first_loss"].detach().cpu()),
                  "later_token_loss": float(result["later_loss"].detach().cpu()) if result["later_loss"] is not None else None,
                  "answer_token_count": result["answer_token_count"],
                  "target_id": target.target_id, "target_position": target.target_position,
                  "grad_norm": result["grad_norm"], "w_grad_norm": result["w_grad_norm"],
                  "w_grad_nonzero": result["w_grad_nonzero"],
                  "entropy_mean": float(result["entropy"].mean().detach().cpu()),
                  "entropy_min": float(result["entropy"].min().detach().cpu()),
                  "entropy_max": float(result["entropy"].max().detach().cpu()),
                  "adapter_changed": result["adapter_changed"]}
        with train_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if step in CHECKPOINTS:
            checkpoint = root / f"checkpoint_{step:03d}.pt"
            backend.checkpoint(checkpoint, step, {"role": "train", "prior_sha256": sha256(root / "train_teacher_priors.pt")})
            checkpoints[step] = str(checkpoint)
    if len(checkpoints) != len(CHECKPOINTS):
        raise RealQwenRuntimeError("training did not produce all required checkpoints")
    selections = []
    for step in CHECKPOINTS:
        backend.load_checkpoint(checkpoints[step])
        losses = []
        for row in validation_rows:
            target = backend.prompt_target(row["prompt"], row["answer"])
            rows = backend._student_rows(target)
            with torch.no_grad():
                treatment = backend.sequence_treatment_step(target, validation_priors[row["canonical_source_id"]], rows)
                treatment_loss = float(treatment["loss"].cpu())
                null_teacher_prior = validation_priors[row["canonical_source_id"]].roll(shifts=1, dims=-1)
                null = backend.sequence_treatment_step(target, null_teacher_prior, rows)
                null_loss = float(null["loss"].cpu())
            losses.append((treatment_loss + null_loss) / 2.0)
        valid_rate = len(losses) / max(1, len(validation_rows))
        mean_loss = sum(losses) / max(1, len(losses))
        selections.append({"step": step, "paired_ce_mean": mean_loss, "valid_rate": valid_rate,
                           "valid": valid_rate >= 0.95})
    valid = [x for x in selections if x["valid"]]
    if not valid:
        raise RealQwenRuntimeError("no valid validation checkpoint")
    selected = min(valid, key=lambda x: (x["paired_ce_mean"], x["step"]))
    (root / "validation_selection.json").write_text(json.dumps({"selection": selected, "all": selections,
                                                                  "rule": "validation paired first-gold-token CE; tie earliest; no confirmation"}, indent=2) + "\n", encoding="utf-8")
    backend.load_checkpoint(checkpoints[int(selected["step"])])
    predictions = []
    alpha_records = []
    def norm(text: str) -> str:
        return " ".join(str(text).lower().strip().split())
    def f1(pred: str, gold: str) -> float:
        p, g = norm(pred).split(), norm(gold).split()
        if not p or not g: return float(p == g)
        common = sum(min(p.count(x), g.count(x)) for x in set(p))
        if not common: return 0.0
        precision, recall = common / len(p), common / len(g)
        return 2 * precision * recall / (precision + recall)
    for row in validation_rows:
        target = backend.prompt_target(row["prompt"], row["answer"])
        with torch.no_grad():
            baseline_out = backend.student(**backend._ids(target), use_cache=False, output_attentions=False)
            rows = backend._student_rows(target)
            mask = torch.tensor(target.attention_mask, device=backend.device)
            mapped, alpha, entropy = backend.adapter(validation_priors[row["canonical_source_id"]].to(backend.device), rows, mask)
            null_mapped, null_alpha, null_entropy = backend.adapter(validation_priors[row["canonical_source_id"]].roll(1, dims=-1).to(backend.device), rows, mask)
        arms = {
            "StudentBaseline": backend.generate(target),
            "StudentAdapter": backend.generate(target, mapped.detach(), alpha.detach()),
            "PositionAlignmentNull": backend.generate(target, null_mapped.detach(), null_alpha.detach()),
        }
        predictions.append({"canonical_source_id": row["canonical_source_id"], "source_index": row["source_index"],
                            "gold": row["answer"], "predictions": arms,
                            "em": {k: int(norm(v) == norm(row["answer"])) for k, v in arms.items()},
                            "f1": {k: f1(v, row["answer"]) for k, v in arms.items()}})
        alpha_records.append({"canonical_source_id": row["canonical_source_id"],
                             "source_index": row["source_index"],
                             "entropy": entropy.detach().float().cpu().tolist(),
                             "null_entropy": null_entropy.detach().float().cpu().tolist(),
                             "alpha_treatment": alpha.detach().float().cpu().tolist(),
                             "alpha_null": null_alpha.detach().float().cpu().tolist()})
    (root / "validation_predictions.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in predictions) + "\n", encoding="utf-8")
    summary = {arm: {metric: sum(x[metric][arm] for x in predictions) / max(1, len(predictions))
                     for metric in ("em", "f1")} for arm in ARMS}
    (root / "validation_summary.json").write_text(json.dumps({"checkpoint": selected["step"], "development_only": True, "summary": summary}, indent=2) + "\n", encoding="utf-8")
    (root / "alpha_behavior_records.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in alpha_records) + "\n", encoding="utf-8")
    return {"status": "DEVELOPMENT_BASELINE_COMPLETE", "train_steps": len(train_rows),
            "validation_rows": len(validation_rows), "checkpoints": list(CHECKPOINTS),
            "selected_checkpoint": selected, "confirmation": "NOT_STARTED"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("smoke", "train", "validation"), required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--teacher-model", required=True)
    p.add_argument("--student-model", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--train-ledger", required=True)
    p.add_argument("--validation-ledger", required=True)
    p.add_argument("--source-lock", required=True)
    p.add_argument("--teacher-device", default="cuda:0")
    p.add_argument("--student-device", default="cuda:1")
    args = p.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "runner_command.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    if args.mode != "smoke":
        smoke_result = root / "smoke_result.json"
        if not smoke_result.exists() or json.loads(smoke_result.read_text(encoding="utf-8")).get("status") != "PASS":
            raise SystemExit("DEVELOPMENT_BASELINE_BLOCKED_UNTIL_REAL_SMOKE_PASS")
        if args.mode == "train":
            result = train_and_validate(args, root)
            (root / "baseline_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result, sort_keys=True))
            return 0
        raise SystemExit("validation is executed after fixed baseline training")
    try:
        result = smoke(args, root)
        (root / "smoke_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        result = {"status": "BLOCKED_BY_REAL_QWEN_RUNTIME", "real_qwen": False,
                  "error": repr(exc), "gpu_started": True}
        (root / "smoke_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
