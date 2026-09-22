"""Assemble the RESCUE 002 evidence pack from real artefacts (runs on the GPU host).

Reads only files that already exist; a missing stage becomes an explicit
NOT_RUN/absent record instead of a fabricated value.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

ARMS = ("StudentBaseline", "StudentAdapter", "PositionAlignmentNull")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jload(p: Path, default=None):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def jlload(p: Path):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="k4_seqloss_dev_closure_002")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    diag = root / "diagnostics_001"
    fix = root / "prefix_fix_001"
    run = fix / "run_001"
    k4 = fix / "k4_001"

    for name in ("adapter_path_diagnostic.jsonl", "adapter_gradient_diagnostic.json",
                 "adapter_path_trace.json"):
        if (diag / name).exists():
            shutil.copy2(diag / name, out / name)

    # ---- training curve ----
    curve = jlload(run / "training_curve.jsonl")
    if curve:
        with (out / "training_curve.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(curve[0]))
            w.writeheader()
            w.writerows(curve)

    # ---- validation selection: merge shards ----
    shard_files = sorted(run.glob("validation_shard_*.json"))
    if not shard_files:
        shard_files = sorted(run.glob("validation_partial_*.json"))
        shards = [{"all": [jload(p)] } for p in shard_files] if False else []
        merged = [jload(p) for p in shard_files]
    else:
        merged = []
        for p in shard_files:
            merged.extend(jload(p, []))
    if merged:
        merged.sort(key=lambda r: r["step"])
        valid = [r for r in merged if r.get("valid")]
        sel = min(valid, key=lambda r: (r["paired_first_gold_ce"], r["step"])) if valid else None
        (out / "validation_selection.json").write_text(json.dumps({
            "selection": sel, "all": merged,
            "rule": ("frozen candidate contract: mean((CE_treatment + CE_null)/2) over validation "
                     "rows, computed on the first gold answer token, both arms finite, "
                     "valid_rate >= 0.95, tie chooses the earliest step, validation only, "
                     "no confirmation, no early stopping"),
            "shards": [p.name for p in shard_files],
        }, indent=2) + "\n", encoding="utf-8")

    # ---- QA merge ----
    qa_files = sorted(run.glob("qa_shard_*.jsonl"))
    qa_rows = []
    for p in qa_files:
        qa_rows.extend(jlload(p))
    if qa_rows:
        file_qa = out / "qa_results.csv"
        with file_qa.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["canonical_source_id", "source_index", "gold", "condition", "prediction",
                        "em", "f1"])
            for r in qa_rows:
                for arm in ARMS:
                    w.writerow([r["canonical_source_id"], r["source_index"], r["gold"], arm,
                                r["predictions"][arm], r["em"][arm], r["f1"][arm]])
        (out / "qa_per_sample.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in qa_rows) + "\n", encoding="utf-8")

    # ---- K4 results merge ----
    k4_files = sorted(k4.glob("k4_results_shard_*.jsonl"))
    k4_rows = []
    for p in k4_files:
        k4_rows.extend(jlload(p))
    if k4_rows:
        cols = [c for c in k4_rows[0] if not isinstance(k4_rows[0][c], (list, dict))]
        with (out / "k4_results.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(k4_rows)
        (out / "k4_per_sample.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in k4_rows) + "\n", encoding="utf-8")
    for name in ("k4_correctness.json",):
        if (k4 / name).exists():
            shutil.copy2(k4 / name, out / name)

    # ---- K4 with incremental (path-identical) verification ----
    k4i = fix / "k4_incr"
    k4i_rows = []
    for p in sorted(k4i.glob("k4_results_shard_*.jsonl")):
        k4i_rows.extend(jlload(p))
    if k4i_rows:
        cols = [c for c in k4i_rows[0] if not isinstance(k4i_rows[0][c], (list, dict))]
        with (out / "k4_results_incremental.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(k4i_rows)
        (out / "k4_per_sample_incremental.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in k4i_rows) + "\n", encoding="utf-8")
    if (k4i / "k4_correctness.json").exists():
        shutil.copy2(k4i / "k4_correctness.json", out / "k4_correctness_incremental_gate.json")

    def k4_block(rows):
        per = {}
        for c in sorted({r["condition"] for r in rows}):
            sub = [r for r in rows if r["condition"] == c]
            prop = sum(int(r["proposed_draft_tokens"]) for r in sub)
            acc = sum(int(r["accepted_draft_tokens"]) for r in sub)
            blocks = sum(int(r["blocks"]) for r in sub)
            walls = [float(r["wall_seconds"]) for r in sub]
            per[c] = {
                "valid_N": len(sub), "proposed": prop, "accepted": acc,
                "proposal_acceptance": acc / prop if prop else None,
                "mean_accepted_draft_length": acc / blocks if blocks else None,
                "full_accept_blocks": sum(int(r["full_accept_blocks"]) for r in sub),
                "reject_blocks": sum(int(r["reject_blocks"]) for r in sub),
                "correction_tokens": sum(int(r["correction_tokens"]) for r in sub),
                "bonus_tokens": sum(int(r["bonus_tokens"]) for r in sub),
                "eos_terminated_rows": sum(1 for r in sub if r["eos_terminated"]),
                "matches_target_only_greedy_rows": sum(
                    1 for r in sub if r.get("matches_target_only_greedy")),
                "correctness_failures": sum(
                    1 for r in sub if not r.get("matches_target_only_greedy")),
                "wall_seconds_mean": sum(walls) / len(walls),
                "wall_seconds_total": sum(walls),
            }
        return {"planned_N": 128, "conditions": per}

    # ---- metrics summary ----
    summary = {"task_id": "RESCUE-ADAPTER-DEV-CLOSURE-AND-K4-SEQOSS-002",
               "note": "every number below is computed from the files in this directory",
               "stages": {}}

    gtrace = jload(diag / "adapter_path_trace.json")
    if gtrace:
        summary["stages"]["adapter_path"] = gtrace["aggregate"] | {
            "top1_changed_counts": gtrace["top1_changed_counts"],
            "checkpoint_sha256": gtrace["checkpoint_sha256"],
            "checkpoint_step": gtrace["checkpoint_step"],
            "intervention_invocations": jlload(diag / "adapter_path_diagnostic.jsonl")[0][
                "intervention_invocations"] if (diag / "adapter_path_diagnostic.jsonl").exists() else None}

    ggrad = jload(diag / "adapter_gradient_diagnostic.json")
    if ggrad:
        recs = ggrad["records"]
        summary["stages"]["task_gradient"] = {
            "regularization_term_present": ggrad["regularization_term_present_in_training_step"],
            "samples": len(recs),
            "first_token_gradient_nonzero_all_samples": all(
                r["grad_first_token_only"]["_all"]["norm"] > 0 for r in recs),
            "later_token_gradient_max_norm": max(
                r["grad_later_tokens_only"]["_all"]["norm"] for r in recs),
            "total_over_first_ratio_matches_1_over_T": all(
                abs(r["grad_total_over_first_norm_ratio"] - r["expected_ratio"]) < 1e-3 for r in recs),
            "per_answer_position_nonzero_count": {
                str(i): sum(1 for r in recs for p in r["per_answer_position"]
                            if p["answer_position"] == i and p["grad"]["_all"]["norm"] > 0)
                for i in range(max(len(r["per_answer_position"]) for r in recs))},
        }

    smoke = jload(fix / "gradsmoke_001" / "prefix_gradient_smoke.json")
    if smoke:
        summary["stages"]["supervision_path_fix"] = {
            "samples": len(smoke["records"]),
            "all_positions_nonzero_task_gradient": all(
                r["all_positions_nonzero_task_gradient"] for r in smoke["records"]),
            "student_base_grad_leak": any(r["student_base_grad_leak"] for r in smoke["records"]),
        }

    tra = jload(run / "training_result.json")
    if tra:
        summary["stages"]["sequence_loss_training"] = tra

    if merged:
        summary["stages"]["validation_selection"] = {
            "selected_step": sel["step"] if sel else None,
            "checkpoints": [r["step"] for r in merged],
            "metric": "paired first-gold-token CE (frozen rule)",
            "metric_string_written_by_lineage_001": "validation paired first-gold-token CE; "
                                                    "tie earliest; no confirmation",
        }

    if qa_rows:
        n = len(qa_rows)
        summary["stages"]["qa"] = {
            "effective_N": n,
            "arms": {arm: {"em": sum(r["em"][arm] for r in qa_rows) / n,
                           "f1": sum(r["f1"][arm] for r in qa_rows) / n} for arm in ARMS},
            "adapter_vs_null_text_identical_rows": sum(
                1 for r in qa_rows if r["predictions"]["StudentAdapter"] == r["predictions"]["PositionAlignmentNull"]),
            "baseline_vs_adapter_text_identical_rows": sum(
                1 for r in qa_rows if r["predictions"]["StudentBaseline"] == r["predictions"]["StudentAdapter"]),
        }

    if k4_rows:
        summary["stages"]["k4_development_batched_verify"] = k4_block(k4_rows)
    if k4i_rows:
        summary["stages"]["k4_development_incremental_verify"] = k4_block(k4i_rows)
        summary["stages"]["k4_development_incremental_verify"]["note"] = (
            "verification evaluates one proposal per target forward, i.e. structurally identical "
            "to the Target-only reference computation; this is the variant whose accept/reject "
            "decisions are exact against Target-only greedy")
        if k4_rows:
            om = {(r["canonical_source_id"], r["condition"]): tuple(r["tokens"]) for r in k4_rows}
            nm = {(r["canonical_source_id"], r["condition"]): tuple(r["tokens"]) for r in k4i_rows}
            summary["stages"]["k4_development_incremental_verify"][
                "emitted_sequences_identical_to_batched"] = {
                    "identical": sum(1 for k in om if k in nm and om[k] == nm[k]),
                    "pairs": len(om)}
    k4corr = jload(k4 / "k4_correctness.json")
    if k4corr:
        summary["stages"]["k4_correctness"] = {
            "all_identical": k4corr.get("all_identical"),
            "samples": len(k4corr.get("records", [])),
            "k": k4corr.get("k"), "target_dtype": k4corr.get("target_dtype"),
            "configurations": sorted({c for r in k4corr.get("records", [])
                                      for c in r["configurations"]}),
        }

    (out / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")

    # ---- manifest ----
    lines = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            lines.append(f"{sha256(p)}  {p.relative_to(out).as_posix()}")
    (out / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "files": len(lines),
                      "stages": list(summary["stages"])}, indent=2))


if __name__ == "__main__":
    main()
