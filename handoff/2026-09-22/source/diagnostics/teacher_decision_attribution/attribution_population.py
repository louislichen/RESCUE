"""PHASE A — freeze the teacher-attribution population (no GPU attribution, no backward).

RESCUE-DECISION-AWARE-TEACHER-SIGNAL-SPRINT-001 §2/§3.

Builds teacher_attribution_population.json for the 67 S!=T decision positions by identity-matching
three already-frozen runs (parent diagnostic, mechanism sprint Track A, R1 pilot), and re-derives
the prefix token ids from the frozen ledger prompt + base tokens with the Student tokenizer alone
(no model, no GPU), verifying them against the frozen prefix_sha256.

That is the 67/67 identity gate.  It also records, explicitly, which of the §3 reuse items exist as
persisted tensors and which do not — a reviewer must not be told "reused" for something that only
exists as a scalar summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def sha256_prefix(prompt_ids, toks) -> str:
    return hashlib.sha256(json.dumps([list(map(int, prompt_ids)), list(map(int, toks))],
                                     separators=(",", ":")).encode()).hexdigest()


def load(path, need=None):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "__result_header__" in r:
            continue
        if need and need not in r:
            continue
        out[(r["canonical_source_id"], int(r["position"]))] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dr-rows", required=True)
    ap.add_argument("--ta-rows", required=True)
    ap.add_argument("--r1-rows", required=True)
    ap.add_argument("--validation-ledger", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--student-model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from datasets import load_from_disk
    from transformers import AutoTokenizer
    import csv as _csv

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.student_model, local_files_only=True, use_fast=True)
    ds = load_from_disk(args.dataset_path)
    with Path(args.validation_ledger).open(encoding="utf-8-sig", newline="") as f:
        ledger = list(_csv.DictReader(f))

    dr = load(args.dr_rows)
    ta = load(args.ta_rows, need="M_base")
    r1 = load(args.r1_rows, need="arms")
    pop_keys = sorted([k for k, r in dr.items() if r["group"] == "S!=" + "T"],
                      key=lambda k: (int(dr[k]["source_index"]), k[1]))

    rows, mismatches = [], []
    for k in pop_keys:
        sid, pos = k
        d = dr[k]
        t = ta.get(k)
        p = r1.get(k)
        li = next(i for i, r in enumerate(ledger) if r["canonical_source_id"] == sid)
        lrow = ledger[li]
        srow = ds[int(lrow["source_index"])]
        prompt = str(srow["question"]) if False else None
        from sequence_loss_runner import make_row
        row_src = make_row(lrow, srow)
        prompt = row_src["prompt"]
        base_tokens = list(map(int, d["base_tokens"]))
        prefix_ids = list(tok(prompt, add_special_tokens=False)["input_ids"]) + base_tokens[:pos]
        recomputed_sha = sha256_prefix(list(tok(prompt, add_special_tokens=False)["input_ids"]),
                                       base_tokens[:pos])
        checks = {
            "prefix_sha256_matches_parent": recomputed_sha == d["prefix_sha256"],
            "prefix_len_matches_parent": len(prefix_ids) == int(d["prefix_len"]),
            "track_a_present": t is not None,
            "r1_pilot_present": p is not None,
            "prefix_sha256_track_a": (t is not None and t["prefix_sha256"] == d["prefix_sha256"]),
            "prefix_sha256_r1": (p is not None and p["prefix_sha256"] == d["prefix_sha256"]),
            "y_target_consistent": bool(t is not None and p is not None
                                        and int(t["y_target"]) == int(d["y_target"])
                                        and int(p["y_target"]) == int(d["y_target"])),
            "y_base_consistent": bool(t is not None and p is not None
                                      and int(t["y_base"]) == int(d["y_base"])
                                      and int(p["y_base"]) == int(d["y_base"])),
            "base_margin_identical_track_a_vs_r1": bool(
                t is not None and p is not None
                and abs(float(t["M_base"]) - float(p["M_base"])) < 1e-9),
            "logp_base_present": bool(p is not None and "logp_base_yT" in p),
        }
        entry = {
            "sample_id": sid, "source_index": int(d["source_index"]), "position": pos,
            "group": d["group"], "prefix_len": len(prefix_ids),
            "prefix_sha256": d["prefix_sha256"], "prefix_sha256_recomputed": recomputed_sha,
            "prefix_token_ids": prefix_ids, "base_tokens": base_tokens,
            "y_S": int(d["y_base"]), "y_T": int(d["y_target"]),
            "base_logits_identity": {
                "M_base_track_a": None if t is None else t["M_base"],
                "M_base_alt_track_a": None if t is None else t.get("M_base_alt_definition"),
                "M_base_r1_pilot": None if p is None else p["M_base"],
                "logp_base_yT_r1_pilot": None if p is None else p.get("logp_base_yT"),
                "M_base_alt_minus_parent": None if t is None else t.get("M_base_alt_minus_parent"),
                "framing": "single full-prefix forward, intervention at the last query row",
            },
            "student_local_oracle_identity": None if t is None else {
                "V_oracle": t["V_oracle"], "V_teacher": t["V_teacher"], "V_null": t["V_null"],
                "teacher_efficiency": t["teacher_efficiency"],
                "null_efficiency": t["null_efficiency"],
                "dM_oracle_actual": t["dM_oracle_actual"],
                "margin_closure": (None if t["M_base"] == 0 else
                                   t["dM_oracle_actual"] / (-float(t["M_base"]))),
                "top1_flip_under_oracle": t["top1_flip_under_oracle"],
                "oracle_l1_total": t["oracle_l1_total"],
                "teacher_l1_total": t["teacher_l1_total"],
                "grad_connected_per_layer": t["grad_connected_per_layer"],
                "per_layer_budget_l1": {L: v["budget_l1_per_head"]
                                        for L, v in t["layers"].items()},
                "per_layer_g_abs_sum": {L: v["g_abs_sum_per_head"]
                                        for L, v in t["layers"].items()},
            },
            "R0_identity": {
                "incremental_framing_parent": {"dM": d["dM_teacher"],
                                               "dlogp": d["dlogp_T_teacher"],
                                               "cos": d["cos_teacher"]},
                "same_framing_r1_pilot": None if p is None else p["arms"]["R0"],
                "V_first_order": None if p is None else p["V_first_order"]["R0"],
                "l1_first_order": None if p is None else p["l1_first_order"]["R0"],
            },
            "R1_identity": None if p is None else {
                "same_framing_r1_pilot": p["arms"]["R1"],
                "V_first_order": p["V_first_order"]["R1"],
                "l1_first_order": p["l1_first_order"]["R1"],
                "status": "RETIRED_NEGATIVE_CONTROL (R1-FAIL)",
            },
            "Null_identity": {
                "incremental_framing_parent": {"dM": d["dM_null"], "cos": d["cos_null"]},
                "same_framing_r1_pilot": None if p is None else p["arms"]["NULL"],
                "V_first_order": None if p is None else p["V_first_order"]["Null"],
                "l1_first_order": None if p is None else p["l1_first_order"]["Null"],
            },
            "identity_checks": checks,
            "identity_ok": all(checks.values()),
        }
        rows.append(entry)
        if not entry["identity_ok"]:
            mismatches.append({"sample_id": sid, "position": pos,
                               "failed": [k for k, v in checks.items() if not v]})

    availability = {
        "reusable_as_persisted_evidence": [
            "prefix_sha256 + prefix token ids (re-derived and hash-verified here)",
            "y_S, y_T",
            "Base margin identity (M_base in two independent runs)",
            "V_Oracle, teacher/null efficiency, oracle_l1_total, per-layer per-head L1 budgets",
            "actual Oracle dM, margin closure, top-1 flip outcome, grad-connected flags",
            "R0 / R1 / Null per-arm outcomes and per-head first-order values",
        ],
        "NOT_persisted_recompute_required": {
            "student_local_gradient_g": {
                "why": "Track A persisted only per-head |g|_1 sums and V values, not the raw "
                       "gradient tensor",
                "recompute": "Track A pass B (autograd graph bounded to the 4 patched layers, "
                             "retain_grad + backward) — deterministic",
                "cost": "~4 s/position, 67 positions => ~4.5 min on one GPU, ~1.2 min on four",
                "verification_hook": "the recomputed per-head |g|_1 must reproduce the persisted "
                                     "g_abs_sum_per_head exactly, and V_teacher/V_null must "
                                     "reproduce the persisted values",
            },
            "oracle_direction_d_O": {
                "why": "only the per-head L1 budgets and V_Oracle were persisted",
                "recompute": "deterministic greedy mass transfer from the recomputed g under the "
                             "same per-head budget",
                "verification_hook": "per-head L1 must equal the persisted budget_l1_per_head",
            },
            "R1_direction_d_R1_raw": {
                "why": "the R1 pilot persisted per-arm L1 totals and first-order values only",
                "recompute": "R1 mapped priors ARE persisted (r1_mapped_priors.pt), so d_R1 can be "
                             "rebuilt by the frozen mixing function without another Teacher pass",
            },
        },
    }
    payload = {
        "task_id": "RESCUE-DECISION-AWARE-TEACHER-SIGNAL-SPRINT-001",
        "phase": "A — preparation, no GPU attribution",
        "sources": {"parent_diagnostic": args.dr_rows, "track_a": args.ta_rows,
                    "r1_pilot": args.r1_rows, "ledger": args.validation_ledger},
        "population": "the 67 S!=T decision positions",
        "population_planned": len(pop_keys), "population_matched": len(rows) - len(mismatches),
        "identity_gate": "67/67" if not mismatches else f"{len(rows)-len(mismatches)}/{len(rows)}",
        "identity_gate_pass": not mismatches,
        "mismatches": mismatches,
        "fixed_before_attribution": True,
        "note": ("this population and its identities were frozen before any Teacher attribution "
                 "value was computed; it must not be reselected after seeing attribution results"),
        "reuse_availability": availability,
        "rows": rows,
    }
    (out / "teacher_attribution_population.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"population_planned": len(pop_keys), "matched": len(rows) - len(mismatches),
                      "identity_gate": payload["identity_gate"],
                      "identity_gate_pass": payload["identity_gate_pass"],
                      "mismatches": mismatches[:5],
                      "recompute_required": list(availability[
                          "NOT_persisted_recompute_required"].keys())},
                     ensure_ascii=False, indent=2))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
