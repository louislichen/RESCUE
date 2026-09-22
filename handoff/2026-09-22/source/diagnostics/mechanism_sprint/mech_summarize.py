"""Track B / Track C summaries and the Track E comparison tables (CPU, real data only)."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dr_common import bootstrap_ci, load_harness  # noqa: E402


def st(vals):
    if not vals:
        return None
    s = bootstrap_ci(vals)
    s["min"] = min(vals)
    s["max"] = max(vals)
    return s


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_envelopes(H, out, pattern, config_hash, ckpt):
    recs, integral = [], []
    for p in sorted(out.glob(pattern)):
        v = H.verify_result(p, expected_config_hash=config_hash,
                            expected_checkpoint_sha256=ckpt)
        integral.append({"path": str(p), "verify": v})
        if v["status"] == "PASS":
            recs.extend(H.read_result_records(p))
    return recs, integral


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ta-summary", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    H = load_harness(args.harness_root)
    ta = json.loads(Path(args.ta_summary).read_text(encoding="utf-8"))

    # ---------------- Track B ----------------
    tb, tb_int = read_envelopes(H, out, "tb_shard_*.jsonl", cfg["config_hash"],
                                cfg["checkpoint_sha256"])
    repr_b = {}
    for name in ("R0_TEACHER_ATTENTION", "R1_TEACHER_ATTN_VALUE_NORM"):
        got = [r for r in tb if name in r.get("representations", {})]
        if not got:
            continue
        def col(k, sub=None):
            vals = []
            for r in got:
                d = r["representations"][name]
                v = d.get(k) if sub is None else d
                if isinstance(v, dict):
                    v = v.get(k)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            return vals
        repr_b[name] = {
            "samples": len(got),
            "entropy": st(col("entropy")),
            "topK_mass_k16": st(col("topK_mass")),
            "alpha_mean": st(col("alpha_mean")),
            "l1_vs_student_attention": st(col("l1_vs_student_attention")),
            "jsd_vs_student_attention": st(col("jsd_vs_student_attention")),
            "finite_all": all(r["representations"][name].get("finite", True) for r in got),
            "non_negative_all": all(r["representations"][name].get("non_negative", True)
                                    for r in got),
        }
    r1rows = [r["representations"]["R1_TEACHER_ATTN_VALUE_NORM"] for r in tb
              if "R1_TEACHER_ATTN_VALUE_NORM" in r.get("representations", {})]
    repr_b["R1_TEACHER_ATTN_VALUE_NORM"].update({
        "topK_jaccard_vs_R0": st([r["topK_jaccard_vs_R0"] for r in r1rows
                                  if r.get("topK_jaccard_vs_R0") is not None]),
        "l1_vs_R0": st([r["l1_vs_R0"] for r in r1rows if r.get("l1_vs_R0") is not None]),
        "jsd_vs_R0": st([r["jsd_vs_R0"] for r in r1rows if r.get("jsd_vs_R0") is not None]),
    })
    tb_meta = {}
    for p in sorted(out.glob("tb_shard_*_meta.json")):
        tb_meta.update(json.loads(p.read_text(encoding="utf-8")))
    summary_b = {
        "task_id": cfg["task_id"], "track": "B — Teacher representation decomposition",
        "samples": len(tb), "integrity": tb_int,
        "granularity": tb_meta.get("granularity"),
        "teacher_layers": tb_meta.get("teacher_layers"),
        "top_k": tb_meta.get("top_k", 16),
        "representations": repr_b,
        "R2_status": ("IDENTICAL_TO_R1_BEFORE_NORMALISATION: ||attention_j * V_j|| == "
                      "attention_j * ||V_j||, so the candidate set collapses to two distinct "
                      "Teacher attention reweightings plus the Null control"),
        "does_not_prove": "these are representation diagnostics; they do NOT establish decision "
                          "usefulness — that is measured in Track A and Track C",
        "claim_boundary": cfg["claim_boundary"],
    }
    H.atomic_write_text(out / "teacher_representation_summary.json",
                        json.dumps(summary_b, ensure_ascii=False, indent=2) + "\n")

    # ---------------- Track C ----------------
    tc, tc_int = read_envelopes(H, out, "tc_shard_*.jsonl", cfg["config_hash"],
                                cfg["checkpoint_sha256"])
    tc_meta = []
    for p in sorted(out.glob("tc_shard_*_meta.json")):
        tc_meta.append(json.loads(p.read_text(encoding="utf-8")))
    names = ["R0_TEACHER_ATTENTION_student_space", "R0_TEACHER_ATTENTION_teacher_space",
             "R1_TEACHER_ATTN_VALUE_NORM_teacher_space", "NULL_student_space",
             "NULL_teacher_space", "STUDENT_ATTENTION_student_space"]
    groups = {"all": tc,
              "has_S!=T": [r for r in tc if "S!=T" in (r.get("decision_positions") or [])],
              "S=T_only": [r for r in tc if "S!=T" not in (r.get("decision_positions") or [])]}
    repr_c = {}
    for name in names:
        per_group = {}
        for gname, rs in groups.items():
            got = [r["representations"][name] for r in rs
                   if name in r.get("representations", {})]
            if not got:
                per_group[gname] = {"n": 0}
                continue
            per_group[gname] = {
                "n": len(got),
                "evidence_mass": st([g["evidence_mass"] for g in got]),
                "evidence_to_non_evidence_ratio": st(
                    [g["evidence_to_non_evidence_ratio"] for g in got
                     if g["evidence_to_non_evidence_ratio"] is not None]),
                "topk_evidence_recall": st([g["topk_evidence_recall"] for g in got]),
                "topk_evidence_fraction": st([g["topk_evidence_fraction"] for g in got]),
                "entropy_within_evidence": st([g["entropy_within_evidence"] for g in got]),
            }
        repr_c[name] = per_group
    with (out / "supporting_fact_grounding.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["sample_id", "source_index", "has_S_ne_T", "evidence_tokens_student",
                  "evidence_tokens_teacher"]
        for name in names:
            header += [f"{name}__evidence_mass", f"{name}__ratio",
                       f"{name}__topk_recall", f"{name}__topk_fraction"]
        w.writerow(header)
        for r in sorted(tc, key=lambda x: int(x.get("source_index") or 0)):
            row = [r["canonical_source_id"], r.get("source_index"),
                   int("S!=T" in (r.get("decision_positions") or [])),
                   r.get("evidence_token_count_student"), r.get("evidence_token_count_teacher")]
            for name in names:
                g = r.get("representations", {}).get(name)
                if g:
                    row += [round(g["evidence_mass"], 6),
                            "" if g["evidence_to_non_evidence_ratio"] is None
                            else round(g["evidence_to_non_evidence_ratio"], 6),
                            round(g["topk_evidence_recall"], 6),
                            round(g["topk_evidence_fraction"], 6)]
                else:
                    row += ["", "", "", ""]
            w.writerow(row)
    excluded_total = sum(m.get("excluded_samples", 0) for m in tc_meta)
    summary_c = {
        "task_id": cfg["task_id"], "track": "C — semantic evidence grounding",
        "samples_measured": len(tc), "samples_excluded_no_mappable_fact": excluded_total,
        "exclusion_examples": [e for m in tc_meta for e in m.get("exclusion_examples", [])][:10],
        "spaces": tc_meta[0].get("spaces") if tc_meta else None,
        "top_k": tc_meta[0].get("top_k") if tc_meta else None,
        "representations": repr_c,
        "S_eq_T_vs_S_ne_T": ("the same representation is reported separately for samples whose "
                             "decision positions include an S!=T position and for samples with "
                             "S=T positions only"),
        "R2": "identical to R1 before normalisation (see Track B)",
        "integrity": tc_int,
        "claim_boundary": cfg["claim_boundary"],
    }
    H.atomic_write_text(out / "semantic_grounding_summary.json",
                        json.dumps(summary_c, ensure_ascii=False, indent=2) + "\n")

    # ---------------- Track E table ----------------
    pa = ta["primary_population"]
    grp = {}
    fs = out / "failure_structure.csv"
    if fs.exists():
        for r in csv.DictReader(fs.open(encoding="utf-8")):
            if r["bin_family"] == "group":
                grp[r["bin"]] = {k: fnum(v) for k, v in r.items()}
    g_snt = grp.get("S!=T", {})
    g_seq = grp.get("S=T", {})

    def fmt(v, nd=4):
        return "n/a" if v is None else f"{v:.{nd}f}"
    r0 = repr_b.get("R0_TEACHER_ATTENTION", {})
    r1 = repr_b.get("R1_TEACHER_ATTN_VALUE_NORM", {})
    g0 = repr_c.get("R0_TEACHER_ATTENTION_student_space", {}).get("all", {})
    g1 = repr_c.get("R1_TEACHER_ATTN_VALUE_NORM_teacher_space", {}).get("all", {})
    null_mass = (((repr_c.get("NULL_student_space") or {}).get("all") or {})
                 .get("evidence_mass") or {}).get("mean")
    lines = [
        "# REPRESENTATION_CANDIDATE_COMPARISON",
        "",
        "All numbers are development diagnostics from this sprint; none of them establishes a "
        "method result.",
        "",
        "| Representation | Decision alignment (Track A) | Oracle efficiency (Track A) | "
        "Evidence grounding (Track C) | Representation stats (Track B) | Cost | Status |",
        "|---|---|---|---|---|---|---|",
        f"| Current attention prior (R0) | Teacher ΔM on S≠T "
        f"{fmt(g_snt.get('dM_teacher_mean'))} "
        f"[{fmt(g_snt.get('dM_teacher_ci_low'))}, {fmt(g_snt.get('dM_teacher_ci_high'))}]; "
        f"V_T {fmt(pa['V_teacher']['mean'], 5)} "
        f"[{fmt(pa['V_teacher']['ci95_low'], 5)}, {fmt(pa['V_teacher']['ci95_high'], 5)}] | "
        f"{fmt(pa['teacher_efficiency']['median'], 5)} "
        f"[{fmt(pa['teacher_efficiency']['ci95_low'], 5)}, "
        f"{fmt(pa['teacher_efficiency']['ci95_high'], 5)}] | evidence mass "
        f"{fmt((g0.get('evidence_mass') or {}).get('mean'))}; top-k recall "
        f"{fmt((g0.get('topk_evidence_recall') or {}).get('mean'))} | entropy "
        f"{fmt((r0.get('entropy') or {}).get('mean'))}; top-16 mass "
        f"{fmt((r0.get('topK_mass_k16') or {}).get('mean'))}; L1 to Student "
        f"{fmt((r0.get('l1_vs_student_attention') or {}).get('mean'))} | zero (already computed) "
        f"| baseline; decision-irrelevant |",
        f"| Attention × value norm (R1) | not measured directly in Track A | not measured "
        f"directly in Track A | evidence mass "
        f"{fmt((g1.get('evidence_mass') or {}).get('mean'))}; top-k recall "
        f"{fmt((g1.get('topk_evidence_recall') or {}).get('mean'))} | entropy "
        f"{fmt((r1.get('entropy') or {}).get('mean'))}; top-16 mass "
        f"{fmt((r1.get('topK_mass_k16') or {}).get('mean'))}; top-16 Jaccard vs R0 "
        f"{fmt((r1.get('topK_jaccard_vs_R0') or {}).get('mean'))}; L1 vs R0 "
        f"{fmt((r1.get('l1_vs_R0') or {}).get('mean'))}; L1 to Student "
        f"{fmt((r1.get('l1_vs_student_attention') or {}).get('mean'))} | one 7B forward + value "
        f"hook | diagnostic only; NOT injected or tested |",
        f"| Context contribution (R2) | identical to R1 before normalisation | identical to R1 | "
        f"identical to R1 | identical to R1 | zero extra | collapses into R1 |",
        f"| Null control | ΔM {fmt(g_snt.get('dM_null_mean'))} "
        f"[{fmt(g_snt.get('dM_null_ci_low'))}, {fmt(g_snt.get('dM_null_ci_high'))}] | "
        f"{fmt(pa['null_efficiency']['median'], 5)} "
        f"[{fmt(pa['null_efficiency']['ci95_low'], 5)}, "
        f"{fmt(pa['null_efficiency']['ci95_high'], 5)}] | evidence mass "
        f"{fmt(null_mass)} | — | zero | control |",
        f"| Same-budget local Oracle (diagnostic) | ΔM {fmt(pa['dM_oracle_actual']['mean'])} "
        f"[{fmt(pa['dM_oracle_actual']['ci95_low'])}, "
        f"{fmt(pa['dM_oracle_actual']['ci95_high'])}]; median closure "
        f"{fmt(pa['margin_closure_fraction']['median'])}; top-1 flip rate "
        f"{fmt(pa['oracle_top1_flip_rate'])} | 1.0 by construction | not applicable | — | "
        f"diagnostic only | **LOCAL_DECISION_ORACLE_DIAGNOSTIC — not deployable** |",
        "",
        "Budget control: `||d_oracle||_1 / ||d_teacher||_1` = "
        f"{fmt(pa['budget_ratio_oracle_over_teacher']['mean'], 6)} (1.0 = exactly matched), "
        "per head, with simplex and non-negativity asserted per layer.",
    ]
    out.joinpath("representation_candidate_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"track_b_samples": len(tb), "track_c_samples": len(tc),
                      "track_c_excluded": excluded_total,
                      "R1_jaccard_vs_R0": (r1.get("topK_jaccard_vs_R0") or {}).get("mean"),
                      "R1_evidence_mass": (g1.get("evidence_mass") or {}).get("mean"),
                      "R0_evidence_mass": (g0.get("evidence_mass") or {}).get("mean"),
                      "student_evidence_mass": ((repr_c.get(
                          "STUDENT_ATTENTION_student_space", {}).get("all") or {})
                          .get("evidence_mass") or {}).get("mean")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
