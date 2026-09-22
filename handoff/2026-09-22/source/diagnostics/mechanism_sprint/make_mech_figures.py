"""§Required figures for RESCUE-GUIDANCE-REPRESENTATION-MECHANISM-SPRINT-001 (real data only).

  1  Teacher / Null / Oracle local controllability value
  2  Teacher efficiency relative to the same-budget Oracle
  3  Candidate representation evidence-grounding comparison
  4  Decision alignment vs Student margin deficit
  5  S=T vs S!=T mechanism comparison
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

T_C, N_C, O_C = "#dd8452", "#4c72b0", "#55a868"


def rows(p: Path):
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fnum(v):
    return None if v in ("", None) else float(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dr-summary", default="")
    args = ap.parse_args()
    d, o = Path(args.dir), Path(args.out)
    o.mkdir(parents=True, exist_ok=True)
    ta = json.loads((d / "student_controllability_summary.json").read_text(encoding="utf-8"))
    dr = json.loads(Path(args.dr_summary).read_text(encoding="utf-8")) \
        if args.dr_summary else {}
    sg = json.loads((d / "semantic_grounding_summary.json").read_text(encoding="utf-8"))
    orc = rows(d / "oracle_same_budget_results.csv")
    snt = [r for r in orc if r["group"] == "S!=T"]

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(o / f"{name}.png", dpi=170)
        fig.savefig(o / f"{name}.pdf")
        plt.close(fig)
        print("wrote", o / f"{name}.png")

    # ---- fig 1: controllability values ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    vals = [[float(r["V_teacher"]) for r in snt], [float(r["V_null"]) for r in snt],
            [float(r["V_oracle"]) for r in snt]]
    axes[0].boxplot(vals, tick_labels=["Teacher", "Null", "Oracle (same budget)"],
                    showfliers=True)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("first-order value  g · d")
    axes[0].set_title(f"Local controllability value, S≠T (n={len(snt)})")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].hist([float(r["dM_oracle_actual"]) for r in snt], bins=18, color=O_C,
                 edgecolor="black", linewidth=0.4)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Δ target-token margin under the same-budget Oracle")
    axes[1].set_ylabel("decision positions")
    axes[1].set_title("ACTUAL forward margin movement (A8)")
    axes[1].grid(axis="y", alpha=0.3)
    save(fig, "fig1_controllability_values")

    # ---- fig 2: teacher efficiency -------------------------------------
    eff = [fnum(r["teacher_efficiency"]) for r in snt if fnum(r["teacher_efficiency"])
           is not None]
    neff = [fnum(r["null_efficiency"]) for r in snt if fnum(r["null_efficiency"]) is not None]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].hist(eff, bins=22, color=T_C, edgecolor="black", linewidth=0.4, label="Teacher")
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].axvline(1, color="grey", linestyle=":", linewidth=1)
    axes[0].set_xlabel("Teacher efficiency  V_T / V_O")
    axes[0].set_ylabel("decision positions")
    axes[0].set_title(f"Teacher captures almost none of the achievable direction\n"
                      f"median {sorted(eff)[len(eff)//2]:+.4f}, "
                      f"{sum(1 for v in eff if v <= 0)}/{len(eff)} ≤ 0")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)
    clo = [fnum(r["margin_closure"]) for r in snt if fnum(r["margin_closure"]) is not None]
    axes[1].hist(clo, bins=22, color=O_C, edgecolor="black", linewidth=0.4)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].axvline(1, color="grey", linestyle=":", linewidth=1, label="boundary reached")
    axes[1].set_xlabel("Oracle margin closure  ΔM / (−M_base)")
    axes[1].set_ylabel("decision positions")
    axes[1].set_title(f"Same-budget Oracle closes the deficit\nmedian "
                      f"{sorted(clo)[len(clo)//2]:.3f}, top-1 flip rate "
                      f"{sum(1 for r in snt if r['top1_flip'] == '1')/len(snt):.3f}")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    save(fig, "fig2_teacher_efficiency_vs_oracle")

    # ---- fig 3: evidence grounding comparison --------------------------
    names = [("R0_TEACHER_ATTENTION_student_space", "R0 attention"),
             ("R1_TEACHER_ATTN_VALUE_NORM_teacher_space", "R1 attention×|V|"),
             ("NULL_student_space", "Null"),
             ("STUDENT_ATTENTION_student_space", "Student own")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    labels, mass, recall = [], [], []
    for key, lab in names:
        g = (sg["representations"].get(key) or {}).get("all") or {}
        if not g.get("n"):
            continue
        labels.append(f"{lab}\nn={g['n']}")
        mass.append(g["evidence_mass"]["mean"])
        recall.append(g["topk_evidence_recall"]["mean"])
    x = range(len(labels))
    axes[0].bar(x, mass, color=[T_C, "#8172b3", N_C, "#937860"])
    for i, v in enumerate(mass):
        axes[0].annotate(f"{v:.3f}", (i, v), ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("supporting-evidence probability mass")
    axes[0].set_title("Evidence grounding by representation")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(x, recall, color=[T_C, "#8172b3", N_C, "#937860"])
    for i, v in enumerate(recall):
        axes[1].annotate(f"{v:.3f}", (i, v), ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("top-16 evidence recall")
    axes[1].set_title("Top-16 attention lands on supporting facts")
    axes[1].grid(axis="y", alpha=0.3)
    save(fig, "fig3_evidence_grounding")

    # ---- fig 4: decision alignment vs margin deficit -------------------
    pos = d / "failure_structure_positions.csv"
    if pos.exists():
        pr = [r for r in rows(pos) if r["group"] == "S!=T"]
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        xs = [fnum(r["student_margin_deficit"]) for r in pr]
        ys = [fnum(r["dM_teacher"]) for r in pr]
        cs = [fnum(r["cos_teacher"]) for r in pr]
        sc = ax.scatter(xs, ys, c=cs, cmap="coolwarm", vmin=-0.5, vmax=0.5, s=28,
                        edgecolor="black", linewidth=0.3)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        fig.colorbar(sc, ax=ax, label="cos(Δz_teacher, z_Target − z_Base)")
        ax.set_xlabel("Student decision margin deficit  −M_A")
        ax.set_ylabel("Teacher Δ target-token margin")
        ax.set_title(f"Decision alignment vs deficit, S≠T (n={len(pr)})")
        ax.grid(alpha=0.3)
        save(fig, "fig4_alignment_vs_deficit")

    # ---- fig 5: S=T vs S!=T mechanism comparison -----------------------
    gsnt = dr["groups"]["S!=T"]
    gseq = dr["groups"]["S=T"]
    metrics = [("dM_teacher", "Teacher ΔM"), ("dM_null", "Null ΔM"),
               ("cos_teacher", "cos Teacher"), ("cos_null", "cos Null"),
               ("dlogp_T_teacher", "Teacher Δlogp")]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    w = 0.38
    xs = list(range(len(metrics)))
    a_vals = [gsnt[k]["mean"] for k, _ in metrics]
    b_vals = [gseq[k]["mean"] for k, _ in metrics]
    ax.bar([i - w / 2 for i in xs], a_vals, w, label=f"S≠T (n={gsnt['n']})", color="#c44e52")
    ax.bar([i + w / 2 for i in xs], b_vals, w, label=f"S=T (n={gseq['n']})", color="#4c72b0")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in metrics], fontsize=8)
    ax.set_ylabel("mean value")
    ax.set_title("S=T vs S≠T mechanism comparison (parent diagnostic, 496 positions)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig5_ST_vs_SneT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
