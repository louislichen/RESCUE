"""§15 required figures for the decision-relevance diagnostic (real data only).

  fig1  Teacher vs Null Target-margin delta (all positions)
  fig2  S!=T subset paired delta
  fig3  prior-space distance vs decision-margin improvement
  fig4  linearized boundary-distance distribution
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

T_C, N_C = "#dd8452", "#4c72b0"


def read_csv(p: Path) -> list[dict]:
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fnum(v):
    return None if v in ("", None) else float(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d, o = Path(args.dir), Path(args.out)
    o.mkdir(parents=True, exist_ok=True)
    summary = json.loads((d / "decision_relevance_summary.json").read_text(encoding="utf-8"))
    rows = read_csv(d / "teacher_vs_null_paired.csv")
    prior = read_csv(d / "prior_distance_vs_decision_value.csv")

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(o / f"{name}.png", dpi=170)
        fig.savefig(o / f"{name}.pdf")
        plt.close(fig)
        print("wrote", o / f"{name}.png")

    def panel(ax, rs, title):
        t = [fnum(r["dM_teacher"]) for r in rs]
        n = [fnum(r["dM_null"]) for r in rs]
        lo = min(min(t), min(n))
        hi = max(max(t), max(n))
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        bins = [lo + (hi - lo) * i / 40 for i in range(41)]
        ax.hist(t, bins=bins, alpha=0.62, color=T_C, label="Teacher (C − A)")
        ax.hist(n, bins=bins, alpha=0.62, color=N_C, label="Null (D − A)")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        st = summary["groups"]["S!=T" if rs is snt_rows else "S=T"]["dM_teacher"]
        ax.set_title(f"{title}  (n={len(rs)})\n"
                     f"teacher mean {st['mean']:+.4f} CI [{st['ci95_low']:+.4f}, "
                     f"{st['ci95_high']:+.4f}]", fontsize=9)
        ax.set_xlabel("Δ target-token margin (logits)")
        ax.set_ylabel("decision positions")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    snt_rows = [r for r in rows if r["group"] == "S!=T"]
    st_rows = [r for r in rows if r["group"] == "S=T"]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    panel(ax, rows, "Teacher vs Null: target-margin movement, all positions")
    save(fig, "fig1_teacher_vs_null_margin_delta")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    if snt_rows:
        panel(axes[0], snt_rows, "S≠T (Student top1 ≠ Target top1)")
    if st_rows:
        panel(axes[1], st_rows, "S=T (already agreeing)")
    save(fig, "fig2_ST_subset_paired_delta")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for grp, col, mk in (("S!=T", "#c44e52", "o"), ("S=T", "#55a868", "x")):
        xs = [fnum(r["teacher_l1"]) for r in prior if r["group"] == grp]
        ys = [fnum(r["dM_teacher"]) for r in prior if r["group"] == grp]
        if xs:
            ax.scatter(xs, ys, s=14, alpha=0.6, color=col, marker=mk, label=grp)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    sp = summary["prior_space"].get("spearman_teacher_l1_vs_dM_teacher")
    sp_snt = summary["prior_space"].get("S!=T_spearman_teacher_l1_vs_dM_teacher")
    ax.set_xlabel("teacher prior vs Student attention row  (mean L1 over layers)")
    ax.set_ylabel("Δ target-token margin under the Teacher arm")
    ax.set_title("prior-space distance vs decision usefulness\n"
                 f"Spearman all={sp if sp is None else round(sp, 3)} · "
                 f"S≠T={sp_snt if sp_snt is None else round(sp_snt, 3)}  (association, not causation)",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "fig3_prior_distance_vs_decision_value")

    lin = [fnum(r["linearized_scale_teacher"]) for r in rows
           if fnum(r["linearized_scale_teacher"]) is not None]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    if lin:
        ax.hist(lin, bins=30, color="#8172b3", edgecolor="black", linewidth=0.4)
        ax.set_xlabel("estimated scale to reach the Target decision boundary "
                      "(linearized, diagnostic only)")
        ax.set_ylabel("decision positions")
        ax.set_title(f"Linearized boundary distance\nn={len(lin)} positions with S≠T and ΔM>0 "
                     f"· median {sorted(lin)[len(lin)//2]:.1f}×", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "not defined (no position with S≠T and ΔM_teacher > 0)",
                ha="center", va="center")
        ax.set_title("Linearized boundary distance")
    save(fig, "fig4_linearized_boundary_distance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
