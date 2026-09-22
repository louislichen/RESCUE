"""§17 required figures for RESCUE-R1-VALUE-NORM-DECISION-ALIGNMENT-PILOT-001 (real data only).

  1  ΔTargetMargin: R0 vs R1 vs Null
  2  Oracle efficiency: R0 vs R1 vs Null
  3  Direction cosine comparison
  4  R1 semantic grounding vs decision value
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLS = {"R0": "#dd8452", "R1": "#55a868", "Null": "#4c72b0"}


def rows(p: Path):
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d, o = Path(args.dir), Path(args.out)
    o.mkdir(parents=True, exist_ok=True)
    summ = json.loads((d / "r1_decision_alignment_summary.json").read_text(encoding="utf-8"))
    rvn = rows(d / "r1_vs_r0_vs_null.csv")
    eff = rows(d / "r1_oracle_efficiency.csv")
    link = rows(d / "r1_semantic_decision_link.csv")

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(o / f"{name}.png", dpi=170)
        fig.savefig(o / f"{name}.pdf")
        plt.close(fig)
        print("wrote", o / f"{name}.png")

    # 1 dM
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    data = [[fnum(r["dM_R0"]) for r in rvn], [fnum(r["dM_R1"]) for r in rvn],
            [fnum(r["dM_Null"]) for r in rvn]]
    bp = ax.boxplot(data, tick_labels=["R0 attention", "R1 attention×|V|", "Null"],
                    showfliers=True, patch_artist=True)
    for patch, c in zip(bp["boxes"], [COLS["R0"], COLS["R1"], COLS["Null"]]):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Δ target-token margin")
    st = summ["per_arm"]
    ax.set_title("Δ target margin per arm, 67 S≠T positions\n"
                 f"means: R0 {st['R0']['dM']['mean']:+.4f} · R1 {st['R1']['dM']['mean']:+.4f} · "
                 f"Null {st['NULL']['dM']['mean']:+.4f}", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig1_delta_margin_by_arm")

    # 2 oracle efficiency
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    data = [[fnum(r["eff_R0"]) for r in eff], [fnum(r["eff_R1"]) for r in eff],
            [fnum(r["eff_Null"]) for r in eff]]
    bp = ax.boxplot(data, tick_labels=["R0", "R1", "Null"], showfliers=True, patch_artist=True)
    for patch, c in zip(bp["boxes"], [COLS["R0"], COLS["R1"], COLS["Null"]]):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1, label="Oracle = 1.0 reference")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axhline(0.10, color="red", linestyle="-.", linewidth=1,
               label="preregistered 10 % floor")
    ax.set_ylabel("V_arm / V_Oracle")
    ax.set_title("Oracle efficiency recovered per arm (same L1 budget)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig2_oracle_efficiency_by_arm")

    # 3 cosine
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    data = [[fnum(r["cos_R0"]) for r in rvn], [fnum(r["cos_R1"]) for r in rvn],
            [fnum(r["cos_Null"]) for r in rvn]]
    bp = ax.boxplot(data, tick_labels=["R0", "R1", "Null"], showfliers=True, patch_artist=True)
    for patch, c in zip(bp["boxes"], [COLS["R0"], COLS["R1"], COLS["Null"]]):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("cos(Δz, z_Target − z_Base), full vocabulary")
    ax.set_title("Direction alignment per arm, 67 S≠T positions", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig3_direction_cosine_by_arm")

    # 4 semantic vs decision
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    xs = [fnum(r["evidence_mass_R1"]) for r in link]
    for ax, ys, lab, key in ((axes[0], [fnum(r["dM_R1"]) for r in link],
                              "ΔM_R1", "spearman_evidence_mass_vs_dM_R1"),
                             (axes[1], [fnum(r["V_R1_over_V_oracle"]) for r in link],
                              "V_R1 / V_Oracle", "spearman_evidence_mass_vs_oracle_efficiency_R1")):
        ax.scatter(xs, ys, s=26, color=COLS["R1"], edgecolor="black", linewidth=0.3)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        rho = summ["semantic_decision_link"].get(key)
        ax.set_xlabel("R1 evidence mass (Track C, teacher space)")
        ax.set_ylabel(lab)
        ax.set_title(f"{lab} vs R1 semantic grounding\nSpearman "
                     f"{'n/a' if rho is None else round(rho, 3)} (association only)", fontsize=9)
        ax.grid(alpha=0.3)
    save(fig, "fig4_r1_semantic_vs_decision")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
