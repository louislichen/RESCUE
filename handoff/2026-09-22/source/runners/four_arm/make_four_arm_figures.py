"""§21 required figures for RESCUE-MINIMAL-FOUR-ARM-VALID-RERUN-001.

Real data only, from the aggregated outputs:
  qa_four_arm_summary.json / k4_four_arm_summary.json / four_arm_results.json
  qa_paired_per_sample.csv  / k4_paired_per_sample.csv

Produces (PNG + PDF):
  fig1_qa_em_f1_by_arm
  fig2_k4_acceptance_by_arm
  fig3_paired_accepted_delta_CB_CD
  fig4_qa_em_transitions
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARMS = ["A_base", "B_prefill_only_teacher", "C_full_window_teacher", "D_full_window_null"]
SHORT = {"A_base": "A\nBASE", "B_prefill_only_teacher": "B\nPREFILL\nTEACHER",
         "C_full_window_teacher": "C\nFULL\nTEACHER", "D_full_window_null": "D\nFULL\nNULL"}
COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="aggregated output directory")
    ap.add_argument("--out", required=True, help="figure output directory")
    args = ap.parse_args()
    d = Path(args.dir)
    o = Path(args.out)
    o.mkdir(parents=True, exist_ok=True)
    res = json.loads((d / "four_arm_results.json").read_text(encoding="utf-8"))
    qa = res["qa"]["per_arm"]
    k4 = res["k4"]["per_arm"]
    qa_pairs = res["qa"]["pairwise"]
    n = res["valid_paired_N"]

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(o / f"{name}.png", dpi=170)
        fig.savefig(o / f"{name}.pdf")
        plt.close(fig)
        print("wrote", o / f"{name}.png")

    # ---- fig 1: QA EM / F1 by arm -------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = range(len(ARMS))
    em = [qa[a]["EM"] for a in ARMS]
    f1 = [qa[a]["F1"] for a in ARMS]
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], em, w, label="EM", color="#4c72b0")
    b2 = ax.bar([i + w / 2 for i in x], f1, w, label="F1", color="#dd8452")
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.3f}", (r.get_x() + r.get_width() / 2,
                                                  r.get_height()),
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([SHORT[a] for a in ARMS], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(f"QA EM / F1 by arm (development, paired N={n})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig1_qa_em_f1_by_arm")

    # ---- fig 2: K4 acceptance by arm ----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ratio = [(k4[a]["pooled"]["acceptance_ratio"] or 0.0) for a in ARMS]
    per_sample = [k4[a]["acceptance_rate_per_sample_mean"] for a in ARMS]
    axes[0].bar(range(len(ARMS)), ratio, color=COLORS)
    for i, v in enumerate(ratio):
        axes[0].annotate(f"{v:.3f}", (i, v), ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(range(len(ARMS)))
    axes[0].set_xticklabels([SHORT[a] for a in ARMS], fontsize=8)
    axes[0].set_ylabel("accepted / proposed (pooled)")
    axes[0].set_title("K=4 pooled acceptance ratio")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(range(len(ARMS)), per_sample, color=COLORS)
    for i, v in enumerate(per_sample):
        axes[1].annotate(f"{v:.3f}", (i, v), ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(range(len(ARMS)))
    axes[1].set_xticklabels([SHORT[a] for a in ARMS], fontsize=8)
    axes[1].set_ylabel("mean per-sample acceptance rate")
    axes[1].set_title("K=4 per-sample acceptance rate (paired unit)")
    axes[1].grid(axis="y", alpha=0.3)
    save(fig, "fig2_k4_acceptance_by_arm")

    # ---- fig 3: paired per-sample accepted-length deltas ---------------
    rows = read_csv(d / "k4_paired_per_sample.csv")
    cb = [int(r["C_minus_B_accepted"]) for r in rows]
    cd = [int(r["C_minus_D_accepted"]) for r in rows]
    rb = [float(r["C_minus_B_rate"]) for r in rows]
    rd = [float(r["C_minus_D_rate"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.0))

    lo = min(min(cb), min(cd))
    hi = max(max(cb), max(cd))
    bins = [b - 0.5 for b in range(lo, hi + 2)]
    axes[0].hist([cb, cd], bins=bins, label=["C − B", "C − D"],
                 color=["#55a868", "#c44e52"], edgecolor="black", linewidth=0.4)
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("paired per-sample accepted-token delta")
    axes[0].set_xlabel("accepted draft tokens(high) − (low)")
    axes[0].set_ylabel("samples")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].hist([rb, rd], bins=25, label=["C − B", "C − D"],
                 color=["#55a868", "#c44e52"], edgecolor="black", linewidth=0.4)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("paired per-sample acceptance-rate delta")
    axes[1].set_xlabel("rate(high) − rate(low)")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    for a, col in zip(ARMS, COLORS):
        axes[2].plot(range(len(rows)), [int(r[f"{a}_accepted"]) for r in rows],
                     color=col, linewidth=1.0, alpha=0.85, label=a.split("_")[0])
    axes[2].set_title("per-sample accepted draft tokens (all four arms)")
    axes[2].set_xlabel("sample position (ledger order)")
    axes[2].set_ylabel("accepted draft tokens")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    save(fig, "fig3_paired_accepted_delta_CB_CD")

    # ---- fig 4: QA EM transitions -------------------------------------
    pairs = ["B−A", "C−B", "C−D", "C−A", "D−A"]
    keys = ["better", "worse", "both_correct", "both_wrong"]
    labels = ["better (low wrong → high right)", "worse (low right → high wrong)",
              "both correct", "both wrong"]
    colors = ["#55a868", "#c44e52", "#4c72b0", "#b0b0b0"]
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    bottom = [0] * len(pairs)
    for k, lab, col in zip(keys, labels, colors):
        vals = [qa_pairs[p]["em_transitions"][k] for p in pairs]
        ax.bar(pairs, vals, bottom=bottom, label=lab, color=col)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, tot in enumerate(bottom):
        ax.annotate(f"n={tot}", (i, tot), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("samples")
    ax.set_title("QA EM transitions per contrast (paired)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig4_qa_em_transitions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
