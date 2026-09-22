"""§17 required figures for RESCUE-R1-VALUE-NORM-DECISION-ALIGNMENT-PILOT-001 (real data only)."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
COLS={"R0":"#dd8452","R1":"#55a868","Null":"#4c72b0"}
def rows(p):
    with p.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def fnum(v):
    try:return float(v)
    except (TypeError,ValueError):return None
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--dir",required=True);ap.add_argument("--out",required=True);args=ap.parse_args()
    d,o=Path(args.dir),Path(args.out);o.mkdir(parents=True,exist_ok=True)
    summ=json.loads((d/"r1_decision_alignment_summary.json").read_text(encoding="utf-8"))
    rvn=rows(d/"r1_vs_r0_vs_null.csv");eff=rows(d/"r1_oracle_efficiency.csv");link=rows(d/"r1_semantic_decision_link.csv")
    def save(fig,name):
        fig.tight_layout();fig.savefig(o/f"{name}.png",dpi=170);fig.savefig(o/f"{name}.pdf");plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.8,4.4))
    data=[[fnum(r["dM_R0"]) for r in rvn],[fnum(r["dM_R1"]) for r in rvn],[fnum(r["dM_Null"]) for r in rvn]]
    ax.boxplot(data,tick_labels=["R0 attention","R1 attention×|V|","Null"],showfliers=True);ax.axhline(0,color="black",linestyle="--");save(fig,"fig1_delta_margin_by_arm")
    fig,ax=plt.subplots(figsize=(7.8,4.4))
    data=[[fnum(r["eff_R0"]) for r in eff],[fnum(r["eff_R1"]) for r in eff],[fnum(r["eff_Null"]) for r in eff]]
    ax.boxplot(data,tick_labels=["R0","R1","Null"],showfliers=True);ax.axhline(0,color="black",linestyle="--");ax.axhline(.10,color="red",linestyle="-.");save(fig,"fig2_oracle_efficiency_by_arm")
    fig,ax=plt.subplots(figsize=(7.8,4.4))
    data=[[fnum(r["cos_R0"]) for r in rvn],[fnum(r["cos_R1"]) for r in rvn],[fnum(r["cos_Null"]) for r in rvn]]
    ax.boxplot(data,tick_labels=["R0","R1","Null"],showfliers=True);ax.axhline(0,color="black",linestyle="--");save(fig,"fig3_direction_cosine_by_arm")
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.4));xs=[fnum(r["evidence_mass_R1"]) for r in link]
    axes[0].scatter(xs,[fnum(r["dM_R1"]) for r in link],s=26);axes[1].scatter(xs,[fnum(r["V_R1_over_V_oracle"]) for r in link],s=26);save(fig,"fig4_r1_semantic_vs_decision")
    return 0
if __name__=="__main__":raise SystemExit(main())
