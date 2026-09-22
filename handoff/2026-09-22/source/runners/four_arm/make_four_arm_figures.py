"""Figures for the valid four-arm development rerun."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ARMS=["A_base","B_prefill_only_teacher","C_full_window_teacher","D_full_window_null"]
def read_csv(p):
    with p.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--dir",required=True);ap.add_argument("--out",required=True);args=ap.parse_args()
    d,o=Path(args.dir),Path(args.out);o.mkdir(parents=True,exist_ok=True);res=json.loads((d/"four_arm_results.json").read_text(encoding="utf-8"))
    def save(fig,name):fig.tight_layout();fig.savefig(o/f"{name}.png",dpi=170);fig.savefig(o/f"{name}.pdf");plt.close(fig)
    qa=res["qa"]["per_arm"];fig,ax=plt.subplots(figsize=(7.2,4.2));x=range(len(ARMS));w=.38
    ax.bar([i-w/2 for i in x],[qa[a]["EM"] for a in ARMS],w,label="EM");ax.bar([i+w/2 for i in x],[qa[a]["F1"] for a in ARMS],w,label="F1");ax.legend();save(fig,"fig1_qa_em_f1_by_arm")
    k4=res["k4"]["per_arm"];fig,ax=plt.subplots(figsize=(7.2,4.2));ax.bar(range(4),[(k4[a]["pooled"]["acceptance_ratio"] or 0) for a in ARMS]);save(fig,"fig2_k4_acceptance_by_arm")
    rows=read_csv(d/"k4_paired_per_sample.csv");fig,ax=plt.subplots(figsize=(7.2,4.2));ax.hist([[int(r["C_minus_B_accepted"]) for r in rows],[int(r["C_minus_D_accepted"]) for r in rows]],bins=20);save(fig,"fig3_paired_accepted_delta_CB_CD")
    return 0
if __name__=="__main__":raise SystemExit(main())
