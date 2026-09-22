"""Mechanism-sprint figures from real aggregated outputs."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def rows(p):
    with p.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def fnum(v):return None if v in ("",None) else float(v)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--dir",required=True);ap.add_argument("--out",required=True);ap.add_argument("--dr-summary",default="");args=ap.parse_args()
    d,o=Path(args.dir),Path(args.out);o.mkdir(parents=True,exist_ok=True);sg=json.loads((d/"semantic_grounding_summary.json").read_text(encoding="utf-8"));orc=rows(d/"oracle_same_budget_results.csv");snt=[r for r in orc if r["group"]=="S!=T"]
    def save(fig,name):fig.tight_layout();fig.savefig(o/f"{name}.png",dpi=170);fig.savefig(o/f"{name}.pdf");plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.4));ax.boxplot([[float(r["V_teacher"]) for r in snt],[float(r["V_null"]) for r in snt],[float(r["V_oracle"]) for r in snt]],tick_labels=["Teacher","Null","Oracle"]);ax.axhline(0,color="black",linestyle="--");save(fig,"fig1_controllability_values")
    fig,ax=plt.subplots(figsize=(7.5,4.4));ax.hist([fnum(r["teacher_efficiency"]) for r in snt if fnum(r["teacher_efficiency"]) is not None],bins=22);ax.axvline(0,color="black",linestyle="--");save(fig,"fig2_teacher_efficiency_vs_oracle")
    return 0
if __name__=="__main__":raise SystemExit(main())
