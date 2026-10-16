"""Standalone render from saved plot_data.json; no engine/model imports."""
import argparse, json
from pathlib import Path
BUDGETS=(8, 16, 24, 32)
MODES=('csgb', 'constant', 'history_native')
POLICIES=('fixed8_native', 'fixed16_native', 'fixed24_native', 'fixed32_native', 'csgb', 'constant', 'history_native')
def render_figures(audit, out):
    """Render only recorded values; uncertainty unit is the paired document."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap
    summary=audit["summary"];smoke=summary["smoke"];pre=summary.get("pre_freeze_partial",False)
    caption=("TECHNICAL SMOKE: doc0 only, n=1, no CI; NOT efficacy" if smoke else
        "PARTIAL FIT/CALIBRATION: evolving observed phases only; no comparator, no CI" if pre else
        f"PARTIAL STUDY: 4K historical training-monitor, n={len(summary['complete_document_units'])}/8 docs; "
        "median2 timing repeats → mean3 action streams; paired-doc95% bootstrap CI")
    if audit.get("synthetic_qa"):caption="SYNTHETIC CPU QA — NOT EXPERIMENT DATA | "+caption
    figures=[]
    def save(fig,name):
        width=max(60,int(fig.get_figwidth()*18))
        footer=textwrap.fill(caption,width)+"\n"+textwrap.fill(
            "Eager FP32 Target / FP16 Draft; server-reported 48GB RTX4090; instrumented, NOT production speed",width)
        fig.text(.5,.015,footer,ha="center",va="bottom",fontsize=7)
        for ax in fig.axes:
            title_width=max(32,int(fig.get_figwidth()*ax.get_position().width*14))
            ax.set_title(textwrap.fill(ax.get_title(),title_width))
        fig.tight_layout(rect=(0,.20,1,1))
        for suffix in ("png","svg"):fig.savefig(out/f"{name}.{suffix}",dpi=170)
        plt.close(fig);figures.append(name)
    for metric,name,ylabel in (("acceptance","acceptance","Accepted Draft tokens / scored round" if smoke or pre else "Δ accepted Draft tokens / scored round"),
                                ("tps","request_tps","Whole-request useful output tokens/s" if smoke or pre else "Δ whole-request TPS (%)")):
        fig,ax=plt.subplots(figsize=(9,4.5));values=[];errors=[]
        for p in POLICIES:
            row=(summary["absolute_metrics"][p][metric] if smoke or pre else summary["comparisons_all_fixed"][summary["primary_baseline"]][p][metric])
            value=row["mean"];ci=row["ci95"];values.append(float("nan") if value is None else value)
            errors.append((0.,0.) if value is None or ci is None else (max(0.,value-ci[0]),max(0.,ci[1]-value)))
        ax.bar(range(7),values,color=["#8795a1"]*4+["#167a9c","#d79137","#9365a3"])
        ax.errorbar(range(7),values,yerr=list(zip(*errors)),fmt="none",ecolor="black",capsize=4)
        ax.set(xticks=range(7),xticklabels=POLICIES,ylabel=ylabel,
               title=("TECHNICAL SMOKE — " if smoke else "PARTIAL FIT STAGES — " if pre else "TRAINING-ONLY MONITOR — ")+name.replace("_"," ")+" (actual4096 prompt tokens)")
        ax.tick_params(axis="x",rotation=25);ax.axhline(0,color="black",lw=.7);ax.grid(axis="y",alpha=.2)
        if pre:ax.text(.02,.97,"PARTIAL evolving fit/calibration phases; no comparison",transform=ax.transAxes,va="top",fontsize=8)
        elif not smoke:ax.text(.02,.97,"Fit-only frozen comparator: "+summary["primary_baseline"],transform=ax.transAxes,va="top",fontsize=8)
        save(fig,name)
    rows=[r for r in audit["records"] if r["policy"]=="csgb" and r["phase"]==("smoke_train" if smoke else "train" if pre else "monitor") and (smoke or pre or r["timing_repeat"]==0)]
    eligible=[r for request in rows for r in request["rounds"] if r.get("raw_x") is not None]
    fig,ax=plt.subplots(figsize=(7,4.6))
    if eligible:
        scatter=ax.scatter([r["raw_x"][1] for r in eligible],[r["raw_x"][2] for r in eligible],
            c=[r["budget"] for r in eligible],cmap="viridis",vmin=8,vmax=32,s=18,alpha=.6)
        fig.colorbar(scatter,ax=ax,ticks=BUDGETS,label="Observed sampled proposal-node budget")
    else:ax.text(.5,.5,"No eligible C/D rounds recorded",transform=ax.transAxes,ha="center")
    ax.set(xlabel="Attention Coverage C (next Root cache)",ylabel="Cache-anchored normalized JS Shift D",
           xlim=(-.02,1.02),title=f"Observed CSGB actions, {len(eligible)} eligible rounds; NOT inferred decision boundary")
    if eligible:
        lo=min(r["raw_x"][2] for r in eligible);hi=max(r["raw_x"][2] for r in eligible)
        padding=max(.005,(hi-lo)*.12);ax.set_ylim(max(0.,lo-padding),min(1.,hi+padding))
        ax.text(.02,.98,"D zoomed to observations; theoretical normalized JS range [0,1]",transform=ax.transAxes,va="top",fontsize=7)
    else:ax.set_ylim(0.,1.)
    save(fig,"observed_budget_map")
    fig,ax=plt.subplots(figsize=(9,4.5));ys=[];es=[]
    for p in POLICIES:
        row=summary["absolute_metrics"][p]["decision_ms"];value=row["mean"];ci=row["ci95"]
        ys.append(float("nan") if value is None else value)
        es.append((0.,0.) if value is None or ci is None else (max(0.,value-ci[0]),max(0.,ci[1]-value)))
    ax.bar(range(7),ys,color="#53768c");ax.errorbar(range(7),ys,yerr=list(zip(*es)),fmt="none",ecolor="black",capsize=4)
    ax.set(xticks=range(7),xticklabels=POLICIES,ylabel="Feature/decision wall (ms/round)",
           title="Measured decision region includes implicit host/GPU waits; not pure arithmetic latency")
    ax.tick_params(axis="x",rotation=25);save(fig,"decision_latency")
    if not smoke and not pre:
        fig,axes=plt.subplots(2,1,figsize=(8,6),sharex=True)
        for mode in MODES:
            points=[r for r in audit["training"] if r["policy"]==mode]
            axes[0].plot(range(1,len(points)+1),[r["episode_return"] for r in points],label=mode,alpha=.8)
            axes[1].plot(range(1,len(points)+1),[r["advantage"] for r in points],label=mode,alpha=.8)
        axes[0].set(ylabel="Observed N − ρT",title="Fit0..15 learning trace: evolving policy, NOT held-out efficacy")
        axes[1].set(xlabel="Whole-request SGD step per policy (16docs ×2 fixed passes)",ylabel="Return − independent Native8 baseline")
        axes[0].legend();save(fig,"training_learning_curve")
        fig,axes=plt.subplots(1,2,figsize=(10,4.5))
        for ax,metric in zip(axes,("acceptance","tps")):
            for i,mode in enumerate(MODES):
                values=[v[metric]["sample_sd"] for v in summary["action_stream_variability"][mode].values()]
                ax.scatter([i]*len(values),values,alpha=.65)
            ax.set(xticks=range(3),xticklabels=MODES,ylabel="Within-doc SD across3 action streams",
                   title="Sampling variability: "+metric)
            ax.tick_params(axis="x",rotation=15)
        save(fig,"action_stream_variability")
    return figures

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path(__file__).with_name('plot_data.json'));p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    d=json.loads(a.data.read_text());a.out.mkdir(parents=True,exist_ok=False)
    render_figures({'summary':d['summary'],'training':d['training'],'records':d['observed_rounds'],'synthetic_qa':d.get('synthetic_qa',False)},a.out)
