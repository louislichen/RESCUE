"""Standalone figures; run in an environment with NumPy and Matplotlib."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='exp/attention_horizon/out');a=p.parse_args()
    out=Path(a.output);s=json.loads((out/'summary.json').read_text());runtime=s['runtime']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(16,4.4))
    rules=['teacher','union','read_teacher','read_joint025','read_joint1']
    colors=['#64748b','#0d9488','#818cf8','#2563eb','#c026d3']
    for rule,color in zip(rules,colors):
        r=s['offline'][rule]
        axes[0].scatter(r['teacher_read_error'],r['student_read_error'],c=color,label=rule,s=65)
    axes[0].set(xlabel='Teacher normalized read error',ylabel='Student normalized read error',title='A. Native-space read preservation')
    axes[0].legend(fontsize=8)
    methods=['B_best_static','B_structural','B_confidence','B_attention','B_attention_read','B_combined','B_attention_online']
    labels=['Static','Structure','Confidence','Attention','Attn+read','Combined','Attn online']
    axes[1].barh(labels,[runtime[k]['decode_tps'] for k in methods],color=['#64748b','#a1a1aa','#0d9488','#2563eb','#818cf8','#c026d3','#0284c7'])
    axes[1].invert_yaxis();axes[1].set(xlabel='Actual decode tokens/s',title='B. Horizon policy (equal instrumentation)')
    if 'O_teacher_static' in runtime:
        axes[1].axvline(runtime['O_teacher_static']['decode_tps'],ls='--',color='#222',label='Static without student probes')
        axes[1].legend(fontsize=7,loc='lower left')
    methods=['C_jointworkset_static','C_sequential','C_joint','C_sequential_V','C_joint_V']
    labels=['Selected R + static k','Sequential','Joint','Sequential + V','Joint + V']
    stages=['selection','decision','source_pack','draft','verify','cache_update']
    names=['Workset','Horizon','V source','Draft','Verify','Cache']
    colors=['#93c5fd','#c4b5fd','#f9a8d4','#6ee7b7','#fbbf24','#9ca3af']
    left=np.zeros(len(methods))
    for stage,name,color in zip(stages,names,colors):
        values=np.array([runtime[m]['stage_seconds'][stage]/runtime[m]['tokens']*1000 for m in methods])
        axes[2].barh(labels,values,left=left,color=color,label=name);left+=values
    axes[2].invert_yaxis();axes[2].set(xlabel='Measured stage milliseconds / output token',title='C. Coupling benefit and planning cost')
    axes[2].legend(fontsize=7,ncol=3,loc='lower right')
    fig.tight_layout();fig.savefig(out/'mechanisms.png',dpi=180);fig.savefig(out/'mechanisms.pdf')
    plt.close(fig)


if __name__=='__main__':main()
