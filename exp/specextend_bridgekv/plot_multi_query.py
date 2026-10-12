#!/usr/bin/env python3
"""Analyze and plot the EAGLE multi-query attention-retrieval ablation."""

import html, json, math, random, statistics
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"results"/"rtx4090_reproduction"/"multi_query"
POLICIES=["attention_q1","attention_q2","attention_q4"]
LABELS={"attention_q1":"Q=1 (original)","attention_q2":"Q=2","attention_q4":"Q=4"}
COLORS={"attention_q1":"#3569A8","attention_q2":"#E47C20","attention_q4":"#2E8B57"}
LENGTHS=["4K","8K","16K"]

def mean(xs):
    xs=list(xs); return statistics.fmean(xs) if xs else None
def load():
    rows=[]
    for p in sorted(DATA.glob("eagle_mq_*.json")): rows += json.loads(p.read_text())["raw_runs"]
    return rows
def bootstrap(xs,seed,n=8000):
    xs=list(map(float,xs)); rng=random.Random(seed)
    bs=sorted(mean(rng.choices(xs,k=len(xs))) for _ in range(n))
    return mean(xs),bs[int(.025*n)],bs[int(.975*n)]
def paired(a,b,fn,seed,n=12000):
    aa={r['sample_index']:fn(r) for r in a}; bb={r['sample_index']:fn(r) for r in b}; ids=sorted(aa.keys()&bb.keys())
    d=[aa[i]-bb[i] for i in ids]; return bootstrap(d,seed,n)
def jaccard(r): return mean(e['jaccard_with_previous'] for e in r['retrieval_events'] if e['jaccard_with_previous'] is not None)
def position(r): return mean(x for e in r['retrieval_events'] if e['kind']=='retrieval' for x in e['selected_chunk_centers_normalized'])
def metric(r,m):
    if m=='accept': return r['avg_accept_length']
    if m=='tps': return r['wall_tokens_per_sec']
    if m=='kl': return r['mean_root_target_draft_kl']
    if m=='jaccard': return jaccard(r)
    if m=='retrieval': return r['stage_time_ms'].get('retrieval',0)/r['total_generated']
    if m=='position': return position(r)
def esc(x): return html.escape(str(x))
def text(x,y,v,size=14,anchor='start',weight='400',fill='#222',rotate=None):
    tr=f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ''
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial,sans-serif" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}"{tr}>{esc(v)}</text>'
def line(x1,y1,x2,y2,c='#d9dde2',w=1): return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" stroke-width="{w}"/>'

def main():
    rows=load(); groups=defaultdict(list)
    for r in rows: groups[(r['length_label'],r['policy'])].append(r)
    metrics=[('accept','A  Draft quality','Acceptance length'),('tps','B  End-to-end performance','Throughput (token/s)'),('kl','C  Target–draft mismatch','Root KL(target || draft)'),('jaccard','D  Retrieval stability','Consecutive-set Jaccard'),('retrieval','E  Retrieval overhead','Retrieval time (ms/token)'),('position','F  Selected location','Mean normalized position')]
    W,H=1500,1515; out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',f'<rect width="{W}" height="{H}" fill="#fff"/>',text(44,40,'P1: multi-query attention retrieval',25,weight='600'),text(44,65,'EAGLE · GovReport · n=20 paired documents · mean with bootstrap 95% CI · same 64-chunk cache budget',14,fill='#555')]
    for i,p in enumerate(POLICIES):
        x=48+i*210; out += [f'<circle cx="{x}" cy="91" r="5" fill="{COLORS[p]}"/>',text(x+13,96,LABELS[p],14)]
    for mi,(m,title_,ylabel) in enumerate(metrics):
        col,row=mi%2,mi//2; x0,y0=34+col*730,115+row*460; w,h=700,440; left,right,top,bottom=x0+72,x0+w-20,y0+48,y0+h-54
        series={}; allv=[]
        for pi,p in enumerate(POLICIES):
            arr=[]
            for li,L in enumerate(LENGTHS):
                vals=[metric(r,m) for r in groups[(L,p)]]; vals=[v for v in vals if v is not None]
                arr.append(bootstrap(vals,1000+mi*100+pi*10+li)); allv += list(arr[-1][1:])
            series[p]=arr
        ymin=0 if m not in {'position','jaccard'} else min(allv)*.9; ymax=max(allv)*1.12
        if m in {'position','jaccard'}: ymin=max(0,ymin); ymax=min(1.02,max(ymax,ymin+.1))
        out += [text(x0+4,y0+24,title_,18,weight='600'),f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="#fff" stroke="#b9bec6"/>']
        sy=lambda v: bottom-(v-ymin)/(ymax-ymin)*(bottom-top)
        for j in range(5):
            v=ymin+(ymax-ymin)*j/4; yy=sy(v); fmt=f'{v:.3f}' if m=='retrieval' else f'{v:.2f}'
            out += [line(left,yy,right,yy),text(left-8,yy+5,fmt,12,'end',fill='#555')]
        for li,L in enumerate(LENGTHS): out.append(text(left+(right-left)*li/2,bottom+25,L,13,'middle'))
        out += [text((left+right)/2,y0+h-8,'Context length',13,'middle'),text(x0+17,(top+bottom)/2,ylabel,13,'middle',rotate=-90)]
        for p in POLICIES:
            pts=[]
            for li,(v,lo,hi) in enumerate(series[p]):
                xx=left+(right-left)*li/2; pts.append((xx,sy(v))); out += [line(xx,sy(lo),xx,sy(hi),COLORS[p],1.5),line(xx-4,sy(lo),xx+4,sy(lo),COLORS[p],1.5),line(xx-4,sy(hi),xx+4,sy(hi),COLORS[p],1.5)]
            out.append(f'<path d="'+" ".join(("M" if k==0 else "L")+f"{x:.1f},{y:.1f}" for k,(x,y) in enumerate(pts))+f'" fill="none" stroke="{COLORS[p]}" stroke-width="2.5"/>')
            out += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{COLORS[p]}" stroke="#fff"/>' for x,y in pts]
    out += [text(44,1495,'Q aggregates the last Q accepted-branch queries; Q=1 exactly reproduces the original attention retrieval.',13,fill='#555'),'</svg>']
    (DATA/'multi_query_ablation.svg').write_text('\n'.join(out)+'\n')

    report=['# P1: Multi-query attention retrieval','', '![Multi-query ablation](multi_query_ablation.png)','', 'All conditions use EAGLE, a 64-node candidate tree, 64 retrieved chunks, chunk size 32, retrieval every 8 decoding steps, greedy decoding, and the same 20 paired GovReport documents.','', '## Means','', '| Length | Queries | Acceptance | token/s | Root KL | Jaccard | Retrieval ms/token | Selected position |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for L in LENGTHS:
        for p in POLICIES:
            rs=groups[(L,p)]; report.append(f"| {L} | {p.rsplit('q',1)[1]} | {mean(r['avg_accept_length'] for r in rs):.3f} | {mean(r['wall_tokens_per_sec'] for r in rs):.3f} | {mean(r['mean_root_target_draft_kl'] for r in rs):.3f} | {mean(jaccard(r) for r in rs):.3f} | {mean(metric(r,'retrieval') for r in rs):.3f} | {mean(position(r) for r in rs):.3f} |")
    report += ['', '## Paired differences versus Q=1', '']
    for L in LENGTHS:
        base=groups[(L,'attention_q1')]; report += [f'### {L}','']
        for p in ('attention_q2','attention_q4'):
            rs=groups[(L,p)]; a=paired(rs,base,lambda r:r['avg_accept_length'],2000+len(report)); t=paired(rs,base,lambda r:r['wall_tokens_per_sec'],3000+len(report)); k=paired(rs,base,lambda r:r['mean_root_target_draft_kl'],4000+len(report))
            fci=lambda z:f"{z[0]:+.3f} [{z[1]:+.3f}, {z[2]:+.3f}]"
            report.append(f"- {LABELS[p]}: acceptance {fci(a)}; token/s {fci(t)}; root KL {fci(k)}")
        report.append('')
    report += ['Artifacts: [SVG source](multi_query_ablation.svg), [plotting code](../../../plot_multi_query.py), and raw results for [4K](eagle_mq_4k.json), [8K](eagle_mq_8k.json), and [16K](eagle_mq_16k.json).','']
    (DATA/'REPORT.md').write_text('\n'.join(report))
    print(f'wrote figure/report from {len(rows)} runs')

if __name__=='__main__': main()
