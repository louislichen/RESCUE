#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze and visualize the P2b dynamic Top-K pilot."""
import html,json,math,random,statistics
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'results'/'rtx4090_reproduction'/'p2b_dynamic_pilot'
LENGTHS=['8K','16K']
POLICIES=['attention_k16','attention_k32','attention_k64','dynamic_accept','dynamic_kl','dynamic_accept_kl']
LABELS={'attention_k16':'Fixed-16','attention_k32':'Fixed-32','attention_k64':'Fixed-64','dynamic_accept':'Dynamic-Accept','dynamic_kl':'Dynamic-KL','dynamic_accept_kl':'Dynamic-Accept+KL'}
COLORS={'8K':'#E47C20','16K':'#2E8B57'}
KCOLORS={16:'#6BAED6',32:'#2171B5',64:'#B2182B'}

def mean(xs):
    xs=[float(x) for x in xs if x is not None and math.isfinite(float(x))]; return statistics.fmean(xs) if xs else None
def bootstrap(xs,seed,n=10000):
    xs=[float(x) for x in xs]; rng=random.Random(seed); bs=sorted(mean(rng.choices(xs,k=len(xs))) for _ in range(n)); return mean(xs),bs[int(.025*n)],bs[int(.975*n)]
def paired(a,b,fn,seed):
    aa={r['sample_index']:fn(r) for r in a}; bb={r['sample_index']:fn(r) for r in b}; ids=sorted(aa.keys()&bb.keys()); return bootstrap([aa[i]-bb[i] for i in ids],seed)
def load():
    rows=[]
    for p in sorted(DATA.glob('classic_p2b_*.json')): rows+=json.loads(p.read_text())['raw_runs']
    return rows
def groups(rows):
    g=defaultdict(list)
    for r in rows:g[(r['length_label'],r['policy'])].append(r)
    return g
def avgk(r): return mean(x['requested_cache_top_k'] for x in r['rounds'])
def cache(r): return mean(x['working_cache_tokens'] for x in r['rounds'])
def metric(r,m):
    if m=='accept':return r['avg_accept_length']
    if m=='tps':return r['wall_tokens_per_sec']
    if m=='kl':return r['mean_root_target_draft_kl']
    if m=='avgk':return avgk(r)
    if m=='cache':return cache(r)
    if m in {'draft','verification','retrieval'}:return r['stage_time_ms'].get(m,0)/r['total_generated']
def esc(x):return html.escape(str(x))
def text(x,y,v,size=14,anchor='start',weight='400',fill='#222',rotate=None):
    tr=f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ''
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial,sans-serif" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}"{tr}>{esc(v)}</text>'
def line(x1,y1,x2,y2,c='#d9dde2',w=1):return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" stroke-width="{w}"/>'

def bar_panel(out,x0,y0,w,h,title,ylabel,g,m,seed):
    left,right,top,bottom=x0+75,x0+w-18,y0+50,y0+h-75
    values={}; bounds=[]
    for li,L in enumerate(LENGTHS):
        for pi,p in enumerate(POLICIES):
            z=bootstrap([metric(r,m) for r in g[(L,p)]],seed+li*100+pi);values[(L,p)]=z;bounds+=list(z[1:])
    ymin=min(bounds);ymax=max(bounds);pad=max((ymax-ymin)*.12,.01);ymin=max(0,ymin-pad);ymax+=pad
    sy=lambda v:bottom-(v-ymin)/(ymax-ymin)*(bottom-top)
    out += [text(x0+4,y0+25,title,18,weight='600'),f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="#fff" stroke="#b9bec6"/>']
    for j in range(5):
        v=ymin+(ymax-ymin)*j/4;yy=sy(v);out += [line(left,yy,right,yy),text(left-8,yy+4,f'{v:.2f}',11,'end',fill='#555')]
    groupw=(right-left)/len(POLICIES);bw=groupw*.32
    for pi,p in enumerate(POLICIES):
        center=left+(pi+.5)*groupw
        for li,L in enumerate(LENGTHS):
            v,lo,hi=values[(L,p)];x=center+(li-.5)*bw
            out += [f'<rect x="{x-bw*.45}" y="{sy(v)}" width="{bw*.9}" height="{bottom-sy(v)}" fill="{COLORS[L]}" opacity=".88"/>',line(x,sy(lo),x,sy(hi),COLORS[L],1.5),line(x-3,sy(lo),x+3,sy(lo),COLORS[L],1.5),line(x-3,sy(hi),x+3,sy(hi),COLORS[L],1.5)]
        out.append(text(center,bottom+18,LABELS[p].replace('Dynamic-','Dyn-'),10,'middle',rotate=-20))
    out.append(text(x0+18,(top+bottom)/2,ylabel,12,'middle',rotate=-90))

def draw_summary(g):
    specs=[('accept','A  Acceptance length','Tokens'),('tps','B  End-to-end throughput','token/s'),('avgk','C  Average requested cache budget','Chunks'),('kl','D  Root KL','KL(target || draft)')]
    W,H=1500,1040;out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<rect width="100%" height="100%" fill="#fff"/>',text(42,40,'P2b pilot: dynamic Top-K controllers',25,weight='600'),text(42,66,'Classic · interval 4 · n=5 paired documents · bootstrap 95% CI (pilot only)',14,fill='#555')]
    for i,L in enumerate(LENGTHS):x=44+i*170;out += [f'<circle cx="{x}" cy="91" r="5" fill="{COLORS[L]}"/>',text(x+13,96,L,14)]
    for i,(m,t,y) in enumerate(specs):bar_panel(out,25+(i%2)*735,115+(i//2)*445,710,420,t,y,g,m,1000+i*200)
    out += [text(42,1024,'Intervals are wide because this is a five-document pilot; final decisions require n=20.',13,fill='#555'),'</svg>'];(DATA/'p2b_pilot_summary.svg').write_text('\n'.join(out)+'\n')

def state_proportions(rs):
    c=Counter(x['requested_cache_top_k'] for r in rs for x in r['rounds']);n=sum(c.values());return {k:c[k]/n for k in (16,32,64)}
def draw_states(g):
    dyn=POLICIES[3:];W,H=1500,900;out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<rect width="100%" height="100%" fill="#fff"/>',text(42,40,'P2b controller behavior',25,weight='600'),text(42,66,'State occupancy across all rounds and representative sample-0 trajectories',14,fill='#555')]
    for i,k in enumerate((16,32,64)):x=44+i*140;out += [f'<rect x="{x}" y="82" width="12" height="12" fill="{KCOLORS[k]}"/>',text(x+18,93,f'K={k}',13)]
    left,right,top=230,1430,145;barw=(right-left)/len(dyn)
    for li,L in enumerate(LENGTHS):
        y=top+li*135;out += [text(42,y+30,f'A{li+1}  {L} state occupancy',17,weight='600')]
        for pi,p in enumerate(dyn):
            x=left+pi*barw;pos=x;props=state_proportions(g[(L,p)])
            for k in (16,32,64):
                w=(barw-45)*props[k];out.append(f'<rect x="{pos}" y="{y}" width="{w}" height="42" fill="{KCOLORS[k]}"/>')
                if props[k]>.07:out.append(text(pos+w/2,y+27,f'{props[k]*100:.0f}%',12,'middle',fill='#fff',weight='600'))
                pos+=w
            out.append(text(x+(barw-45)/2,y+66,LABELS[p],12,'middle'))
    for li,L in enumerate(LENGTHS):
        x0,y0=35+li*735,455;leftp,rightp,top2,bottom=x0+90,x0+700,y0+55,y0+350
        out += [text(x0+4,y0+25,f'B{li+1}  {L} sample-0 state trajectories',17,weight='600'),f'<rect x="{leftp}" y="{top2}" width="{rightp-leftp}" height="{bottom-top2}" fill="#fff" stroke="#bbb"/>']
        for j,k in enumerate((16,32,64)):
            yy=bottom-j*(bottom-top2)/2;out += [line(leftp,yy,rightp,yy),text(leftp-8,yy+4,k,11,'end')]
        for pi,p in enumerate(dyn):
            r=next(x for x in g[(L,p)] if x['sample_index']==0);vals=[x['requested_cache_top_k'] for x in r['rounds']];color=['#7A5195','#EF5675','#FFA600'][pi];pts=[]
            for i,v in enumerate(vals):pts.append((leftp+(rightp-leftp)*i/max(1,len(vals)-1),bottom-(v-16)/48*(bottom-top2)))
            path=' '.join(('M' if i==0 else 'L')+f'{x:.1f},{y:.1f}' for i,(x,y) in enumerate(pts));out.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" opacity=".85"/>')
            out += [line(leftp+pi*185,bottom+28,leftp+pi*185+25,bottom+28,color,3),text(leftp+pi*185+32,bottom+33,LABELS[p].replace('Dynamic-',''),11)]
        out.append(text((leftp+rightp)/2,bottom+62,'Verification round',12,'middle'))
    out += [text(42,884,'Requested K is the controller decision for the next draft; actual cache size updates on retrieval rounds.',13,fill='#555'),'</svg>'];(DATA/'p2b_state_behavior.svg').write_text('\n'.join(out)+'\n')

def draw_latency(g):
    W,H=1500,610;out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<rect width="100%" height="100%" fill="#fff"/>',text(42,40,'P2b pilot: stage latency',25,weight='600'),text(42,66,'CUDA stage time normalized by generated tokens',14,fill='#555')]
    for i,(m,t) in enumerate([('draft','A  Draft'),('verification','B  Verification'),('retrieval','C  Retrieval')]):bar_panel(out,18+i*492,105,470,460,t,'ms/token',g,m,3000+i*200)
    out += [text(42,596,'Retrieval is a subset of draft time. Policy execution order is randomized within each sample.',13,fill='#555'),'</svg>'];(DATA/'p2b_stage_latency.svg').write_text('\n'.join(out)+'\n')

def validate(rows,g):
    ex={(L,p) for L in LENGTHS for p in POLICIES};bad={x:len(g[x]) for x in ex if len(g[x])!=5}
    if set(g)!=ex or bad:raise SystemExit(f'groups mismatch extra={set(g)-ex} missing={ex-set(g)} bad={bad}')
    keys=[(r['length_label'],r['policy'],r['sample_index']) for r in rows]
    if len(keys)!=len(set(keys)):raise SystemExit('duplicates')

def write_report(g):
    rep=['# P2b：动态 Top-K pilot','', '![汇总](p2b_pilot_summary.png)','', '![控制器状态](p2b_state_behavior.png)','', '![阶段延迟](p2b_stage_latency.png)','', '## 设置','', '8K/16K；各 5 个配对样本；Fixed-16/32/64、Dynamic-Accept、Dynamic-KL、Dynamic-Accept+KL；Last-1 attention；检索间隔 4；EMA alpha=0.25；连续两轮满足才切换。阈值来自 P2a K=32 的 EMA p30/p70。','', '## 均值','', '| Length | Policy | Acceptance | token/s | Avg K | Cache tokens | Root KL | K=16/32/64 share |','|---|---|---:|---:|---:|---:|---:|---|']
    for L in LENGTHS:
        for p in POLICIES:
            rs=g[(L,p)];prop=state_proportions(rs);rep.append(f"| {L} | {LABELS[p]} | {mean(metric(r,'accept') for r in rs):.3f} | {mean(metric(r,'tps') for r in rs):.3f} | {mean(avgk(r) for r in rs):.1f} | {mean(cache(r) for r in rs):.1f} | {mean(metric(r,'kl') for r in rs):.3f} | {prop[16]:.0%}/{prop[32]:.0%}/{prop[64]:.0%} |")
    rep += ['', '## 相对 Fixed-32 的配对差值（95% bootstrap CI；n=5）','']
    for li,L in enumerate(LENGTHS):
        rep += [f'### {L}',''];base=g[(L,'attention_k32')]
        for j,p in enumerate(POLICIES):
            if p=='attention_k32':continue
            a=paired(g[(L,p)],base,lambda r:metric(r,'accept'),5000+li*100+j);t=paired(g[(L,p)],base,lambda r:metric(r,'tps'),6000+li*100+j);fmt=lambda z:f'{z[0]:+.3f} [{z[1]:+.3f}, {z[2]:+.3f}]';rep.append(f'- {LABELS[p]}: acceptance {fmt(a)}; token/s {fmt(t)}')
        rep.append('')
    rep += ['## 结论','', '**原始 P2b 控制规则没有通过 pilot gate，因此不应直接扩展到 20 样本，也不应进入 P2c。** 三个动态控制器都没有在 8K 与 16K 上同时取得正吞吐差，更没有配对 95% CI 全部高于 0。','', 'Dynamic-Accept 在 8K 相对 Fixed-32 的吞吐均值为 +0.873 token/s，但 16K 为 -1.186 token/s；Dynamic-KL 两个长度分别为 -0.513 和 -0.407 token/s。Dynamic-Accept+KL 平均使用 41.4/42.4 个 chunk，却只得到 -0.455/+0.075 token/s，并且接受长度没有改善，不能证明动态分配优于更多缓存。','', '失败机制与 P2a 一致：联合控制器在 8K/16K 分别有 37%/42% 的轮次处于 K=64，而 Fixed-64 本身已经是较差状态。P2a 的正确结论是“最佳预算随上下文变化”，不是“低接受或高 KL 时必然增大到 64”。当前信号也许能识别困难状态，但动作映射是错的。','', '建议把下一版动作空间改成 context-aware：8K 使用 24/32/48，16K 使用 16/24/32；4K 暂时维持固定 32。该修改属于新的 P2b-v2 假设，需要单独预注册和 pilot，不能与本轮失败结果合并汇报。','', '## 产物','', '- [汇总 SVG](p2b_pilot_summary.svg)','- [状态行为 SVG](p2b_state_behavior.svg)','- [阶段延迟 SVG](p2b_stage_latency.svg)','- [分析代码](../../../plot_p2b_dynamic_pilot.py)','- [8K 原始结果](classic_p2b_8k.json)','- [16K 原始结果](classic_p2b_16k.json)',''];(DATA/'REPORT.md').write_text('\n'.join(rep))

def main():
    DATA.mkdir(parents=True,exist_ok=True);rows=load();g=groups(rows);validate(rows,g);draw_summary(g);draw_states(g);draw_latency(g);write_report(g);print(f'wrote P2b pilot from {len(rows)} runs')
if __name__=='__main__':main()
