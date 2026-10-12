#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze P2a fixed attention-cache budget scans and create SVG/report artifacts."""

import html, json, math, random, statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results" / "rtx4090_reproduction" / "p2a_fixed_budget"
KS = [16, 24, 32, 48, 64]
LENGTHS = ["4K", "8K", "16K"]
COLORS = {"4K": "#3569A8", "8K": "#E47C20", "16K": "#2E8B57"}
KCOLORS = {16: "#6BAED6", 24: "#4292C6", 32: "#2171B5", 48: "#EF8A62", 64: "#B2182B"}

def mean(xs):
    xs=[float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return statistics.fmean(xs) if xs else None

def quantile(xs,q):
    xs=sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    pos=(len(xs)-1)*q; lo=int(pos); hi=min(lo+1,len(xs)-1); w=pos-lo
    return xs[lo]*(1-w)+xs[hi]*w

def bootstrap(xs,seed,n=5000):
    xs=[float(x) for x in xs if x is not None and math.isfinite(float(x))]
    rng=random.Random(seed); bs=sorted(mean(rng.choices(xs,k=len(xs))) for _ in range(n))
    return mean(xs),bs[int(.025*n)],bs[int(.975*n)]

def paired(a,b,fn,seed,n=7000):
    aa={r['sample_index']:fn(r) for r in a}; bb={r['sample_index']:fn(r) for r in b}; ids=sorted(aa.keys()&bb.keys())
    return bootstrap([aa[i]-bb[i] for i in ids],seed,n)

def load():
    rows=[]
    for p in sorted(DATA.glob('classic_p2a_*.json')): rows += json.loads(p.read_text())['raw_runs']
    return rows

def rmean(r,key): return mean(x.get(key) for x in r['rounds'])
def metric(r,m):
    if m=='accept': return r['avg_accept_length']
    if m=='tps': return r['wall_tokens_per_sec']
    if m=='kl': return r['mean_root_target_draft_kl']
    if m=='cache': return rmean(r,'working_cache_tokens')
    if m=='reject': return rmean(r,'rejected_position')
    if m in {'draft','verification','retrieval'}: return r['stage_time_ms'].get(m,0)/r['total_generated']
    raise KeyError(m)

def esc(x): return html.escape(str(x))
def text(x,y,v,size=14,anchor='start',weight='400',fill='#222',rotate=None):
    tr=f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ''
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial,sans-serif" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}"{tr}>{esc(v)}</text>'
def line(x1,y1,x2,y2,c='#d9dde2',w=1,dash=None):
    ds=f' stroke-dasharray="{dash}"' if dash else ''
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" stroke-width="{w}"{ds}/>'

def groups(rows):
    g=defaultdict(list)
    for r in rows: g[(r['length_label'],r['cache_top_k'])].append(r)
    return g

def panel(out,x0,y0,w,h,title,ylabel,series,yzero=False):
    left,right,top,bottom=x0+72,x0+w-20,y0+48,y0+h-54
    vals=[v for arr in series.values() for triple in arr for v in triple[1:] if v is not None]
    ymin=0 if yzero else min(vals); ymax=max(vals); pad=max((ymax-ymin)*.12,.001); ymax+=pad
    if not yzero: ymin-=pad
    sy=lambda v:bottom-(v-ymin)/(ymax-ymin)*(bottom-top); sx=lambda i:left+(right-left)*i/(len(KS)-1)
    out += [text(x0+4,y0+24,title,18,weight='600'),f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="#fff" stroke="#b9bec6"/>']
    for j in range(5):
        v=ymin+(ymax-ymin)*j/4; yy=sy(v); out += [line(left,yy,right,yy),text(left-8,yy+5,f'{v:.2f}',12,'end',fill='#555')]
    for i,k in enumerate(KS): out.append(text(sx(i),bottom+25,k,13,'middle'))
    out += [text((left+right)/2,y0+h-8,'Retrieved chunks (K)',13,'middle'),text(x0+17,(top+bottom)/2,ylabel,13,'middle',rotate=-90)]
    for length,arr in series.items():
        pts=[]
        for i,(v,lo,hi) in enumerate(arr):
            xx=sx(i); pts.append((xx,sy(v))); out += [line(xx,sy(lo),xx,sy(hi),COLORS[length],1.4),line(xx-4,sy(lo),xx+4,sy(lo),COLORS[length],1.4),line(xx-4,sy(hi),xx+4,sy(hi),COLORS[length],1.4)]
        path=' '.join(('M' if i==0 else 'L')+f'{x:.1f},{y:.1f}' for i,(x,y) in enumerate(pts))
        out.append(f'<path d="{path}" fill="none" stroke="{COLORS[length]}" stroke-width="2.5"/>')
        out.extend(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{COLORS[length]}" stroke="#fff"/>' for x,y in pts)

def draw_primary(g):
    W,H=1500,610; out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<rect width="100%" height="100%" fill="#fff"/>',text(42,40,'P2a: fixed cache-budget scan',25,weight='600'),text(42,66,'Classic · Last-1 attention · retrieval interval 4 · n=20 paired documents · bootstrap 95% CI',14,fill='#555')]
    for i,L in enumerate(LENGTHS): x=44+170*i; out += [f'<circle cx="{x}" cy="91" r="5" fill="{COLORS[L]}"/>',text(x+13,96,L,14)]
    for idx,(m,title_,ylabel) in enumerate([('accept','A  Draft quality','Acceptance length'),('tps','B  End-to-end performance','token/s')]):
        series={L:[bootstrap([metric(r,m) for r in g[(L,k)]],1000+idx*100+i*10+j) for j,k in enumerate(KS)] for i,L in enumerate(LENGTHS)}
        panel(out,25+idx*735,115,710,465,title_,ylabel,series)
    out += [text(42,596,'K counts retrieved chunks; chunk size is 32 tokens. Axes use the measured range rather than starting at zero.',13,fill='#555'),'</svg>']
    (DATA/'p2a_quality_performance.svg').write_text('\n'.join(out)+'\n')

def draw_diagnostics(g):
    specs=[('kl','A  Root KL','KL(target || draft)'),('cache','B  Working cache','Tokens'),('draft','C  Draft latency','ms/token'),('verification','D  Verification latency','ms/token'),('retrieval','E  Retrieval latency','ms/token'),('reject','F  Rejection position','Token position')]
    W,H=1500,1450; out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<rect width="100%" height="100%" fill="#fff"/>',text(42,40,'P2a mechanism diagnostics',25,weight='600'),text(42,66,'Per-run means · n=20 paired documents · bootstrap 95% CI',14,fill='#555')]
    for i,L in enumerate(LENGTHS): x=44+170*i; out += [f'<circle cx="{x}" cy="91" r="5" fill="{COLORS[L]}"/>',text(x+13,96,L,14)]
    for idx,(m,title_,ylabel) in enumerate(specs):
        col,row=idx%2,idx//2; series={L:[bootstrap([metric(r,m) for r in g[(L,k)]],2000+idx*100+i*10+j) for j,k in enumerate(KS)] for i,L in enumerate(LENGTHS)}
        panel(out,25+col*735,115+row*435,710,420,title_,ylabel,series)
    out += [text(42,1435,'Stage latency is normalized by generated tokens; working-cache size is averaged across verification rounds.',13,fill='#555'),'</svg>']
    (DATA/'p2a_diagnostics.svg').write_text('\n'.join(out)+'\n')

def best_counts(g,m):
    result={}
    for L in LENGTHS:
        by={(r['sample_index'],r['cache_top_k']):metric(r,m) for k in KS for r in g[(L,k)]}
        result[L]=Counter(max(KS,key=lambda k:by[(i,k)]) for i in range(20))
    return result

def stage_best(g):
    result={}
    for L in LENGTHS:
        score=defaultdict(list)
        for k in KS:
            for r in g[(L,k)]:
                rounds=r['rounds']; n=len(rounds)
                for i,item in enumerate(rounds): score[(k,min(4,int(i/max(n,1)*5)))].append(item['accept_length'])
        result[L]=[max(KS,key=lambda k:mean(score[(k,b)])) for b in range(5)]
    return result

def ema_values(r,key,alpha=.25):
    values=[]; current=None
    for item in r['rounds']:
        value=float(item[key])
        current=value if current is None else (1-alpha)*current+alpha*value
        values.append(current)
    return values

def draw_heterogeneity(g):
    W,H=1500,790; out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<rect width="100%" height="100%" fill="#fff"/>',text(42,40,'P2a: is one fixed K optimal everywhere?',25,weight='600'),text(42,66,'Sample-level oracle and descriptive generation-stage analysis',14,fill='#555')]
    for i,k in enumerate(KS): x=44+130*i; out += [f'<rect x="{x}" y="82" width="12" height="12" fill="{KCOLORS[k]}"/>',text(x+18,93,f'K={k}',13)]
    for pi,(m,title_) in enumerate([('accept','A  Best K per sample: acceptance'),('tps','B  Best K per sample: token/s')]):
        x0,y0=32+pi*735,115; left,right,top,bottom=x0+150,x0+700,y0+58,y0+340; counts=best_counts(g,m)
        out += [text(x0+4,y0+25,title_,18,weight='600')]
        for li,L in enumerate(LENGTHS):
            y=top+li*78; out.append(text(left-15,y+25,L,14,'end',weight='600')); x=left
            for k in KS:
                w=(right-left)*counts[L][k]/20
                out += [f'<rect x="{x}" y="{y}" width="{w}" height="36" fill="{KCOLORS[k]}"/>']
                if counts[L][k]: out.append(text(x+w/2,y+24,counts[L][k],12,'middle',fill='#fff',weight='600'))
                x+=w
        out += [text((left+right)/2,bottom+18,'Documents (20 total)',13,'middle')]
    sb=stage_best(g); x0,y0=32,495; left=x0+230; cellw=205
    out += [text(x0+4,y0+25,'C  K with highest mean acceptance by normalized generation stage',18,weight='600')]
    for b,label in enumerate(['0–20%','20–40%','40–60%','60–80%','80–100%']): out.append(text(left+b*cellw+cellw/2,y0+55,label,13,'middle',weight='600'))
    for li,L in enumerate(LENGTHS):
        y=y0+70+li*55; out.append(text(left-18,y+29,L,14,'end',weight='600'))
        for b,k in enumerate(sb[L]):
            x=left+b*cellw; out += [f'<rect x="{x}" y="{y}" width="{cellw-3}" height="42" fill="{KCOLORS[k]}"/>',text(x+(cellw-3)/2,y+27,f'K={k}',13,'middle',fill='#fff',weight='600')]
    out += [text(42,775,'Stage comparison is descriptive: fixed-K generations can diverge after early token decisions.',13,fill='#555'),'</svg>']
    (DATA/'p2a_heterogeneity.svg').write_text('\n'.join(out)+'\n')
    return {m:best_counts(g,m) for m in ('accept','tps')},sb

def validate(rows,g):
    expected={(L,k) for L in LENGTHS for k in KS}
    if set(g)!=expected: raise SystemExit(f'groups mismatch: missing={expected-set(g)}, extra={set(g)-expected}')
    bad={key:len(g[key]) for key in expected if len(g[key])!=20}
    if bad: raise SystemExit(f'expected 20 rows per group: {bad}')
    keys=[(r['length_label'],r['cache_top_k'],r['sample_index']) for r in rows]
    if len(keys)!=len(set(keys)): raise SystemExit('duplicate rows')

def write_report(rows,g,best,stages):
    report=['# P2a：固定缓存预算扫描','', '![质量与性能](p2a_quality_performance.png)','', '![机制诊断](p2a_diagnostics.png)','', '![异质性](p2a_heterogeneity.png)','', '## 实验设置','', 'Classic draft；原始 Last-1 attention；检索间隔 4；K=16/24/32/48/64；4K/8K/16K；每组 20 个配对样本；chunk size 32；候选树 64 nodes；greedy；seed 42；样本内随机化 K 的执行顺序。','', '## 均值','', '| Length | K | Acceptance | token/s | Root KL | Working-cache tokens | Draft ms/token | Verify ms/token | Retrieval ms/token |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for L in LENGTHS:
        for k in KS:
            rs=g[(L,k)]; report.append(f"| {L} | {k} | {mean(metric(r,'accept') for r in rs):.3f} | {mean(metric(r,'tps') for r in rs):.3f} | {mean(metric(r,'kl') for r in rs):.3f} | {mean(metric(r,'cache') for r in rs):.1f} | {mean(metric(r,'draft') for r in rs):.3f} | {mean(metric(r,'verification') for r in rs):.3f} | {mean(metric(r,'retrieval') for r in rs):.4f} |")
    report += ['', '## 相对 K=32 的配对差值（95% bootstrap CI）','']
    for li,L in enumerate(LENGTHS):
        report += [f'### {L}','']; base=g[(L,32)]
        for j,k in enumerate([16,24,48,64]):
            a=paired(g[(L,k)],base,lambda r:metric(r,'accept'),4000+li*100+j); t=paired(g[(L,k)],base,lambda r:metric(r,'tps'),5000+li*100+j); kl=paired(g[(L,k)],base,lambda r:metric(r,'kl'),6000+li*100+j); fmt=lambda z:f'{z[0]:+.3f} [{z[1]:+.3f}, {z[2]:+.3f}]'
            report.append(f'- K={k}: acceptance {fmt(a)}; token/s {fmt(t)}; Root KL {fmt(kl)}')
        report.append('')
    report += ['## P2b 阈值候选（来自 P2a K=32 的 EMA 分布）','', '下表先对每个样本按 alpha=0.25 计算 EMA，再对 EMA 状态取分位数；控制器比较的是 EMA，因此不能直接使用原始整数接受长度的分位数。','', '| Length | EMA accept p30 | EMA accept p70 | EMA KL p30 | EMA KL p70 |','|---|---:|---:|---:|---:|']
    for L in LENGTHS:
        acc=[x for r in g[(L,32)] for x in ema_values(r,'accept_length')]; kl=[x for r in g[(L,32)] for x in ema_values(r,'root_target_draft_kl')]
        report.append(f'| {L} | {quantile(acc,.3):.3f} | {quantile(acc,.7):.3f} | {quantile(kl,.3):.3f} | {quantile(kl,.7):.3f} |')
    report += ['', '## 结论','', '**P2 的动机成立，但不是“困难位置一律需要更大 K”。** 4K 的质量与吞吐最优点都是 K=32；8K 在 K=32–48 形成平台；16K 则由 K=16 获得最高平均接受长度、K=24 获得最高平均吞吐。固定 K=64 在 4K/8K 显著降低接受长度和吞吐，说明过大的工作缓存会引入干扰和额外开销。','', '最优 K 也存在明显样本异质性：4K 大多偏向 K=32，8K 分散在 K=32/48，16K 则主要偏向 K=16/24。按生成阶段分箱后，描述性最优 K 也发生变化。因而动态预算值得进入 pilot，但控制器必须通过端到端 token/s 和 budget-matched fixed-K 验证，不能预设 Hard=64 一定改善质量。','', 'P2b pilot 仍按预注册规则测试 K=16/32/64，以避免看完结果后改规则；同时记录每个状态的占比。如果 Hard=64 频繁触发但性能下降，下一轮应把控制动作改为 16/24/32/48，而不是继续调阈值掩盖问题。','', '## 产物','', '- [质量与性能 SVG](p2a_quality_performance.svg)','- [机制诊断 SVG](p2a_diagnostics.svg)','- [异质性 SVG](p2a_heterogeneity.svg)','- [分析代码](../../../plot_p2a_fixed_budget.py)','- [4K 原始结果](classic_p2a_4k.json)','- [8K 原始结果](classic_p2a_8k.json)','- [16K 原始结果](classic_p2a_16k.json)','']
    (DATA/'REPORT.md').write_text('\n'.join(report))

def main():
    DATA.mkdir(parents=True,exist_ok=True); rows=load(); g=groups(rows); validate(rows,g)
    draw_primary(g); draw_diagnostics(g); best,stages=draw_heterogeneity(g); write_report(rows,g,best,stages)
    print(f'wrote P2a figures/report from {len(rows)} runs')

if __name__=='__main__': main()
