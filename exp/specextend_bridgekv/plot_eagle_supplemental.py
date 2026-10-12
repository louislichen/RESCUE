#!/usr/bin/env python3
"""Render supplemental EAGLE diagnostics without third-party dependencies."""

from __future__ import annotations

import html
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results" / "rtx4090_reproduction" / "reliable_baseline"
POLICIES = ["full", "recent", "random", "attention"]
RETRIEVAL_POLICIES = ["recent", "random", "attention"]
LABELS = {"full": "Full KV", "recent": "Recent-only", "random": "Random chunks", "attention": "Attention Top-K"}
COLORS = {"full": "#3569A8", "recent": "#E47C20", "random": "#777777", "attention": "#2E8B57"}
LENGTHS = ["1K", "2K", "4K", "8K", "16K"]


def load_rows():
    rows = []
    for path in sorted(DATA.glob("eagle_*.json")):
        rows.extend(json.loads(path.read_text())["raw_runs"])
    return rows


def esc(x):
    return html.escape(str(x))


def text(x, y, value, size=14, anchor="start", weight="400", fill="#222", rotate=None):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial,sans-serif" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}"{transform}>{esc(value)}</text>'


def line(x1, y1, x2, y2, stroke="#d9dde2", width=1):
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"/>'


def bootstrap(values, seed, n=4000):
    values = list(map(float, values))
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n))
    return statistics.fmean(values), means[int(.025*n)], means[int(.975*n)]


def quantile(values, q):
    values = sorted(values)
    if not values:
        return 0
    pos = (len(values)-1)*q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo]*(hi-pos)+values[hi]*(pos-lo)


def svg_start(width, height, title, subtitle):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>',
        text(44, 40, title, 25, weight="600"), text(44, 65, subtitle, 14, fill="#555"),
    ]


def stage_figure(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["length_label"], r["policy"])].append(r)
    metrics = [("draft", "A  Draft construction", "Draft time (ms/token)"),
               ("verification", "B  Target verification", "Verification time (ms/token)"),
               ("retrieval", "C  Cache retrieval", "Retrieval time (ms/token)")]
    width, height = 1500, 610
    out = svg_start(width, height, "EAGLE stage-time decomposition",
                    "GovReport · n=20 paired documents · mean with bootstrap 95% CI · retrieval time is a subset of draft time")
    for i, policy in enumerate(POLICIES):
        x = 48 + i*190
        out += [f'<circle cx="{x}" cy="91" r="5" fill="{COLORS[policy]}"/>', text(x+12, 96, LABELS[policy], 14)]
    for mi, (metric, title_, ylabel) in enumerate(metrics):
        x0, y0, w, h = 34+mi*490, 116, 460, 450
        left, right, top, bottom = x0+70, x0+w-18, y0+48, y0+h-52
        series = {}
        all_ci = []
        for pi, p in enumerate(POLICIES):
            vals = []
            for li, L in enumerate(LENGTHS):
                xs = [r.get("stage_time_ms", {}).get(metric, 0)/r["total_generated"] for r in grouped[(L,p)]]
                vals.append(bootstrap(xs, 1000+mi*100+pi*10+li))
                all_ci.extend(vals[-1][1:])
            series[p] = vals
        ymax = max(all_ci)*1.12 or 1
        out += [text(x0+4, y0+24, title_, 18, weight="600"),
                f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="#fff" stroke="#b9bec6"/>']
        for j in range(5):
            v = ymax*j/4; yy = bottom-(bottom-top)*j/4
            label = f"{v:.3f}" if metric == "retrieval" else f"{v:.1f}"
            out += [line(left,yy,right,yy), text(left-8,yy+5,label,12,"end",fill="#555")]
        for li,L in enumerate(LENGTHS):
            xx=left+(right-left)*li/4; out += [text(xx,bottom+24,L,12,"middle",fill="#444")]
        out += [text((left+right)/2,y0+h-8,"Context length",13,"middle"), text(x0+17,(top+bottom)/2,ylabel,13,"middle",rotate=-90)]
        for p in POLICIES:
            pts=[]
            for li,(v,lo,hi) in enumerate(series[p]):
                xx=left+(right-left)*li/4; sy=lambda z: bottom-(z/ymax)*(bottom-top)
                pts.append((xx,sy(v))); out += [line(xx,sy(lo),xx,sy(hi),COLORS[p],1.5),line(xx-4,sy(lo),xx+4,sy(lo),COLORS[p],1.5),line(xx-4,sy(hi),xx+4,sy(hi),COLORS[p],1.5)]
            out.append(f'<path d="'+" ".join(("M" if k==0 else "L")+f"{x:.1f},{y:.1f}" for k,(x,y) in enumerate(pts))+f'" fill="none" stroke="{COLORS[p]}" stroke-width="2.5"/>')
            out += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{COLORS[p]}" stroke="#fff"/>' for x,y in pts]
    out.append('</svg>')
    (DATA/"eagle_stage_times.svg").write_text("\n".join(out)+"\n")


def box_figure(rows):
    values = defaultdict(list)
    for r in rows:
        for q in r["rounds"]:
            values[(r["length_label"],r["policy"],"accept")].append(q["accept_length"])
            values[(r["length_label"],r["policy"],"reject")].append(q["rejected_position"])
    width,height=1500,700
    out=svg_start(width,height,"EAGLE per-round draft outcomes",
                  "All decoding rounds pooled within each condition · boxes show median and IQR · whiskers show 5th–95th percentiles")
    for i,p in enumerate(POLICIES):
        x=48+i*190; out += [f'<rect x="{x-5}" y="85" width="10" height="10" fill="{COLORS[p]}"/>',text(x+12,96,LABELS[p],14)]
    for mi,(metric,title_,ylabel) in enumerate([("accept","A  Accepted draft tokens per round","Accepted tokens"),("reject","B  First rejected position","1-indexed position")]):
        x0,y0,w,h=34+mi*730,120,700,520; left,right,top,bottom=x0+70,x0+w-20,y0+48,y0+h-54
        ymax=max(max(v) for k,v in values.items() if k[2]==metric)+1
        out += [text(x0+4,y0+24,title_,18,weight="600"),f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="#fff" stroke="#b9bec6"/>']
        for j in range(int(ymax)+1):
            yy=bottom-(bottom-top)*j/ymax
            if j%2==0: out += [line(left,yy,right,yy),text(left-8,yy+5,j,12,"end",fill="#555")]
        group_w=(right-left)/5
        for li,L in enumerate(LENGTHS):
            cx=left+group_w*(li+.5); out.append(text(cx,bottom+26,L,13,"middle"))
            for pi,p in enumerate(POLICIES):
                xs=values[(L,p,metric)]; q05,q25,med,q75,q95=[quantile(xs,q) for q in (.05,.25,.5,.75,.95)]
                xx=cx+(pi-1.5)*group_w*.18; sy=lambda z: bottom-(bottom-top)*z/ymax; bw=group_w*.13
                out += [line(xx,sy(q05),xx,sy(q95),COLORS[p],1.5),f'<rect x="{xx-bw/2:.1f}" y="{sy(q75):.1f}" width="{bw:.1f}" height="{sy(q25)-sy(q75):.1f}" fill="{COLORS[p]}" fill-opacity=".25" stroke="{COLORS[p]}"/>',line(xx-bw/2,sy(med),xx+bw/2,sy(med),COLORS[p],2)]
        out += [text((left+right)/2,y0+h-9,"Context length",13,"middle"),text(x0+17,(top+bottom)/2,ylabel,13,"middle",rotate=-90)]
    out += [text(44,680,"Rejected position is deterministically accept_length + 1; the second panel makes that measurement definition explicit.",13,fill="#555"),'</svg>']
    (DATA/"eagle_round_outcomes.svg").write_text("\n".join(out)+"\n")


def heat_color(t):
    t=max(0,min(1,t)); a=(245,249,246); b=(24,120,75)
    return '#'+''.join(f'{round(a[i]+(b[i]-a[i])*t):02x}' for i in range(3))


def position_figure(rows):
    bins=20; hist=defaultdict(lambda:[0]*bins)
    for r in rows:
        if r["policy"] not in RETRIEVAL_POLICIES: continue
        for event in r["retrieval_events"]:
            for x in event["selected_chunk_centers_normalized"]:
                hist[(r["policy"],r["length_label"])][min(bins-1,int(x*bins))]+=1
    width,height=1500,560
    out=svg_start(width,height,"Where the working cache comes from",
                  "Distribution of selected chunk centers over all retrieval events · position 0 = context start, 1 = context end · row-normalized")
    for pi,p in enumerate(RETRIEVAL_POLICIES):
        x0,y0,w,h=34+pi*490,105,460,400; left,right,top,bottom=x0+72,x0+w-20,y0+50,y0+h-52
        out += [text(x0+4,y0+25,f"{chr(65+pi)}  {LABELS[p]}",18,weight="600"),f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="#fff" stroke="#b9bec6"/>']
        cw=(right-left)/bins; ch=(bottom-top)/5
        for li,L in enumerate(LENGTHS):
            vals=hist[(p,L)]; total=sum(vals) or 1; probs=[v/total for v in vals]; peak=max(probs) or 1
            yy=top+li*ch; out.append(text(left-10,yy+ch*.62,L,13,"end"))
            for bi,v in enumerate(probs):
                out.append(f'<rect x="{left+bi*cw:.1f}" y="{yy:.1f}" width="{cw+.2:.1f}" height="{ch+.2:.1f}" fill="{heat_color(v/peak)}"/>')
        for tick,label in [(0,"0"),(.25,".25"),(.5,".50"),(.75,".75"),(1,"1")]: out.append(text(left+(right-left)*tick,bottom+24,label,12,"middle"))
        out += [text((left+right)/2,y0+h-8,"Normalized context position",13,"middle"),text(x0+17,(top+bottom)/2,"Context length",13,"middle",rotate=-90)]
    out += [f'<rect x="1160" y="525" width="240" height="12" fill="url(#g)"/>','<defs><linearGradient id="g"><stop offset="0" stop-color="#f5f9f6"/><stop offset="1" stop-color="#18784b"/></linearGradient></defs>',text(1140,537,"low",12,"end",fill="#555"),text(1410,537,"high",12,fill="#555"),'</svg>']
    (DATA/"eagle_chunk_positions.svg").write_text("\n".join(out)+"\n")


if __name__ == "__main__":
    data=load_rows(); stage_figure(data); box_figure(data); position_figure(data)
    print("wrote supplemental EAGLE SVG figures")
