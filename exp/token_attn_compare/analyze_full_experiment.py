#!/usr/bin/env python3
"""Create readable tables and a scientific figure from completed experiment files."""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def block_interval(path, method, control, seed):
    def blocks(name):
        return np.array([len(json.loads(s)["blocks"]) for s in
                         (path/f"test_{name}.jsonl").read_text().splitlines()])
    delta=blocks(method)-blocks(control)
    indices=np.random.default_rng(seed).integers(0,len(delta),(2000,len(delta)))
    return dict(delta_total_blocks=int(delta.sum()),
                paired_95pct_total_blocks_interval=np.quantile(delta[indices].sum(1),[0.025,0.975]).tolist(),
                fewer_blocks_questions=int((delta<0).sum()),more_blocks_questions=int((delta>0).sum()),
                tied_blocks_questions=int((delta==0).sum()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepared",required=True)
    parser.add_argument("--freshness",required=True)
    parser.add_argument("--qkv",required=True)
    args=parser.parse_args()
    output=Path(args.prepared)
    directories={"freshness":Path(args.freshness),"qkv":Path(args.qkv)}
    summaries={k:read(v/"summary.json") for k,v in directories.items()}
    test_frames=[r for r in read(output/"manifest.json") if r["split"]=="test"]
    questions=len(test_frames);tokens=sum(r["labels"] for r in test_frames)
    draft_k=summaries["qkv"]["config"]["draft_k"]
    lines=["# 完整实验结果", "", f"指标均来自同一 {questions} 题、{tokens:,} 个固定 teacher 输出 token，草稿长度 k={draft_k}。",
           "差值单位为百分点，95% 区间按题配对 bootstrap 2,000 次，未做多重比较校正；属于探索性结果。", ""]
    block_results={}
    for experiment,summary in summaries.items():
        lines.extend(["## "+("注意力新鲜度" if experiment=="freshness" else "Q/K/V 训练"),""])
        lines.extend(["| 路线 | λ / epoch | 接受率 | accepted / proposed | 相对基线，百分点 [95% 区间] | 验证块 |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"])
        block_results[experiment]={}
        for name,score in summary["methods"].items():
            choice=(summary["selection"][name]["strength"] if experiment=="freshness" else score.get("selected_epoch",0))
            interval=score.get("paired_95pct_delta_interval",[0,0])
            delta=score.get("delta_accept_rate",0)
            lines.append(f"| {name} | {choice} | {score['accept_rate']*100:.3f}% | {score['accepted']} / {score['proposed']} | {delta*100:+.3f} [{interval[0]*100:+.3f}, {interval[1]*100:+.3f}] | {score['blocks']} |")
            block_results[experiment][name]=block_interval(directories[experiment],name,"baseline",summary["config"]["seed"])
        lines.extend(["", "### 配对对照", "", "| 比较（前者减后者） | 差值，百分点 | 95% 区间 |", "| --- | ---: | ---: |"])
        for name,comparison in summary["paired_controls"].items():
            lo,hi=comparison["paired_95pct_delta_interval"]
            lines.append(f"| {name} | {comparison['delta_accept_rate']*100:+.3f} | [{lo*100:+.3f}, {hi*100:+.3f}] |")
        lines.append("")
    lines.extend(["## 共同前缀输出诊断", "", "各方法在 teacher 正确轨迹的相同位置前向；此命中率不是完整草稿回放的接受率。", "",
                  "| 路线 | 命中数 / 位置数 | 命中率 | 输出 KL |", "| --- | ---: | ---: | ---: |"])
    for name,score in summaries["qkv"]["fixed_context"].items():
        lines.append(f"| {name} | {score['hits']} / {score['tokens']} | {100*score['match']:.3f}% | {score['output_kl']:.5f} |")
    diagnostic_path=directories["qkv"]/"relation_diagnostics.json"
    if diagnostic_path.exists():
        lines.extend(["", "## 共同测试前缀的关系对齐", "", "KL 越低表示更接近 teacher 的头平均 token 关系；这些诊断不参与 checkpoint 选择。", "",
                      "| 路线 | Attention KL | Value 关系 KL |", "| --- | ---: | ---: |"])
        for name,score in read(diagnostic_path)["methods"].items():
            lines.append(f"| {name} | {score['attention_kl']:.5f} | {score['value_relation_kl']:.5f} |")
    lines.extend(["", "## 动态注意力的额外成本", "", "| 路线 | teacher 额外单 query 调用 | 原有验证块数 |", "| --- | ---: | ---: |"])
    for name,score in summaries["freshness"]["methods"].items():
        lines.append(f"| {name} | {score['teacher_attention_calls']} | {score['blocks']} |")
    lines.extend(["", "调用数用于揭示 oracle 成本，不能直接折算成 wall-clock 加速；本轮并行运行，原型计时不作性能比较。",
                  "", "Q/K/V 推理只需合并后的 student 权重，teacher 关系只用于训练。训练方案均为单种子；模型层数相同，本轮不检验不同层数的对应关系。", ""])
    (output/"results.md").write_text("\n".join(lines))
    (output/"statistical_analysis.json").write_text(json.dumps(block_results,indent=2)+"\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout="constrained")
    routes={"freshness":["static_selected","dynamic_selected","dynamic_full_selected","static_unit","dynamic_unit","dynamic_full_unit"],
            "qkv":[p+"_"+s for p in ("q","qk","qv","qkv") for s in ("output","relations")]}
    labels={"static_selected":"Static document (selected)","dynamic_selected":"Current document (selected)",
            "dynamic_full_selected":"Current full (selected)","static_unit":"Static document (λ=1)",
            "dynamic_unit":"Current document (λ=1)","dynamic_full_unit":"Current full (λ=1)"}
    for ax,(experiment,names) in zip(axes,routes.items()):
        for i,name in enumerate(names):
            score=summaries[experiment]["methods"][name]
            delta=100*score["delta_accept_rate"]
            lo,hi=[100*v for v in score["paired_95pct_delta_interval"]]
            color="#147d92" if name.endswith("relations") or name.startswith("dynamic") else "#c87a38"
            ax.plot([lo,hi],[i,i],color=color,lw=2)
            ax.plot(delta,i,"o",color=color,ms=6)
        ax.axvline(0,color="0.5",ls="--",lw=1)
        ax.set_yticks(range(len(names)),[labels.get(n,n.replace("_", " / ").upper()) for n in names])
        ax.invert_yaxis();ax.grid(axis="x",alpha=.2)
        ax.set_xlabel("Acceptance change vs baseline (percentage points)")
        ax.spines[["top","right","left"]].set_visible(False)
        ax.set_title("Attention freshness" if experiment=="freshness" else "Full-prefix projection training")
    fig.suptitle(f"{questions} questions · paired 95% bootstrap intervals · exploratory",fontsize=13)
    fig.savefig(output/"acceptance_comparison.png",dpi=180)
    svg_path=output/"acceptance_comparison.svg"
    fig.savefig(svg_path)
    # Matplotlib emits trailing spaces in SVG paths; keep the Git artifact clean.
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines())+"\n")
    print(output/"results.md")


if __name__=="__main__":
    main()
