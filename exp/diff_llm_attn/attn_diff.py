import argparse
import json

from ..tools.plot import *
from ..tools.corrs import *
from ..tools.utils import *


def parse_args():
    p = argparse.ArgumentParser("Length sweep attention similarity (Route A + Multi-GPU + JSONL + Plot)")

    p.add_argument("--sweep_step", type=int, default=10, help="If >0, ignore --lengths and sweep to --max_L.")
    p.add_argument("--max_L", type=int, default=3400, help="Upper bound when --sweep_step>0.")

    p.add_argument("--plot_dpi", type=int, default=300)
    p.add_argument("--plot_type", type=str, required=True, choices=["Pearson", "Spearman", "JSD"])

    p.add_argument("--large_llm_attn_path", type=str, required=True)
    p.add_argument("--small_llm_attn_path", type=str, required=True)

    p.add_argument("--save_dir", type=str, required=True)

    return p.parse_args()

def main():
    args = parse_args()

    with open(args.large_llm_attn_path, "r", encoding="utf-8") as f:
        large_llm_attn = [json.loads(line) for line in f]

    # Load small LLM attention distribution
    with open(args.small_llm_attn_path, "r", encoding="utf-8") as f:
        small_llm_attn = [json.loads(line) for line in f]

    sample_nums = len(large_llm_attn)
    xs = []
    corrs = []
    means: List[float] = []
    stds: List[float] = []
    L_list = build_sweep_lengths(int(args.sweep_step), int(args.max_L))

    for L in L_list:
        if args.plot_type == "Pearson":
            for i in range(sample_nums):
                corr = pearson_corr(large_llm_attn[i]['vecs'][str(L)], small_llm_attn[i]['vecs'][str(L)])
                corrs.append(corr)
        if args.plot_type == "Spearman":
            for i in range(sample_nums):
                corr = spearman_corr(large_llm_attn[i]['vecs'][str(L)], small_llm_attn[i]['vecs'][str(L)])
                corrs.append(corr)
        if args.plot_type == "JSD":
            for i in range(sample_nums):
                corr = js_divergence(large_llm_attn[i]['vecs'][str(L)], small_llm_attn[i]['vecs'][str(L)])
                corrs.append(corr)
        m, s = mean_std(corrs)

        if m is None:
            continue

        xs.append(L)
        means.append(0.0 if m is None else float(m))
        stds.append(0.0 if s is None else float(s))

    if xs:
        ylim = nice_ylim(means, stds, clamp=(-1.0, 1.0))

        save_single_metric_plot(
            xs, means, stds,
            title=f"Large and small Language Model Attention {args.plot_type}",
            ylabel=f"{args.plot_type}",
            color="tab:blue",
            out_path=f"{args.save_dir}/{args.plot_type}.png",
            ylim=ylim,
            dpi=int(args.plot_dpi),
        )
        print(f">>> Saved plot: {args.save_dir}/{args.plot_type}.png")


if __name__ == "__main__":
    main()
