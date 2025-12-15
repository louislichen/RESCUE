import math
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt


def save_single_metric_plot(
    xs: List[int],
    means: List[float],
    stds: List[float],
    title: str,
    ylabel: str,
    color: str,
    out_path: Path,
    ylim: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
):
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    # Mean line
    ax.plot(xs, means, marker="o", markersize=3.5, linewidth=2.0, color=color, label="Mean")

    # Std band (more readable than dense error bars)
    lower = [m - s for m, s in zip(means, stds)]
    upper = [m + s for m, s in zip(means, stds)]
    ax.fill_between(xs, lower, upper, color=color, alpha=0.18, linewidth=0.0)

    ax.set_title(title)
    ax.set_xlabel("Context length L (tokens)")
    ax.set_ylabel(ylabel)

    # Grid: major + minor
    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", alpha=0.18)

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    # cleaner legend
    ax.legend(loc="best", frameon=True, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def derive_plot_paths(output_plot: str) -> Tuple[Path, Path, Path]:
    """
    If output_plot="length_sweep_corr.png",
    save:
      length_sweep_corr_pearson.png
      length_sweep_corr_spearman.png
      length_sweep_corr_jsd.png
    """
    p = Path(output_plot)
    stem = p.stem
    suffix = p.suffix if p.suffix else ".png"
    base_dir = p.parent if str(p.parent) != "" else Path(".")
    pear = base_dir / f"{stem}_pearson{suffix}"
    spear = base_dir / f"{stem}_spearman{suffix}"
    jsd = base_dir / f"{stem}_jsd{suffix}"
    return pear, spear, jsd


def nice_ylim(data_mean: List[float], data_std: List[float], clamp: Optional[Tuple[float, float]] = None):
    ys = []
    for m, s in zip(data_mean, data_std):
        ys.append(m - s)
        ys.append(m + s)
    ys = [y for y in ys if math.isfinite(float(y))]
    if not ys:
        return None

    lo = min(ys)
    hi = max(ys)
    if lo == hi:
        lo -= 0.05
        hi += 0.05

    # add margin
    span = hi - lo
    lo -= 0.06 * span
    hi += 0.06 * span

    if clamp is not None:
        lo = max(clamp[0], lo)
        hi = min(clamp[1], hi)
        # if clamped to a point, expand a bit
        if lo >= hi:
            mid = 0.5 * (clamp[0] + clamp[1])
            lo, hi = mid - 0.1, mid + 0.1

    return lo, hi