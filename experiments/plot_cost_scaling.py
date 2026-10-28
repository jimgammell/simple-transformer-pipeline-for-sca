import argparse
from pathlib import Path

import numpy as np
from init_things import DPI, OUTPUTS_ROOT, WIDTH
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter


def run_plot_cost_scaling(dest: Path):
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, WIDTH / 4))
    benchmark_path = OUTPUTS_ROOT / "compute_benchmark" / "results.npz"
    benchmark = np.load(benchmark_path, allow_pickle=True)

    sweep_var = benchmark["sweep_var"]
    param_count = benchmark["param_count"]
    flops = benchmark["flops"]
    wall_time_ms = benchmark["wall_time_ms"]
    vram_gb = benchmark["vram_mb"] / 1024

    # Base values used when a parameter is held fixed — x is normalised to these.
    base_vals = {"patch_count": 64, "layer_count": 8, "embedding_dim": 512}

    sweep_cfgs = {
        "embedding_dim": ("Hidden dim (base=512)", benchmark["embedding_dim"]),
        "layer_count": ("Layer count (base=8)", benchmark["layer_count"]),
        "patch_count": ("Patch count (base=64)", benchmark["patch_count"]),
    }
    colors = ["red", "blue", "green"]

    panel_specs = [
        (param_count / 1e6, r"Parameters (M)", FuncFormatter(lambda v, _: f"{v:.0f}M")),
        (flops / 1e12, r"TFLOPs/step", FuncFormatter(lambda v, _: f"{v:.0f}T")),
        (vram_gb, r"VRAM [GB]", None),
        (wall_time_ms, r"Time/step [ms]", None),
    ]

    for ax, (metric_data, ylabel, yfmt) in zip(axes, panel_specs):
        for color, (sv_key, (sv_label, sv_raw)) in zip(colors, sweep_cfgs.items()):
            mask = sweep_var == sv_key
            if not mask.any():
                continue
            x = sv_raw[mask].astype(float) / base_vals[sv_key]
            ax.plot(
                x,
                metric_data[mask],
                color=color,
                marker="none",
                linewidth=0.75,
                label=sv_label,
                rasterized=True,
            )

        if yfmt is not None:
            ax.yaxis.set_major_formatter(yfmt)
        ax.set_xlabel("Fraction of base")
        ax.set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(sweep_cfgs),
        framealpha=0,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.tight_layout()
    fig.savefig(dest, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot saved compute_benchmark/results.npz; no benchmark is run."
    )
    parser.add_argument(
        "--dest",
        default=None,
        type=Path,
        help="Output directory for cost_scaling.pdf (default: OUTPUTS_ROOT/plots_for_paper).",
    )
    args = parser.parse_args()

    dest = args.dest
    if dest is None:
        dest = OUTPUTS_ROOT / "plots_for_paper"
    dest.mkdir(exist_ok=True, parents=True)

    run_plot_cost_scaling(dest / "cost_scaling.pdf")


if __name__ == "__main__":
    main()
