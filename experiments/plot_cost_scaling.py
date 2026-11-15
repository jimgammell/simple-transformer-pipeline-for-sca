import argparse
import json
from pathlib import Path

import numpy as np
from init_things import DPI, OUTPUTS_ROOT, WIDTH
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from utils.artifacts import sha256_file


def run_plot_cost_scaling(dest: Path, benchmark_path: Path | None = None):
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, WIDTH / 4))
    if benchmark_path is None:
        benchmark_path = OUTPUTS_ROOT / "compute_benchmark" / "results.npz"
    benchmark = np.load(benchmark_path, allow_pickle=False)

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

    for ax, (metric_data, ylabel, yfmt) in zip(axes, panel_specs, strict=True):
        for color, (sv_key, (sv_label, sv_raw)) in zip(
            colors, sweep_cfgs.items(), strict=True
        ):
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


def plot_patch_tradeoff(dest, benchmark_path):
    """Single-column saved-artifact plot, normalized to the 64-patch recipe."""
    provenance_path = benchmark_path.with_name("provenance.json")
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    with np.load(benchmark_path, allow_pickle=False) as benchmark:
        settings = {key: benchmark[key].tolist() for key in ("compile", "precision", "flop_convention")
                    if key in benchmark}
        mask = benchmark["sweep_var"] == "patch_count"
        x = benchmark["patch_count"][mask]
        order = np.argsort(x)
        x = x[order]
        if len(x) != len(np.unique(x)) or np.count_nonzero(x == 64) != 1:
            raise ValueError("Expected unique patch counts including the 64-patch reference")
        if np.any(benchmark["oom"][mask]):
            raise ValueError("Patch sweep contains OOM results")
        for key in ("layer_count", "embedding_dim"):
            if len(np.unique(benchmark[key][mask])) != 1:
                raise ValueError(f"Patch sweep changes {key}")
        reference = int(np.flatnonzero(x == 64)[0])
        metrics = {}
        for key, label in (("param_count", "Parameters"), ("wall_time_ms", "Time"), ("vram_mb", "Memory"), ("flops", "FLOPs")):
            values = benchmark[key][mask][order].astype(float)
            if not np.all(np.isfinite(values) & (values > 0)):
                raise ValueError(f"Invalid {key}")
            metrics[label] = dict(field=key, values=values.tolist(),
                                  reference=float(values[reference]), normalized=(values / values[reference]).tolist())
    with plt.rc_context({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8}):
        fig, ax = plt.subplots(figsize=(3.5, 2.0))
        for (label, metric), color, marker in zip(metrics.items(),
                ("#0072B2", "#D55E00", "#009E73", "#CC79A7"), ("o", "s", "^", "D"), strict=True):
            base = metric["reference"]
            legend_label = {
                "Parameters": f"Parameters\n(base = {base / 1e6:.1f}M)",
                "Time": f"Time (fwd+bkwd)\n(base = {base:.0f} ms)",
                "Memory": f"VRAM\n(base = {base * 2**20 / 10**9:.2f} GB)",
                "FLOPs": f"FLOPs (fwd+bkwd)\n(base = {base / 1e12:.2f}T)",
            }[label]
            ax.plot(x, metric["normalized"], label=legend_label, color=color,
                    marker=marker, markersize=4, linewidth=1.3)

        ymax = max(max(metric["normalized"]) for metric in metrics.values())
        ax.set(xlabel="Patch count", ylabel=r"Cost ($\times$ base)",
               xlim=(0, float(x.max()) * 1.03), ylim=(0, ymax * 1.08))
        ax.set_xticks(np.linspace(0, x.max(), 5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
        ax.grid(False)
        ax.legend(frameon=False, loc="upper left", ncol=2, labelspacing=0.2,
                  borderaxespad=0.15, borderpad=0.2, handlelength=1.0,
                  handletextpad=0.5, columnspacing=0.8)
        fig.tight_layout(pad=0.5)
        fig.savefig(dest / "patch-cost-tradeoff.pdf")
        fig.savefig(dest / "patch-cost-tradeoff.png", dpi=200)
        plt.close(fig)
    (dest / "patch-cost-tradeoff.json").write_text(json.dumps(dict(
        source=str(benchmark_path.resolve()), source_sha256=sha256_file(benchmark_path),
        patch_count=x.tolist(), metrics=metrics,
        benchmark_settings=settings, benchmark_provenance=provenance,
        provenance_sha256=sha256_file(provenance_path) if provenance_path.exists() else None,
        scope="Synthetic forward/backward, not optimizer-inclusive training; memory is peak PyTorch allocated MiB, not reserved. Each curve is normalized to its saved 64-patch value. Execution settings are recorded in benchmark_settings and benchmark_provenance when available.",
    ), indent=2) + "\n")


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
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=None,
        help="Saved results.npz (default: OUTPUTS_ROOT/compute_benchmark/results.npz).",
    )
    parser.add_argument("--patch-tradeoff", action="store_true", help="Single-column normalized patch-count plot")
    args = parser.parse_args()

    dest = args.dest
    if dest is None:
        dest = OUTPUTS_ROOT / "plots_for_paper"
    dest.mkdir(exist_ok=True, parents=True)

    if args.patch_tradeoff:
        if args.benchmark_path is None:
            parser.error("--patch-tradeoff requires --benchmark-path")
        plot_patch_tradeoff(dest, args.benchmark_path)
    else:
        run_plot_cost_scaling(dest / "cost_scaling.pdf", args.benchmark_path)


if __name__ == "__main__":
    main()
