"""Plot saved seed-zero sensitivity attacks against five-seed baselines."""

import argparse
import csv
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

from utils.artifacts import load_attack, seed_runs


DATASETS = ("ascadv1-fixed", "ascadv1-variable", "ches-ctf-2018", "eshards-non-shuffled")
PARAMETERS = ("base_lr", "embedding_dim", "trunk_blocks", "patch_size")
LABELS = ("Learning rate", "Width", "Depth", "Patch size")


def attack_row(run, dataset, parameter, value, kind):
    config = yaml.safe_load((run / "config.yaml").read_text())
    steps = int(pd.read_csv(run / "metrics.csv", usecols=["step"]).step.max())
    target = config["training"]["total_steps"]
    if steps < target - 1:
        raise ValueError(f"{run}: incomplete training ({steps + 1}/{target})")
    summary_path = run / "training_summary.yaml"
    summary = yaml.safe_load(summary_path.read_text()) if summary_path.exists() else {}
    ranks = load_attack(run)
    orders, horizon, _ = ranks.shape
    if horizon != config["mtd"]["traces_per_attack"] or orders != config["mtd"]["attack_count"]:
        raise ValueError(f"{run}: attack shape does not match configured horizon/orders")
    # Sustained disclosure time: last error plus one.
    times = np.where(ranks > 1, np.arange(1, horizon + 1)[None, :, None], 0).max(axis=1) + 1
    return dict(dataset=dataset, parameter=parameter, value=value, kind=kind,
                seed=config["training"]["seed"], status="complete",
                full_key_mtd=float(times.max(axis=1).mean()),
                mean_byte_mtd=float(times.mean()), horizon=horizon, attack_orders=orders,
                censored_orders=int((times > horizon).any(axis=1).sum()),
                censored_byte_orders=int((times > horizon).sum()),
                completed_steps=steps + 1, configured_steps=target,
                selected_global_step=summary.get("selected_global_step", ""),
                source=str(run / "attack_metrics.npz"), config_source=str(run / "config.yaml"),
                training_source=str(run / "metrics.csv"),
                censoring="Nonrecovery contributes horizon+1; sentinel-coded, not uncensored")


def panel(ax, rows, baseline, parameter, label, center):
    selected = sorted([r for r in rows if r["parameter"] == parameter], key=lambda r: r["value"])
    for metric, color in (("full_key_mtd", "C0"), ("mean_byte_mtd", "C1")):
        values = [r[metric] for r in baseline]
        mean, sd = np.mean(values), np.std(values, ddof=1)
        ax.axhspan(max(1, mean - sd), mean + sd, color=color, alpha=.12)
        ax.axhline(mean, color=color, ls="--", lw=1)
        points = [(r["value"], r.get(metric, np.nan)) for r in selected]
        points.append((center, baseline[0][metric]))
        points.sort()
        ax.plot(*zip(*points), color=color, marker="o", ms=3, lw=1)
        ax.plot(center, baseline[0][metric], marker="s", color=color, ms=5)
        for row in selected:
            if row.get("censored_orders", 0):
                ax.plot(row["value"], row[metric], marker="^", color=color, ms=7)
    for row in selected:
        if row["status"] != "complete":
            ax.axvline(row["value"], color=".5", ls=":", lw=1)
            ax.text(row["value"], .98, "missing attack", transform=ax.get_xaxis_transform(),
                    rotation=90, va="top", ha="right", fontsize=7, color=".35")
    ax.set_xlabel(label)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    low, high = ax.get_ylim()
    ax.yaxis.set_minor_formatter(
        FuncFormatter(lambda value, _: f"{value:g}") if high / low < 3 else NullFormatter()
    )
    ax.set_ylabel("Sustained MTD (traces)")
    ax.grid(alpha=.2)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.tick_params(axis="both", labelsize=8)


def compact(args):
    """Draft a final-size, paired-seed view using only the existing summary."""
    outputs = [args.dest / f"compact.{suffix}" for suffix in ("pdf", "png", "md")]
    if any(path.exists() for path in outputs) and not args.overwrite:
        warnings.warn("Skipping existing compact outputs; use --overwrite")
        return
    rows = pd.read_csv(args.dest / "metrics.csv")
    if len(rows) != 84 or len(rows[rows.kind == "perturbation"]) != 64:
        raise ValueError("Expected 64 perturbations and 20 baseline rows")
    notes = ["# Compact sensitivity draft", "",
             "One-factor seed-0 perturbations; both axes normalized to the same dataset's seed-0 baseline.",
             "Full-key sustained MTD is averaged over attack orders. The horizontal reference is 1.",
             "Triangles mark lower bounds when any attack order is right-censored; lines break at those points.",
             "For these bounds, each horizon+1 sentinel is replaced by the horizon before averaging.",
             "No training-seed uncertainty is implied; five-seed variability is reported below, not shaded.",
             "Size: 43pc / 72.27 = 7.140 in wide, 2.25 in high (IEEEtran conference full text width).",
             "Fonts: 8 pt at native size. Dataset labels: ASCADf = ASCAD fixed; ASCADr = ASCAD variable; CHES = CHES CTF 2018.", ""]
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                         "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 4, figsize=(516 / 72.27, 2.25), sharey=True)
        for index, (dataset, short) in enumerate(zip(DATASETS, ("ASCADf", "ASCADr", "CHES", "eShard"))):
            baseline = rows[(rows.dataset == dataset) & (rows.kind == "baseline")]
            reference = baseline[baseline.seed == 0].iloc[0]
            denominator = float(reference.full_key_mtd)
            config = yaml.safe_load(Path(reference.config_source).read_text())
            notes.append(f"- {short}: seed-0 baseline {denominator:g} traces; five-seed mean ± sample SD "
                         f"{baseline.full_key_mtd.mean():.4g} ± {baseline.full_key_mtd.std(ddof=1):.4g} traces.")
            for ax, parameter in zip(axes, PARAMETERS):
                center = config["training" if parameter == "base_lr" else "model"][parameter]
                selected = rows[(rows.dataset == dataset) & (rows.parameter == parameter)].sort_values("value")
                points = [(1., 1., False)]
                for row in selected.itertuples():
                    censored = row.status == "complete" and row.censored_orders > 0
                    value = row.full_key_mtd
                    if censored:
                        value -= row.censored_orders / row.attack_orders
                    points.append((row.value / center, value / denominator, censored))
                points.sort()
                x, y, censored = map(np.asarray, zip(*points))
                ax.plot(x, np.where(censored, np.nan, y), color=f"C{index}", marker="o", ms=3, lw=1,
                        label=short)
                ax.plot(x[censored], y[censored], color=f"C{index}", marker="^", ms=5, ls="none")
        for ax, label in zip(axes, LABELS):
            ax.set_title(label)
            ax.set_xlabel("Value / baseline")
            ax.set_yscale("log")
            ax.axhline(1, color=".5", ls="--", lw=.7, zorder=0)
            ax.grid(alpha=.2, which="major")
            ax.set_xticks([.5, 1, 1.5, 2, 2.5] if ax.get_xlim()[1] > 2 else [.5, 1, 1.5])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
        axes[0].set_ylabel("Full-key MTD / baseline")
        handles, labels = axes[0].get_legend_handles_labels()
        handles.append(Line2D([], [], color=".3", marker="^", ls="none"))
        labels.append("Censored lower bound")
        fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
                   bbox_to_anchor=(.52, 1.01), handlelength=1.3, columnspacing=1.2)
        fig.subplots_adjust(left=.085, right=.99, bottom=.22, top=.76, wspace=.16)
        fig.savefig(outputs[0])
        fig.savefig(outputs[1], dpi=200)
        plt.close(fig)
    outputs[2].write_text("\n".join(notes) + "\n")
    print("Saved " + ", ".join(map(str, outputs)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("outputs/sensitivity"))
    parser.add_argument("--baseline-root", type=Path, default=Path("outputs"))
    parser.add_argument("--dest", type=Path, default=Path("outputs/figures/sensitivity"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compact", action="store_true", help="Plot compact draft from existing metrics.csv only")
    args = parser.parse_args()
    if args.compact:
        compact(args)
        return
    if args.dest.exists() and any(args.dest.iterdir()) and not args.overwrite:
        warnings.warn(f"Skipping existing output bundle {args.dest}; use --overwrite")
        return
    args.dest.mkdir(parents=True, exist_ok=True)
    all_rows = []
    overview, overview_axes = plt.subplots(4, 4, figsize=(15, 12))
    legend = [Line2D([], [], color="C0", label="Full-key MTD"),
              Line2D([], [], color="C1", label="Mean-byte MTD (all 16)"),
              Line2D([], [], color=".3", marker="o", label="Seed 0 sweep"),
              Line2D([], [], color=".3", marker="s", label="Seed 0 baseline"),
              Line2D([], [], color=".3", ls="--", label="5-seed baseline mean ± sample SD (band)"),
              Line2D([], [], color=".3", marker="^", ls="none", label="Censored orders: nonrecovery = horizon+1")]
    for index, dataset in enumerate(DATASETS):
        baseline = [attack_row(run, dataset, "baseline", "", "baseline")
                    for run in seed_runs(args.baseline_root / dataset)]
        config = yaml.safe_load((args.baseline_root / dataset / "seed=0/config.yaml").read_text())
        rows = []
        # Use the maintained launch script as the explicit list of requested perturbations.
        script = Path("scripts") / f"run_sensitivity_{dataset.replace('-', '_')}.sh"
        lines = script.read_text().splitlines()
        settings = [line.replace(";", "").split()[3:-1] for line in lines if line.startswith("for value in ")]
        if len(settings) != 4 or sum(map(len, settings)) != 16:
            raise ValueError(f"{script}: expected four parameters and 16 perturbations")
        for parameter, values in zip(PARAMETERS, settings):
            for value in values:
                run = args.results_root / dataset / parameter / value / "seed=0"
                if (run / "attack_metrics.npz").exists():
                    row = attack_row(run, dataset, parameter, float(value), "perturbation")
                    section = "training" if parameter == "base_lr" else "model"
                    actual = yaml.safe_load((run / "config.yaml").read_text())[section][parameter]
                    if not np.isclose(actual, float(value)) or row["seed"] != 0:
                        raise ValueError(f"{run}: setting/seed mismatch")
                else:
                    row = dict(dataset=dataset, parameter=parameter, value=float(value),
                               kind="perturbation", seed=0, status="missing attack",
                               source=str(run / "attack_metrics.npz"))
                rows.append(row)
        horizons = {r["horizon"] for r in baseline + rows if r["status"] == "complete"}
        if len(horizons) != 1:
            raise ValueError(f"{dataset}: incomparable horizons {horizons}")
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for j, (parameter, label) in enumerate(zip(PARAMETERS, LABELS)):
            center = config["training" if parameter == "base_lr" else "model"][parameter]
            panel(axes.flat[j], rows, baseline, parameter, label, center)
            panel(overview_axes[index, j], rows, baseline, parameter, label, center)
            overview_axes[index, j].set_title(dataset, fontsize=9)
        count = sum(r["status"] == "complete" for r in rows)
        fig.suptitle(f"{dataset}: {count}/16 perturbations; baseline five seeds; horizon {next(iter(horizons))}")
        fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=8)
        fig.tight_layout(rect=(0, .13, 1, .95))
        fig.savefig(args.dest / f"{dataset}.pdf")
        plt.close(fig)
        all_rows.extend(baseline + rows)
        print(f"{dataset}: {count}/16; horizon {horizons.pop()}")
    overview.legend(handles=legend, loc="lower center", ncol=3, fontsize=8)
    horizons = sorted({r["horizon"] for r in all_rows if r["status"] == "complete"})
    overview.suptitle(f"Sensitivity: seed 0; five-seed baseline mean ± sample SD; horizons {horizons} traces")
    overview.tight_layout(rect=(0, .06, 1, .96))
    overview.savefig(args.dest / "overview.pdf")
    plt.close(overview)
    fields = list(dict.fromkeys(key for row in all_rows for key in row))
    with (args.dest / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved five PDFs and metrics.csv to {args.dest}")


if __name__ == "__main__":
    main()