import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.artifacts import summarize_ranks

DATASETS = (
    "ascadv1-fixed",
    "ascadv1-variable",
    "ascadv2",
    "ches-ctf-2018",
    "eshards-non-shuffled",
)
PLOTS = (
    "learning-curves",
    "attack-curves",
    "byte-curves",
    "mtd-table",
    "accuracy-table",
    "attributions",
)


def seed_runs(root):
    runs = sorted(Path(root).glob("seed=*"))
    if {run.name for run in runs} != {f"seed={seed}" for seed in range(5)}:
        raise ValueError(
            f"{root}: require exactly seed=0 through seed=4; found {[run.name for run in runs]}"
        )
    return runs


def load_attack(run):
    with np.load(run / "attack_metrics.npz", allow_pickle=False) as data:
        if "ranks_zero_based" not in data:
            raise ValueError(
                f"{run}: missing ranks_zero_based; joint success and order variation cannot be recovered from mean ranks"
            )
        ranks = data["ranks_zero_based"]
        if (
            not np.issubdtype(ranks.dtype, np.integer)
            or not ((ranks >= 0) & (ranks <= 255)).all()
        ):
            raise ValueError(f"{run}: require integer zero-based ranks in 0..255")
        ranks = ranks.astype(np.int16) + 1
    summarize_ranks(ranks)
    if ranks.shape[2] != 16:
        raise ValueError(f"{run}: expected all 16 key-byte ranks, got {ranks.shape}")
    return ranks


def summary_row(group, values, **extra) -> dict:
    return dict(
        group=group,
        mean=float(np.mean(values)),
        sample_sd=float(np.std(values, ddof=1)),
        seed_count=len(values),
        seed_values=[float(value) for value in values],
        **extra,
    )


def write_table(rows, dest, name, overwrite):
    paths = [dest / f"{name}.{suffix}" for suffix in ("json", "csv", "tex")]
    if not overwrite and any(path.exists() for path in paths):
        warnings.warn(
            f"Skipping existing output bundle {dest / name}; use --overwrite",
            stacklevel=2,
        )
        return
    dest.mkdir(parents=True, exist_ok=True)
    paths[0].write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with paths[1].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [r"\begin{tabular}{lrr}", r"Group & Mean & Sample SD \\", r"\hline"]
    for row in rows:
        label = row["group"].replace("_", r"\_")
        lines.append(f"{label} & {row['mean']:.6g} & {row['sample_sd']:.6g} " + r"\\")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\par Mean and sample SD across five training seeds (seed=0 through seed=4)."
    )
    if name == "mtd-table":
        lines.append(
            f"\\par Attack horizon: {rows[0]['horizon']}; nonrecovery contributes horizon+1. Sentinel-coded means are not uncensored estimates."
        )
        lines.extend(
            [
                r"\par",
                r"\begin{tabular}{lll}",
                r"Group & Censored orders by seed & Censored byte-orders by seed \\",
            ]
        )
        for row in rows:
            lines.append(
                f"{row['group']} & {','.join(map(str, row['censored_orders']))} & {','.join(map(str, row['censored_byte_orders']))} "
                + r"\\"
            )
        lines.append(r"\end{tabular}")
    paths[2].write_text("\n".join(lines) + "\n")
    for path in paths:
        print(f"saved {path}")


def mtd_table(runs, dataset, dest, overwrite):
    ranks = [load_attack(run) for run in runs]
    horizons = {rank.shape[1] for rank in ranks}
    if len(horizons) != 1:
        raise ValueError(f"{dataset}: incomparable attack horizons {horizons}")
    horizon = horizons.pop()
    mtd = []
    for rank in ranks:
        incorrect = rank > 1
        times = (
            np.where(incorrect, np.arange(1, horizon + 1)[None, :, None], 0).max(axis=1)
            + 1
        )
        mtd.append(times)
    groups = [(f"byte-{byte}", [byte]) for byte in range(16)]
    groups += [("full-key", list(range(16))), ("mean-all-bytes", list(range(16)))]
    if dataset.startswith("ascadv1"):
        groups.append(("mean-masked-bytes-2..15", list(range(2, 16))))
    rows = []
    for group, indices in groups:
        per_order = [
            values[:, indices].max(axis=1)
            if group == "full-key"
            else values[:, indices].mean(axis=1)
            for values in mtd
        ]
        rows.append(
            summary_row(
                group,
                [values.mean() for values in per_order],
                horizon=horizon,
                censor_sentinel=horizon + 1,
                censored_orders=[
                    int((values[:, indices] > horizon).any(axis=1).sum())
                    for values in mtd
                ],
                censored_byte_orders=[
                    int((values[:, indices] > horizon).sum()) for values in mtd
                ],
                attack_counts=[len(values) for values in mtd],
                censoring="nonrecovery contributes horizon+1; means are sentinel-coded, not uncensored estimates",
            )
        )
    write_table(rows, dest, "mtd-table", overwrite)


def head_groups(dataset):
    if dataset == "ascadv2":
        return {
            "alpha": [0],
            "beta": [1],
            "masked": list(range(2, 18)),
            "permutation": list(range(18, 34)),
        }
    if dataset.startswith("ascadv1"):
        return {"masked-bytes-2..15": list(range(2, 16))}
    return {"all-bytes": list(range(16))}


def accuracy_table(runs, dataset, dest, overwrite):
    heads = 34 if dataset == "ascadv2" else 16
    values = []
    full = []
    for run in runs:
        with np.load(run / "attack_metrics.npz", allow_pickle=False) as data:
            values.append(
                [float(data[f"test/acc/{head}"].item()) for head in range(heads)]
            )
            full.append(float(data["test/acc"].item()))
    values = np.asarray(values)
    if (
        not np.isfinite(values).all()
        or not ((values >= 0) & (values <= 1)).all()
        or not all(0 <= value <= 1 for value in full)
    ):
        raise ValueError(f"{dataset}: accuracy outside [0,1]")
    names = (
        (
            ["alpha", "beta"]
            + [f"masked-byte-{byte}" for byte in range(16)]
            + [f"permutation-{byte}" for byte in range(16)]
        )
        if heads == 34
        else [f"byte-{byte}" for byte in range(16)]
    )
    rows = [
        summary_row(f"{name}-ordinary-accuracy", values[:, head])
        for head, name in enumerate(names)
    ]
    rows.append(summary_row("mean-all-outputs-ordinary-accuracy", values.mean(axis=1)))
    for name, indices in head_groups(dataset).items():
        rows.append(
            summary_row(
                f"mean-{name}-ordinary-accuracy", values[:, indices].mean(axis=1)
            )
        )
    rows.append(
        summary_row(
            "all-34-outputs-correct" if heads == 34 else "all-16-outputs-correct", full
        )
    )
    for row in rows:
        row.update(
            partition="attack",
            reconstructed_key_accuracy=False,
        )
    write_table(rows, dest, "accuracy-table", overwrite)


def learning_figures(runs, dataset):
    data = []
    for run in runs:
        path = run / "training_summary.yaml"
        summary = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
        selected = summary.get("selected_global_step")
        if not isinstance(selected, int) or selected < 0:
            warnings.warn(
                f"{run}: missing selected_global_step; omitting checkpoint marker",
                stacklevel=2,
            )
            summary["selected_global_step"] = None
        data.append((pd.read_csv(run / "metrics.csv"), summary))
    figures = {}
    for group, indices in head_groups(dataset).items():
        fig, axes = plt.subplots(2, 3, figsize=(11, 6), squeeze=False)
        for seed, (metrics, summary) in enumerate(data):
            selected = summary.get("selected_global_step")
            color = f"C{seed}"
            for row, phase in enumerate(("train", "val")):
                for column, metric in enumerate(("loss", "acc", "rank")):
                    ax = axes[row, column]
                    columns = [f"{phase}/{metric}/{index}" for index in indices]
                    values = metrics[columns].to_numpy(dtype=float)
                    valid = np.isfinite(values).all(axis=1)
                    if not valid.any():
                        raise ValueError(
                            f"{runs[seed]}: no complete {phase}/{metric} rows for {group}"
                        )
                    x = metrics.loc[valid, "step"].to_numpy()
                    y = values[valid]
                    ax.plot(
                        x,
                        y.mean(axis=1),
                        color=color,
                        label=f"seed={seed}",
                        rasterized=True,
                    )
                    ax.fill_between(
                        x,
                        y.min(axis=1),
                        y.max(axis=1),
                        color=color,
                        alpha=0.12,
                        label="byte/head min–max" if seed == 0 else None,
                        rasterized=True,
                    )
                    if metric == "acc":
                        full = metrics[f"{phase}/acc"].notna()
                        if not full.any():
                            raise ValueError(
                                f"{runs[seed]}: missing {phase}/acc all-output accuracy"
                            )
                        ax.plot(
                            metrics.loc[full, "step"],
                            metrics.loc[full, f"{phase}/acc"],
                            color=color,
                            linestyle="--",
                            label="all outputs correct" if seed == 0 else None,
                            rasterized=True,
                        )
                    if selected is not None:
                        ax.axvline(
                            selected,
                            color=color,
                            linestyle=":",
                            linewidth=0.8,
                            label="selected checkpoint" if seed == 0 else None,
                        )
                    ylabel = {
                        "loss": "Loss",
                        "acc": "Ordinary accuracy",
                        "rank": "Single-trace rank",
                    }[metric]
                    ax.set(
                        xlabel="Training step", ylabel=ylabel, title=f"{phase}: {group}"
                    )
        for ax in axes.flat:
            ax.legend(fontsize=5)
        fig.suptitle(
            f"{dataset}: byte/head mean and min–max; dashed = all-output accuracy"
        )
        fig.tight_layout()
        figures[f"learning-curves-{group}"] = fig
    return figures


def attack_figures(runs, dataset, per_byte=False):
    ranks = [load_attack(run) for run in runs]
    groups = (
        {f"byte-{byte}": [byte] for byte in range(16)}
        if per_byte
        else {
            "masked-bytes-2..15"
            if dataset.startswith("ascadv1")
            else "all-key-bytes": list(
                range(2 if dataset.startswith("ascadv1") else 0, 16)
            )
        }
    )
    figures = {}
    for group, indices in groups.items():
        fig, axes = plt.subplots(1, 2 if per_byte else 3, figsize=(10, 3))
        for seed, rank in enumerate(ranks):
            x = np.arange(1, rank.shape[1] + 1)
            order_means = rank[:, :, indices].mean(axis=2) - 1
            lo, hi = np.quantile(order_means, [0.25, 0.75], axis=0)
            axes[0].plot(
                x,
                order_means.mean(axis=0),
                color=f"C{seed}",
                label=f"seed={seed}",
                rasterized=True,
            )
            axes[0].fill_between(
                x, lo, hi, color=f"C{seed}", alpha=0.12, rasterized=True
            )
            summary = summarize_ranks(rank)
            axes[1].plot(
                x,
                summary["success_rate"][indices].mean(axis=0),
                rasterized=True,
                color=f"C{seed}",
                label=f"seed={seed}",
            )
            if not per_byte:
                axes[2].plot(
                    x,
                    summary["full_key_success_rate"],
                    rasterized=True,
                    color=f"C{seed}",
                    label=f"seed={seed}",
                )
        axes[0].plot(
            [],
            [],
            color="grey",
            alpha=0.3,
            linewidth=5,
            label="IQR over attack orders (group mean); mean may lie outside IQR",
        )
        axes[0].set_ylabel("Mean per-byte GE (zero-based rank)")
        axes[1].set_ylabel("Mean per-byte success rate")
        if not per_byte:
            axes[2].set_ylabel("Joint full-key success rate (all 16 bytes)")
        for ax in axes:
            ax.set_xscale("log")
            ax.set_xlabel("Traces seen")
            ax.legend(fontsize=5)
        protocol = "; simulated-zero-key benchmark" if dataset == "ascadv2" else ""
        fig.suptitle(f"{dataset}: {group}{protocol}")
        fig.tight_layout()
        figures[f"{'byte-curves' if per_byte else 'attack-curves'}-{group}"] = fig
    return figures


def attribution_figures(runs, dataset):
    methods = set()
    for run in runs:
        for path in run.glob("*.npz"):
            if path.name == "attack_metrics.npz":
                continue
            with np.load(path, allow_pickle=False) as data:
                if "attributions" in data:
                    methods.add(path.stem)
    if not methods:
        raise ValueError(f"{runs[0].parent}: no saved attribution arrays found")
    arrays = {}
    heads = 34 if dataset == "ascadv2" else 16
    for method in sorted(methods):
        arrays[method] = []
        for run in runs:
            path = run / f"{method}.npz"
            if not path.exists():
                raise ValueError(f"{run}: missing saved attribution {method}")
            with np.load(path, allow_pickle=False) as data:
                array = data["attributions"]
            if (
                array.ndim != 2
                or array.shape[0] != heads
                or not np.isfinite(array).all()
            ):
                raise ValueError(
                    f"{path}: require finite saved [{heads},samples] raw-head attributions"
                )
            arrays[method].append(array)
    figures = {}
    for method, seed_arrays in arrays.items():
        for group, indices in head_groups(dataset).items():
            fig, ax = plt.subplots(figsize=(10, 3))
            for seed, array in enumerate(seed_arrays):
                values = array[indices]
                x = np.arange(values.shape[1])
                ax.plot(
                    x,
                    values.mean(axis=0),
                    color=f"C{seed}",
                    label=f"seed={seed}",
                    rasterized=True,
                )
                ax.fill_between(
                    x,
                    values.min(axis=0),
                    values.max(axis=0),
                    rasterized=True,
                    color=f"C{seed}",
                    alpha=0.12,
                )
            ax.set(
                xlabel="Sample index",
                ylabel="Saved attribution (head mean; band min–max)",
                title=f"{dataset}: {method}, {group}",
            )
            ax.legend(fontsize=6)
            fig.tight_layout()
            figures[f"attributions-{method}-{group}"] = fig
    snr_dir = runs[0].parent / "snr"
    for path in sorted(snr_dir.glob("*.npy")):
        array = np.load(path, allow_pickle=False)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError(f"{path}: expected finite SNR [outputs,samples]")
        fig, ax = plt.subplots(figsize=(10, 3))
        for index, values in enumerate(array):
            ax.plot(values, linewidth=0.5, label=f"output {index}", rasterized=True)
        ax.set(
            xlabel="Sample index",
            ylabel="Saved SNR",
            title=f"{dataset}: {path.stem} (separate SNR outputs)",
        )
        ax.legend(fontsize=5, ncol=4)
        fig.tight_layout()
        figures[f"snr-{path.stem}"] = fig
    return figures


def save_figures(figures, dest, overwrite):
    dest.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        path = dest / f"{name}.pdf"
        if path.exists() and not overwrite:
            warnings.warn(f"Skipping existing {path}; use --overwrite", stacklevel=2)
        else:
            figure.savefig(path, bbox_inches="tight")
            print(f"saved {path}")
        plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Artifact-only five-seed reporting; never loads models or traces."
    )
    for dataset in DATASETS:
        parser.add_argument(f"--{dataset}", type=Path)
    parser.add_argument("--plots", nargs="+", choices=PLOTS, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    selected = [
        (dataset, getattr(args, dataset.replace("-", "_")))
        for dataset in DATASETS
        if getattr(args, dataset.replace("-", "_")) is not None
    ]
    if not selected:
        parser.error("provide at least one dataset results root")
    for dataset, root in selected:
        runs = seed_runs(root)
        dest = args.dest / dataset
        for plot in dict.fromkeys(args.plots):
            if plot == "mtd-table":
                mtd_table(runs, dataset, dest, args.overwrite)
            elif plot == "accuracy-table":
                accuracy_table(runs, dataset, dest, args.overwrite)
            elif plot == "learning-curves":
                save_figures(learning_figures(runs, dataset), dest, args.overwrite)
            elif plot in ("attack-curves", "byte-curves"):
                save_figures(
                    attack_figures(runs, dataset, per_byte=plot == "byte-curves"),
                    dest,
                    args.overwrite,
                )
            elif plot == "attributions":
                save_figures(attribution_figures(runs, dataset), dest, args.overwrite)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(f"error: {error}") from error
