"""Generate per-plot PDF files for a trained-model run directory.

One PDF is produced per plot type.  Missing files are silently skipped.


Usage:
    python experiments/plot_performance.py \
        --run-dir ./outputs/ascadv1_fixed/strong_attacker/seed_0


"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.visualize_runs import (
    plot_leakiness_over_time,
    plot_per_byte_bar,
    plot_rank_trajectories,
    plot_wb_comparison_grid,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _save(fig, dest: Path, name: str):
    path = dest / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ── per-plot generators ───────────────────────────────────────────────────────


def gen_training_curves(run_dir: Path, dest: Path):
    csv = run_dir / "metrics.csv"
    if not csv.exists():
        return
    m = pandas.read_csv(csv)
    train_mask = ~m["train/loss"].isna()
    val_mask = ~m["val/loss"].isna()
    t_step = m["step"][train_mask].values
    v_step = m["step"][val_mask].values

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    ax = axes[0]
    ax.plot(
        t_step,
        m["train/loss"][train_mask].values,
        color="royalblue",
        linestyle=":",
        linewidth=0.8,
        label="train",
        rasterized=True,
    )
    ax.plot(
        v_step,
        m["val/loss"][val_mask].values,
        color="royalblue",
        linestyle="-",
        linewidth=1.2,
        label="val",
        rasterized=True,
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend(fontsize=7)

    ax = axes[1]
    byte_cols = [c for c in m.columns if re.fullmatch(r"val/rank/\d+", c)]
    for col in byte_cols:
        ax.plot(
            v_step,
            m[col][val_mask].values,
            color="royalblue",
            linewidth=0.2,
            alpha=0.4,
            rasterized=True,
        )
    for col_suffix, ls, lw, label in [
        ("val/rank_min", "-", 1.0, "best byte"),
        ("val/rank_med", "--", 1.0, "median byte"),
        ("val/rank_max", ":", 1.0, "worst byte"),
    ]:
        if col_suffix in m.columns:
            ax.plot(
                v_step,
                m[col_suffix][val_mask].values,
                color="royalblue",
                linestyle=ls,
                linewidth=lw,
                label=label,
                rasterized=True,
            )
    ax.set_xlabel("Step")
    ax.set_ylabel("Rank")
    ax.set_title("Val Rank")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(
        t_step,
        m["train/acc"][train_mask].values,
        color="royalblue",
        linestyle=":",
        linewidth=0.8,
        label="train",
        rasterized=True,
    )
    ax.plot(
        v_step,
        m["val/acc"][val_mask].values,
        color="royalblue",
        linestyle="-",
        linewidth=1.2,
        label="val",
        rasterized=True,
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend(fontsize=7)

    fig.tight_layout()
    _save(fig, dest, "training_curves")


def gen_attack_performance(run_dir: Path, dest: Path):
    npz = run_dir / "attack_metrics.npz"
    if not npz.exists():
        return
    d = np.load(npz, allow_pickle=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    plot_rank_trajectories(d["guessing_entropy"], axes[0], label="worst-case byte")
    axes[0].set_title("Attack rank over traces")
    axes[0].legend(fontsize=7)
    plot_per_byte_bar(d["per_byte_mtd"], axes[1])
    axes[1].set_ylabel("MTD (traces)")
    axes[1].set_title("Per-byte MTD")
    fig.tight_layout()
    _save(fig, dest, "attack_performance")


def gen_leakiness(run_dir: Path, dest: Path, method: str):
    attr_path = run_dir / f"{method}.npz"
    if not attr_path.exists():
        return
    with np.load(attr_path, allow_pickle=False) as data:
        attr = data["attributions"]
    fig, ax = plt.subplots(figsize=(14, 3))
    plot_leakiness_over_time(attr, ax, title=method.replace("_", " "))
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, dest, f"leakiness_{method}")


def gen_wb_comparison_grid(run_dir: Path, dest: Path, method: str, oracle: np.ndarray):
    attr_path = run_dir / f"{method}.npz"
    if not attr_path.exists():
        return
    with np.load(attr_path, allow_pickle=False) as data:
        attr = data["attributions"]
    byte_count = attr.shape[0]
    fig, axes = plt.subplots(2, byte_count, figsize=(byte_count * 2, 4))
    plot_wb_comparison_grid(attr, oracle, axes)
    fig.suptitle(
        f"Per-byte leakiness — {method.replace('_', ' ')} (top) vs. oracle (bottom)",
        y=1.01,
    )
    fig.tight_layout()
    _save(fig, dest, f"wb_comparison_{method}")


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Plot saved training and attack metrics; no model or dataset is loaded."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Directory containing metrics.csv and/or attack_metrics.npz; missing files are skipped.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Output directory. Defaults to --run-dir.",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    assert run_dir.exists(), f"run-dir not found: {run_dir}"
    dest: Path = args.dest or run_dir
    dest.mkdir(parents=True, exist_ok=True)

    gen_training_curves(run_dir, dest)
    gen_attack_performance(run_dir, dest)


if __name__ == "__main__":
    main()
