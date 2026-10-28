from pathlib import Path
from typing import Optional, Dict, Any

import pandas
from matplotlib.axes import Axes
import numpy as np


from uncropped_transformers.datasets.ascadv1 import repr_target as ascadv1_repr_target


def plot_ascadv1_oracle_leakiness(
    snr_dir: Path,
    ax: Axes,
    byte: int = 2,
    markers: bool = True,
    arb_byte: bool = False,
):
    marker_kwargs = dict(marker=".", markersize=1) if markers else {}
    byte_label = "i" if arb_byte else f"{byte}"

    # Bytes 0 and 1 have no masking — use subbytes SNR directly as a single curve.
    if byte in (0, 1):
        snr = np.load(snr_dir / "subbytes.attack.npy")[byte, :]
        kwargs = dict(
            color="grey",
            linestyle="-",
            label=ascadv1_repr_target("subbytes", byte=byte_label),
        )
        ax.plot(snr, rasterized=True, linewidth=0.3, **marker_kwargs, **kwargs)
        return dict(composite=snr, subbytes=snr)

    int_var_snrs = dict(
        prin=np.load(snr_dir / "p__xor__k__xor__r_in.attack.npy")[byte, :],
        pr=np.load(snr_dir / "p__xor__k__xor__r.profile.npy")[byte, :],
        rin=np.load(snr_dir / "r_in.attack.npy")[0, :],
        rout=np.load(snr_dir / "r_out.attack.npy")[0, :],
        r=np.load(snr_dir / "r.attack.npy")[byte, :],
        yrout=np.load(snr_dir / "subbytes__xor__r_out.attack.npy")[byte, :],
        yr=np.load(snr_dir / "subbytes__xor__r.attack.npy")[byte, :],
    )
    int_var_kwargs = dict(
        prin=dict(
            color="red",
            linestyle="--",
            label=ascadv1_repr_target("p__xor__k__xor__r_in", byte=byte_label),
        ),
        pr=dict(
            color="green",
            linestyle="--",
            label=ascadv1_repr_target("p__xor__k__xor__r", byte=byte_label),
        ),
        rin=dict(
            color="purple",
            linestyle="-",
            label=ascadv1_repr_target("r_in", byte=byte_label),
        ),
        rout=dict(
            color="teal",
            linestyle="-",
            label=ascadv1_repr_target("r_out", byte=byte_label),
        ),
        r=dict(
            color="orange",
            linestyle="-",
            label=ascadv1_repr_target("r", byte=byte_label),
        ),
        yrout=dict(
            color="magenta",
            linestyle="-",
            label=ascadv1_repr_target("subbytes__xor__r_out", byte=byte_label),
        ),
        yr=dict(
            color="black",
            linestyle="-",
            label=ascadv1_repr_target("subbytes__xor__r", byte=byte_label),
        ),
    )
    for int_var_name in int_var_snrs.keys():
        int_var_snr = int_var_snrs[int_var_name]
        if int_var_snr is None:
            continue
        kwargs = int_var_kwargs[int_var_name]
        ax.plot(int_var_snr, rasterized=True, linewidth=0.3, **marker_kwargs, **kwargs)
    white_box_composite = np.stack(list(int_var_snrs.values())).mean(axis=0)
    return dict(composite=white_box_composite, **int_var_snrs)


def plot_leakiness_over_time(
    attr: np.ndarray,
    ax: Axes,
    title: Optional[str] = None,
    per_byte_alpha: float = 0.35,
    per_byte_lw: float = 0.4,
    sum_lw: float = 1.2,
    sum_color: str = "black",
    byte_color: str = "royalblue",
):
    """Line plot of attribution values over time.

    Draws one thin faded line per byte and a thicker line for the per-timestep
    sum across bytes.

    Args:
        attr: shape (byte_count, feature_count)
    """
    byte_count, feature_count = attr.shape
    timesteps = np.arange(feature_count)
    for byte_idx in range(byte_count):
        ax.plot(
            timesteps,
            attr[byte_idx],
            color=byte_color,
            linewidth=per_byte_lw,
            alpha=per_byte_alpha,
            rasterized=True,
        )
    ax.plot(
        timesteps,
        attr.sum(axis=0),
        color=sum_color,
        linewidth=sum_lw,
        label="sum",
        rasterized=True,
    )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Estimated leakiness")
    if title is not None:
        ax.set_title(title)


def plot_wb_comparison_grid(
    attr: np.ndarray,
    oracle: np.ndarray,
    axes,
):
    """2×byte_count grid: top row = estimated per byte, bottom row = oracle per byte.

    Args:
        attr:   (byte_count, feature_count)
        oracle: (byte_count, feature_count)
        axes:   array of Axes with shape (2, byte_count)
    """
    byte_count, feature_count = attr.shape
    timesteps = np.arange(feature_count)
    for byte_idx in range(byte_count):
        ax_est = axes[0, byte_idx]
        ax_ora = axes[1, byte_idx]
        ax_est.plot(
            timesteps, attr[byte_idx], color="royalblue", linewidth=0.4, rasterized=True
        )
        ax_est.set_title(f"byte {byte_idx}", fontsize=6)
        ax_est.tick_params(labelsize=5)
        if byte_idx == 0:
            ax_est.set_ylabel("Estimated", fontsize=6)
        ax_ora.plot(
            timesteps,
            oracle[byte_idx],
            color="darkorange",
            linewidth=0.4,
            rasterized=True,
        )
        ax_ora.tick_params(labelsize=5)
        if byte_idx == 0:
            ax_ora.set_ylabel("Oracle", fontsize=6)
        ax_ora.set_xlabel("Timestep", fontsize=6)


def plot_rank_trajectories(
    rank_over_time: np.ndarray,
    ax: Axes,
    color: str = "blue",
    label: Optional[str] = None,
    per_byte_alpha: float = 0.3,
    worst_case_alpha: float = 0.9,
):
    """Plot per-byte rank trajectories plus the worst-case (max) envelope.

    Args:
        rank_over_time: shape (byte_count, trace_count)
    """
    byte_count, trace_count = rank_over_time.shape
    traces_seen = np.arange(1, trace_count + 1)
    for byte_idx in range(byte_count):
        ax.plot(
            traces_seen,
            rank_over_time[byte_idx],
            color=color,
            linewidth=0.3,
            alpha=per_byte_alpha,
            rasterized=True,
        )
    ax.plot(
        traces_seen,
        rank_over_time.max(axis=0),
        color=color,
        linewidth=1.2,
        alpha=worst_case_alpha,
        label=label,
        rasterized=True,
    )
    ax.set_xlabel("Traces seen")
    ax.set_ylabel("Rank")


def plot_per_byte_bar(
    values: np.ndarray,
    ax: Axes,
    color: str = "steelblue",
    label: Optional[str] = None,
    **bar_kwargs,
):
    """Bar chart of a (byte_count,) array, one bar per byte."""
    byte_count = len(values)
    ax.bar(np.arange(byte_count), values, color=color, label=label, **bar_kwargs)
    ax.set_xlabel("Byte index")
    ax.set_xticks(np.arange(byte_count))


def plot_mtd(
    run_path: Path,
    ax: Axes,
    worst_byte_kwargs: Optional[Dict[str, Any]] = None,
    other_byte_kwargs: Optional[Dict[str, Any]] = None,
    **common_kwargs,
):
    worst_byte_kwargs = worst_byte_kwargs or dict()
    other_byte_kwargs = other_byte_kwargs or dict()
    attack_metrics = np.load(run_path / "attack_metrics.npz", allow_pickle=False)
    rank_over_time = attack_metrics["guessing_entropy"]
    byte_count = rank_over_time.shape[0]
    traces_seen = np.arange(1, rank_over_time.shape[1] + 1)
    _worst_byte_kwargs = dict(
        linewidth=2,
    )
    _worst_byte_kwargs.update(common_kwargs)
    _worst_byte_kwargs.update(worst_byte_kwargs)
    _other_byte_kwargs = dict(linewidth=0.5, alpha=0.5)
    _other_byte_kwargs.update(common_kwargs)
    _other_byte_kwargs.update(other_byte_kwargs)
    ax.plot(traces_seen, rank_over_time.max(axis=0), **_worst_byte_kwargs)
    for byte_idx in range(byte_count):
        ax.plot(traces_seen, rank_over_time[byte_idx, :], **_other_byte_kwargs)


def plot_training_curves(
    run_path: Path,
    ax: Axes,
    metric_key: str,
    train_plot_kwargs: Optional[Dict[str, Any]] = None,
    val_plot_kwargs: Optional[Dict[str, Any]] = None,
    **common_plot_kwargs,
):
    train_plot_kwargs = train_plot_kwargs or dict()
    val_plot_kwargs = val_plot_kwargs or dict()
    metrics = pandas.read_csv(run_path / "metrics.csv")
    train_mask = ~metrics["train/loss"].isna()
    val_mask = ~metrics["val/loss"].isna()
    train_steps = metrics["step"][train_mask]
    val_steps = metrics["step"][val_mask]
    train_metric = metrics[f"train/{metric_key}"][train_mask]
    val_metric = metrics[f"val/{metric_key}"][val_mask]
    _train_plot_kwargs = dict(
        color="blue", linestyle=":", label="train", rasterized=True
    )
    _train_plot_kwargs.update(common_plot_kwargs)
    _train_plot_kwargs.update(train_plot_kwargs)
    _val_plot_kwargs = dict(color="blue", linestyle="-", label="val", rasterized=True)
    _val_plot_kwargs.update(common_plot_kwargs)
    _val_plot_kwargs.update(val_plot_kwargs)
    ax.plot(train_steps, train_metric, **_train_plot_kwargs)
    ax.plot(val_steps, val_metric, **_val_plot_kwargs)
