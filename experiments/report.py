"""Reproduce the ablation figure and table from saved native run outputs.

Run from code-cleanup: python experiments/report.py
Requires NumPy, Matplotlib, and LaTeX/dvipng for manuscript-matched fonts.
No checkpoints, datasets, network, or ML libraries.
Missing files, missing NPZ fields, or unexpected result shapes raise exceptions.
"""
import csv
import json
from pathlib import Path
from typing import Literal

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
ROOT = OUTPUTS / "ablations"
FACTORS = [1, 2, 5, 10, 25, 50, 125, 250, 625, 1250]
DATASETS = ["ascadv1r", "ches-ctf-2018"]
METHODS = ["mean", "max", "stride"]


def disclosure_times(path, outputs):
    with np.load(path, allow_pickle=False) as result:
        ranks = result["ranks_zero_based"]
        metadata = json.loads(str(result["metadata"]))
    assert ranks.shape == (1000, 1000, outputs), str(path)
    assert metadata["attack_count"] == metadata["horizon"] == 1000, str(path)
    assert metadata["mtd_censor_sentinel"] == 1001, str(path)
    wrong = ranks > 0
    # First trace after the final incorrect rank: sustained recovery, not first hit.
    times = 1001 - np.argmax(wrong[:, ::-1, :], axis=1)
    times[~wrong.any(axis=1)] = 1
    return times, metadata


# Downsampling: cap each attack-order/byte time BEFORE averaging.
# Factor 1 is one shared raw baseline per dataset/seed, stored under pool=mean.
sweep_values = {}
for dataset in DATASETS:
    for method in METHODS:
        factors = FACTORS if method == "mean" else FACTORS[1:]
        values = []
        for factor in factors:
            seeds = []
            for seed in range(3):
                path = ROOT / "downsampling-sweeps" / dataset / f"factor={factor}" / f"pool={method}" / f"seed={seed}" / "attack_metrics.npz"
                times, metadata = disclosure_times(path, 16)
                assert metadata["identity"]["settings"]["targets"]["target_byte"] == list(range(16)), str(path)
                seeds.append(float(np.minimum(times, 1000).mean()))
            values.append(seeds)
        sweep_values[dataset, method] = np.array(values)

# Target ablations: byte2 for ASCADv1r; physical masked-SubBytes slot0 for ASCADv2.
# Keep sentinel-based lower bounds for the table, unlike the capped plot statistic.
variants = ["reference", "single-byte", "cropped-single-byte"]
labels = ["All targets, uncropped", "Single target, uncropped", "Single target, cropped"]
table_rows = []
for variant, label in zip(variants, labels, strict=True):
    mtds, accuracies, censored = [], [], 0
    for seed in range(3):
        path = ROOT / "target-ablations" / "ascadv1r" / variant / f"seed={seed}" / "attack_metrics.npz"
        times, metadata = disclosure_times(path, 16 if variant == "reference" else 1)
        targets = metadata["identity"]["settings"]["targets"]["target_byte"]
        assert targets == (list(range(16)) if variant == "reference" else [2]), str(path)
        times = times[:, targets.index(2)]
        mtds.append(float(times.mean()))
        censored += int((times > 1000).sum())
        path = ROOT / "target-ablations" / "ascadv2" / variant / f"seed={seed}" / "classification_metrics.npz"
        with np.load(path, allow_pickle=False) as result:
            rates = result["success_rate"]
            assert int(result["trace_count"]) == 399998, str(path)
            metadata = json.loads(str(result["metadata"]))
        assert rates.shape == ((34,) if variant == "reference" else (1,)), str(path)
        if variant == "reference":
            assert metadata["physical_slot0_output_index"] == 2, str(path)
            assert "micro" in metadata["protocol"], str(path)
        else:
            assert metadata["target_bytes"] == [0], str(path)
            assert metadata["target_variables"] == ["masked_subbytes"], str(path)
        accuracies.append(100 * float(rates[2 if variant == "reference" else 0]))
    mtd_mean, mtd_sd = float(np.mean(mtds)), float(np.std(mtds, ddof=1))
    accuracy_mean, accuracy_sd = float(np.mean(accuracies)), float(np.std(accuracies, ddof=1))
    mtd_text = f"≥{int(np.floor(mtd_mean)):,}" if censored else f"{mtd_mean:.2f} ± {mtd_sd:.2f}"
    table_rows.append(dict(ablation=label, ascadv1r_mtd=mtd_text, ascadv2_accuracy=f"{accuracy_mean:.3f} ± {accuracy_sd:.3f}", mtd_seeds=mtds, accuracy_seeds=accuracies, censored_orders=censored))

# Selected paper figure: dataset colors, strategy markers/lines, no error bars.
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": True, "axes.spines.right": True,
    "font.family": "serif", "font.serif": ["Computer Modern Roman"],
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\renewcommand{\rmdefault}{ptm}",
})
colors = ["#0072B2", "#D55E00"]
styles: list[tuple[str, Literal["-", "--", ":"]]] = [("o", "-"), ("s", "--"), ("^", ":")]
fig, ax = plt.subplots(figsize=(3.5, 2))
for d, dataset in enumerate(DATASETS):
    for m, method in enumerate(METHODS):
        factors = FACTORS if method == "mean" else FACTORS[1:]
        marker, line = styles[m]
        ax.plot(factors, sweep_values[dataset, method].mean(axis=1), color=colors[d], marker=marker, linestyle=line, markersize=3, linewidth=1.3)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_yticks([1, 10, 100, 1000], ["1", "10", "100", r"$\geq 1\mathrm{k}$"])
ax.set_ylim(.8, 1600)
ax.set_xlabel("Downsampling factor", labelpad=1)
ax.set_ylabel("MTD (mean per-byte)", labelpad=1)
ax.grid(False, which="both")
handles = [Line2D([], [], color=c, lw=1.3, label=n) for n, c in zip(["ASCADv1r", "CC'18"], colors, strict=True)]
handles += [Line2D([], [], color="0.35", marker=marker, markersize=3, linewidth=1.3, linestyle=line, label=label) for (marker, line), label in zip(styles, ["Mean", "Max", "Stride"], strict=True)]
# One row: short dataset color swatches, longer method line samples.
dataset_legend = fig.legend(handles=handles[:2], loc="upper left", bbox_to_anchor=(0, .99),
           frameon=False, ncol=2, handlelength=.65, columnspacing=.6,
           handletextpad=.3, borderaxespad=0, borderpad=0)
method_legend = fig.legend(handles=handles[2:], loc="upper left", bbox_to_anchor=(0, .99),
           frameon=False, ncol=3, handlelength=2.1, columnspacing=.6,
           handletextpad=.3, borderaxespad=0, borderpad=0)
ax.tick_params(axis="both", which="major", pad=2)
# Center both groups with the same inter-entry gap used within each group.
fig.canvas.draw()
dataset_width = dataset_legend.get_window_extent().width / fig.bbox.width
method_width = method_legend.get_window_extent().width / fig.bbox.width
gap = .6 * plt.rcParams["legend.fontsize"] / 72 / fig.get_figwidth()
left = (1 - dataset_width - method_width - gap) / 2
dataset_legend.set_bbox_to_anchor((left, .99))
method_legend.set_bbox_to_anchor((left + dataset_width + gap, .99))
legend_height = max(legend.get_window_extent().height for legend in
                    (dataset_legend, method_legend)) / fig.bbox.height
fig.subplots_adjust(left=.16, right=.985, bottom=.21,
                    top=.99 - legend_height - 3 / 72 / fig.get_figheight())

# All required inputs have been read successfully before writing any outputs.
figures = OUTPUTS / "figures"
output = OUTPUTS / "tables"
figures.mkdir(exist_ok=True, parents=True)
output.mkdir(exist_ok=True, parents=True)
fig.savefig(figures / "downsampling.png", dpi=300)
fig.savefig(figures / "downsampling.pdf")
plt.close(fig)
header = ["Supervision and input", "ASCADv1r: byte2 MTD ↓", "ASCADv2: slot0 accuracy (%) ↑"]
lines = ["| " + " | ".join(header) + " |", "|---|---:|---:|"]
with (output / "ablation-table.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for row in table_rows:
        cells = [row["ablation"], row["ascadv1r_mtd"], row["ascadv2_accuracy"]]
        writer.writerow(cells)
        lines.append("| " + " | ".join(cells) + " |")
lines += ["", "Mean ± sample SD across seeds 0–2. ASCADv1r reports sustained byte2 MTD over 1,000 attack orders, horizon 1,000; ≥ entries are lower bounds, not observed times. ASCADv2 reports full-attack-set micro accuracy on physical masked-SubBytes slot0; chance is 0.390625%."]
lines += ["", "Downsampling figure: means over 16 bytes and 1,000 attack orders within each seed, then three seeds; no error bars. Nonrecoveries contribute 1,000. The ≥1k tick denotes the censoring cap, not measured recovery. Raw (1×) is the shared baseline, shown only on the mean curves. All methods and datasets are plotted at their actual downsampling factors, without horizontal offsets."]
(output / "ablation-table.md").write_text("\n".join(lines) + "\n")
tex = [r"\begin{tabular}{lcc}", r"\hline\noalign{\vskip 3pt}",
       r"Supervision and input & \shortstack{ASCADv1r\\MTD $\downarrow$} & \shortstack{ASCADv2\\Accuracy (\%) $\uparrow$} \\[3pt]",
       r"\hline\noalign{\vskip 3pt}"]
for row in table_rows:
    cells = [row[k].replace("±", r"\pm").replace("≥", r"\geq").replace(",", "{,}") for k in ["ascadv1r_mtd", "ascadv2_accuracy"]]
    tex.append(row["ablation"] + " & $" + cells[0] + "$ & $" + cells[1] + r"$ \\[3pt]")
tex += [r"\hline", r"\end{tabular}"]
(output / "ablation-table.tex").write_text("\n".join(tex) + "\n")
print("\n".join(lines))
print(f"\nRead 168 sweep runs and 18 target-ablation runs. Outputs: {figures} and {output}")
