"""Generate the paper's cross-dataset performance table from saved key ranks.

Run from code-cleanup: python experiments/plot_results.py
Writes performance-summary.{tex,csv,json} under outputs/tables by default.
The LaTeX file is a tabular fragment; the manuscript owns its caption and float.
No checkpoints, raw traces, model inference, or training are needed.
Use --cost-ledger PATH for a provisional cost table from a saved native-cost ledger.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from utils.artifacts import seed_runs, sha256_file


DATASETS = {
    "ascadv1-fixed": "ASCADv1f",
    "ascadv1-variable": "ASCADv1r",
    "ches-ctf-2018": "CC'18",
    "eshards-non-shuffled": "eShards",
    "ascadv2": "ASCADv2",
}
CAPTION = (
    "Joint full-key and per-byte attack performance. All entries are mean "
    "and sample standard deviation across five training seeds. Worst-byte and "
    "mean-byte metrics are computed within each seed before summarizing seeds; "
    "the worst byte may differ between seeds. "
    "Single-trace success rate (SR$_1$) is estimated from the first trace of each "
    "saved randomized attack, not from all attack-set traces. MTD requires the "
    "correct candidate to remain ranked first through the evaluation horizon. "
    "All included attack orders recover within that horizon. "
    "ASCADv1f/r denote fixed/random-key ASCADv1; CC'18 denotes CHES-CTF-2018. "
    "eShards is non-shuffled; ASCADv2 uses simulated-fixed-key evaluation."
)


def summarize_run(ranks):
    """Return joint/per-byte SR1 (%) and sustained MTD; reject nonrecovery."""
    if (
        ranks.ndim != 3
        or not all(ranks.shape)
        or ranks.shape[2] != 16
        or not np.issubdtype(ranks.dtype, np.integer)
        or np.any(ranks < 0)
        or np.any(ranks > 255)
    ):
        raise ValueError("Expected integer zero-based ranks [attacks, traces, 16] in 0..255")
    horizon = ranks.shape[1]
    times = np.where(
        ranks != 0, np.arange(1, horizon + 1)[None, :, None], 0
    ).max(axis=1) + 1
    censored = int((times > horizon).any(axis=1).sum())
    if censored:
        raise ValueError(
            f"{censored} attack orders did not recover by {horizon} traces; "
            "report these separately, not as uncensored mean MTD"
        )
    first_correct = ranks[:, 0, :] == 0
    return {
        "attack_count": ranks.shape[0],
        "horizon": horizon,
        "sr1_percent": {
            "full_key": float(first_correct.all(axis=1).mean() * 100),
            "per_byte": (first_correct.mean(axis=0) * 100).tolist(),
        },
        "mtd": {
            "full_key": float(times.max(axis=1).mean()),
            "per_byte": times.mean(axis=0).tolist(),
        },
    }


def summarize_dataset(root, dataset):
    runs = seed_runs(root)
    details = []
    for run in runs:
        path = run / "attack_metrics.npz"
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(saved["metadata"].item())
            if metadata["identity"]["dataset"] != dataset:
                raise ValueError(f"{path}: dataset identity mismatch")
            try:
                summary = summarize_run(saved["ranks_zero_based"])
            except ValueError as error:
                raise ValueError(f"{path}: {error}") from error
        for field in ("attack_count", "horizon"):
            if metadata[field] != summary[field]:
                raise ValueError(f"{path}: {field} disagrees with rank array")
        details.append({
            "seed": int(run.name.split("=")[1]),
            "artifact": str(path.resolve()),
            "sha256": sha256_file(path),
            "metadata": metadata,
            **summary,
        })
    if len({(run["attack_count"], run["horizon"]) for run in details}) != 1:
        raise ValueError(f"{root}: inconsistent evaluation sizes across seeds")
    metrics = {}
    for metric in ("sr1_percent", "mtd"):
        full = np.array([run[metric]["full_key"] for run in details])
        per_byte = np.array([run[metric]["per_byte"] for run in details])
        worst = per_byte.min(axis=1) if metric == "sr1_percent" else per_byte.max(axis=1)
        metrics[metric] = {}
        for name, values in (
            ("full_key", full),
            ("worst_byte", worst),
            ("mean_byte", per_byte.mean(axis=1)),
        ):
            metrics[metric][f"{name}_mean"] = float(values.mean())
            metrics[metric][f"{name}_sample_sd"] = float(values.std(ddof=1))
            metrics[metric][f"{name}_seed_values"] = values.tolist()
    return {"dataset": dataset, "metrics": metrics, "runs": details}


def table_rows(results):
    rows = []
    for result in results:
        dataset = result["dataset"]
        metric = "sr1_percent" if dataset.startswith("ascadv1") else "mtd"
        summary = result["metrics"][metric]
        rows.append({
            "dataset": dataset,
            "metric": metric,
            "full_key_mean": summary["full_key_mean"],
            "full_key_sample_sd": summary["full_key_sample_sd"],
            "worst_byte_mean": summary["worst_byte_mean"],
            "worst_byte_sample_sd": summary["worst_byte_sample_sd"],
            "mean_byte_mean": summary["mean_byte_mean"],
            "mean_byte_sample_sd": summary["mean_byte_sample_sd"],
            "seed_count": len(result["runs"]),
            "attack_count": result["runs"][0]["attack_count"],
            "horizon": result["runs"][0]["horizon"],
        })
    return rows


def latex_table(rows):
    lines = [
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Dataset & Full key & Worst byte & Mean byte \\",
        r"\hline",
    ]
    for metric, heading in (
        ("sr1_percent", r"Single-trace success rate (\%) $\uparrow$"),
        ("mtd", r"Minimum traces to disclosure (MTD) $\downarrow$"),
    ):
        group = [row for row in rows if row["metric"] == metric]
        if not group:
            continue
        lines.append(r"\multicolumn{4}{l}{\textit{" + heading + r"}} \\")
        for row in group:
            lines.append(
                f"{DATASETS[row['dataset']]} & "
                f"${row['full_key_mean']:.1f} \\pm {row['full_key_sample_sd']:.1f}$ & "
                f"${row['worst_byte_mean']:.1f} \\pm {row['worst_byte_sample_sd']:.1f}$ & "
                f"${row['mean_byte_mean']:.1f} \\pm {row['mean_byte_sample_sd']:.1f}$ " + r"\\"
            )
        lines.append(r"\noalign{\vskip 2pt}" if metric == "sr1_percent" else r"\noalign{\vskip 3pt}")
    lines += [
        r"\hline", r"\end{tabular}",

    ]
    return "\n".join(lines) + "\n"


def write_cost_draft(args):
    """Render provisional native costs with author-supplied warm reserved memory."""
    paths = {ext: args.dest / f"cost-summary-draft.{ext}" for ext in ("tex", "csv", "json")}
    if not args.overwrite and any(path.exists() for path in paths.values()):
        raise ValueError("cost output bundle already exists; use --overwrite")
    ledger = json.loads(args.cost_ledger.read_text())
    saved = {row["dataset"].replace("_", "-"): row for row in ledger["rows"]}
    warm_reserved_gib = {
        "ascadv1-fixed": 5.69, "ascadv1-variable": 6.95, "ches-ctf-2018": 4.47,
        "eshards-non-shuffled": 0.13, "ascadv2": 25.95,
    }
    rows = []
    lines = [r"\begin{tabular}{lcccc}", r"\hline\noalign{\vskip 3pt}",
             r"Dataset & Params & FLOPs/step & Est. time & VRAM \\[3pt]",
             r"\hline\noalign{\vskip 3pt}"]
    for dataset in dict.fromkeys(args.datasets):
        source = saved[dataset]
        row = dict(dataset=dataset, parameters=source["param_count"],
                   flops_per_step=source["flops_per_optimizer_step"],
                   estimated_hours=source["estimated_full_recipe_hours"],
                   warm_peak_reserved_gib=warm_reserved_gib[dataset],
                   warm_peak_reserved_gb=warm_reserved_gib[dataset] * 2**30 / 10**9)
        rows.append(row)
        name = DATASETS[dataset] + (r"$^*$" if dataset == "ascadv2" else "")
        lines.append(f"{name} & {row['parameters']/1e6:.2f}M & "
                     f"{row['flops_per_step']/1e12:.2f}T & {row['estimated_hours']:.2f}\\,h & "
                     f"{row['warm_peak_reserved_gb']:.2f}\\,GB " + r"\\[3pt]")
    lines += [r"\hline", r"\end{tabular}"]
    args.dest.mkdir(parents=True, exist_ok=True)
    paths["tex"].write_text("\n".join(lines) + "\n")
    with paths["csv"].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paths["json"].write_text(json.dumps(dict(
        status="Provisional historical native benchmarks, not verified final-recipe costs",
        memory_status="Warm peak PyTorch reserved GiB supplied by author in chat, converted to decimal GB (GiB * 2**30 / 10**9) for display; not an NVML measurement",
        source=str(args.cost_ledger.resolve()), source_sha256=sha256_file(args.cost_ledger),
        generator_sha256=sha256_file(Path(__file__)), table_rows=rows, source_ledger=ledger,
    ), indent=2, allow_nan=False) + "\n")
    for path in paths.values():
        print(f"saved {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("outputs"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--dest", type=Path, default=Path("outputs/tables"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cost-ledger", type=Path, help="Saved native-cost ledger; generate a cost draft instead of performance")
    args = parser.parse_args(argv)
    if args.cost_ledger is not None:
        write_cost_draft(args)
        return
    paths = {suffix: args.dest / f"performance-summary.{suffix}" for suffix in ("tex", "csv", "json")}
    if not args.overwrite and any(path.exists() for path in paths.values()):
        parser.error("output bundle already exists; use --overwrite")
    results = [
        summarize_dataset(args.results_root / dataset, dataset)
        for dataset in dict.fromkeys(args.datasets)
    ]
    rows = table_rows(results)
    args.dest.mkdir(parents=True, exist_ok=True)
    paths["tex"].write_text(latex_table(rows))
    with paths["csv"].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paths["json"].write_text(json.dumps({
        "caption": CAPTION,
        "generator_sha256": sha256_file(Path(__file__)),
        "table_rows": rows,
        "datasets": results,
    }, indent=2, allow_nan=False) + "\n")
    for path in paths.values():
        print(f"saved {path}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(f"error: {error}") from error