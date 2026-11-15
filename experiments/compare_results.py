"""Compare saved attack ranks with audited published results (no inference).

Run: python experiments/compare_results.py
Outputs prior-work-comparison.{tex,csv,json,md} in outputs/tables.
LaTeX is a tabular fragment. Read the companion notes before publishing:
several source conventions cannot be exactly harmonized from released evidence.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from plot_results import DATASETS
from utils.artifacts import seed_runs, sha256_file


# Transcribed from published tables, not silently corrected from released code.
# All arrays list bytes 0..15, including ASCAD's two unmasked bytes.
PUBLISHED_BYTES = {
    "efss-fixed-cnn": [1, 1, 1, ">3000", 2, ">3000", ">3000", ">3000", ">3000", ">3000", ">3000", 1, ">3000", ">3000", 3, ">3000"],
    "efss-fixed-mlp": [1, 1, 4, 3, 3, 2, 3, 3, 7, 4, 4, 2, 14, 20, 2, 7],
    "efss-variable-cnn": [1, 1, 2, 10, 1, 1, 5, 7, 4, 1, 3, 1, 19, 7, 1, 3],
    "efss-variable-mlp": [1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 1, 2],
    "efss-ches-cnn": [">3000", 787, 374, 411, 2518, 1788, 1115, 747, 1142, 1594, 619, 1027, 980, 1109, 317, 555],
    "efss-ches-mlp": [288, 227, 8, 7, 182, 190, 141, 7, 281, 13, 7, 178, 247, 123, 8, 8],
    "mmb-eshards": [119, 504, 858, 248, 1103, 691, 488, 949, 2162, 340, 106, 953, 1503, 541, 417, 470],
}
SOURCES = {
    "efss": {
        "bibkey": "TCHES:PerWuPic22",
        "url": "https://tches.iacr.org/index.php/TCHES/article/view/9842",
        "locations": "Tables 5/6/8, pp.844/848/854; metric section 2.2",
        "code": "https://github.com/AISyLab/feature_selection_dlsca/tree/eb5b909f8d070faa7ae69051e016acaa6dd5f12d",
        "note": "Our sustained zero-based GE<1 comparison follows the released-code interpretation rather than literal GE=1 wording in the paper. The code averages 100 one-based rank curves, then uses a final <2 gate and reverse >2 argmax. Boundary/indexing differences mean this is not exact bug-compatible reproduction. Published numbers are unchanged.",
    },
    "estranet": {
        "bibkey": "TCHES:HajChoMuk24",
        "url": "https://tches.iacr.org/index.php/TCHES/article/view/11255",
        "locations": "Table 5 p.354, synchronized attack traces; section 6.5",
        "code": "https://github.com/suvadeep-iitb/EstraNet/tree/9668d2c60379ff36400433826e8e4fd3d30c34e0",
        "note": "Table 5 attack desync 0: reported averages across three independently trained models are fixed 14.3 and variable 5.0 (fixed median is 15, best is 13). We compare these published averages with our five-seed ranges. Uses 100 randomized attacks, cropped 10K windows, and profiling desynchronization/augmentation. Published GE=1 is interpreted as rank-one recovery, expressed here as zero-based GE=0; ours uses first GE=0. Released ranks are zero-based but scalar extraction is absent. JSON retains alternative thresholds and sustained crossings.",
    },
    "part": {
        "bibkey": "TCHES:LZCGL21",
        "url": "https://doi.org/10.46586/tches.v2021.i3.235-274",
        "locations": "Table 1 p.248 and section 4.1.2",
        "note": "Fixed 8, variable 10, byte 2, first-round S-box output, full traces. Table defines traces to GE=0; no persistence requirement. 100 random attacks, best network instance. Paper calls fixed ASCAD v1 and variable ASCAD v2 (not the later affine-masked dataset). Fixed-key prose says 10; table value 8 is retained explicitly.",
    },
    "gpam": {
        "bibkey": "bursztein2024gpam",
        "url": "https://doi.org/10.46586/tches.v2024.i3.472-499",
        "locations": "Sections 7/7.1 pp.491-492, Table 11; section 7.2 p.493",
        "note": "Variable-key ASCADv1 byte-2 S-box-input accuracy 95.94%, exact evaluation split unstated. S-box is bijective, so output-label accuracy maps to input-label accuracy. Our SR1 is estimated from 1000 first-trace key ranks with pessimistic ties, not whole-partition top-1 accuracy. ASCADv2 approximately 80 traces to every zero-based byte GE<1, 10000 GE samples; best of seven per masked-output target by validation accuracy. Ours uses one checkpoint per seed, 1000 orders and soft auxiliary reconstruction. Do not describe either as joint success probability one.",
    },
    "mmb": {
        "bibkey": "CiC:WRAPP24",
        "url": "https://doi.org/10.62056/ay4c3txol7",
        "locations": "Table 6 p.15, Table 2 p.13, section 5",
        "code": "https://github.com/wu-lichao/Leakage-Model-flexible-Deep-Learning-based-Side-channel-Analysis/tree/50111ad19d0373101178fcbbca81e9543255442f",
        "note": "Published successor: Leakage Model-flexible Deep Learning-based Side-channel Analysis. eShard shuffling variant unspecified; cannot establish equivalence to our non-shuffled variant. Same physical waveforms can have different shuffling metadata. Paper reports 30000 profiling/5000 attack traces and averages ten scenarios without fully specifying threshold/averaging order. Published per-byte T_GE0 is contextual, not protocol-matched to our first GE=0 counts. Released code is not grounds to alter printed values.",
    },
}
NOTES = """# Prior-work comparison notes

This table is an audit-backed draft, not a claim of controlled architectural superiority.
All our results are recomputed from saved zero-based key ranks; no models or raw datasets are loaded.
Our ranges are min–max across five training seeds. For byte-summary rows, best/mean/worst
are computed across the indicated bytes separately within each seed, before taking seed ranges.
Published best/mean/worst triples are our aggregation of the papers' per-byte values.
Censored published counts remain lower bounds, never substituted as measured means.

- A: EFSS comparison follows the released-code interpretation: ours uses sustained zero-based GE<1.
  Boundary/indexing differences are documented in the source ledger; exact bug-compatible reproduction is not claimed.
- B: EstraNet reported GE=1 is interpreted as rank-one recovery; ours uses first zero-based GE=0;
  first/sustained GE=0, GE<1 and GE<=1 are all retained in JSON for sensitivity checks.
- C: GPAM accuracy comparison uses estimated SR1 over randomized first traces for ours.
  Exact whole-attack-set micro accuracy cannot be recovered from these rank arrays.
- D: We assume MMB uses non-shuffled eShard; the source does not establish the variant.
  Its averaging order remains unresolved.
- E: GPAM ASCADv2 uses per-target best-of-seven selection and 10000 GE samples;
  ours uses five separately evaluated single models and 1000 orders.

All trace counts are positive counts, not zero-based array indices. No crossing within the
saved horizon causes an error rather than a fabricated numerical success. GE is an average
rank across attack orders, not mean per-order disclosure time. Sustained means remaining
below the threshold through the saved horizon; first crossing has no persistence condition.

The generator does not change manuscript files. Caption and source citations belong in the
manuscript; the source ledger below identifies published versions and exact locations.
"""


def crossings(ok, sustained=False):
    if sustained:
        ok = np.logical_and.accumulate(ok[::-1], axis=0)[::-1]
    if not np.all(ok.any(axis=0)):
        raise ValueError("A comparison threshold is not reached within the saved horizon")
    return ok.argmax(axis=0) + 1


def load_dataset(root, dataset):
    runs = []
    for run in seed_runs(root):
        path = run / "attack_metrics.npz"
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(saved["metadata"].item())
            ranks = saved["ranks_zero_based"]
        if (ranks.ndim != 3 or not all(ranks.shape) or ranks.shape[2] != 16
                or not np.issubdtype(ranks.dtype, np.integer)
                or np.any(ranks < 0) or np.any(ranks > 255)):
            raise ValueError(f"{path}: invalid key rank array")
        if metadata["identity"]["dataset"] != dataset:
            raise ValueError(f"{path}: dataset mismatch")
        if ranks.shape[:2] != (metadata["attack_count"], metadata["horizon"]):
            raise ValueError(f"{path}: shape differs from metadata")
        ge = ranks.mean(axis=0)
        metrics = {}
        for name, ok in (("ge0", ge == 0), ("ge_lt1", ge < 1), ("ge_le1", ge <= 1)):
            for sustained in (False, True):
                prefix = "sustained" if sustained else "first"
                metrics[f"{prefix}_{name}"] = crossings(ok, sustained).tolist()
                metrics[f"{prefix}_all_{name}"] = int(crossings(ok.all(axis=1), sustained))
        metrics["byte2_sr1_percent"] = float((ranks[:, 0, 2] == 0).mean() * 100)
        runs.append({"seed": int(run.name.split("=")[1]), "artifact": str(path.resolve()),
                     "sha256": sha256_file(path), "metadata": metadata, "metrics": metrics})
    if len({(r["metadata"]["attack_count"], r["metadata"]["horizon"]) for r in runs}) != 1:
        raise ValueError(f"{dataset}: inconsistent evaluation sizes")
    return runs


def number(value):
    return f"{value:.3f}".rstrip("0").rstrip(".")


def published_triple(key):
    raw = PUBLISHED_BYTES[key]
    censored = [isinstance(v, str) for v in raw]
    values = [float(v[1:]) if isinstance(v, str) else float(v) for v in raw]
    # In the transcribed rows, every censoring limit exceeds the observed minimum.
    if any(censored) and min(v for v, c in zip(values, censored) if c) <= min(values):
        raise ValueError("Cannot determine published minimum from censored data")
    prefix = ">" if any(censored) else ""
    mean = np.floor(np.mean(values) * 1000) / 1000 if any(censored) else np.mean(values)
    return f"{number(min(values))} / {prefix}{number(mean)} / {prefix}{number(max(values))}"


def ours(runs, metric, indices=None):
    values = np.array([run["metrics"][metric] for run in runs])
    if indices is not None:
        values = values[:, indices]
        values = np.stack((values.min(axis=1), values.mean(axis=1), values.max(axis=1)), axis=1)
    if values.ndim == 1:
        values = values[:, None]
    formatted = []
    for column in values.T:
        lo, hi = column.min(), column.max()
        formatted.append(number(lo) if lo == hi else f"{number(lo)}--{number(hi)}")
    return " / ".join(formatted), values.tolist()


def build_rows(results):
    rows = []

    def add(dataset, metric, method, result, source, note="", seed_values=None):
        rows.append(dict(dataset=dataset, metric=metric, method=method, result=result,
                         source=source, note=note, seed_values=seed_values))

    def own(dataset, label, metric, indices=None, note=""):
        runs = results[dataset]
        if metric == "byte2":
            values = [r["metrics"]["first_ge0"][2] for r in runs]
            result = number(min(values)) if min(values) == max(values) else f"{number(min(values))}--{number(max(values))}"
        elif metric == "part":
            values = [r["metrics"]["first_ge0"][2] for r in runs]
            result = number(min(values)) if min(values) == max(values) else f"{number(min(values))}--{number(max(values))}"
        else:
            result, values = ours(runs, metric, indices)
        add(dataset, label, "Ours (five-seed range)", result, "saved ranks", note, values)

    for dataset, tag in (("ascadv1-fixed", "fixed"), ("ascadv1-variable", "variable"), ("ches-ctf-2018", "ches")):
        byte_label = "0--15"
        indices = list(range(16))
        leakage = "ID" if tag != "ches" else "HW"
        add(dataset, f"Bytes {byte_label}: published N_GE1 (best/mean/worst)",
            f"EFSS MLP/{leakage}", published_triple(f"efss-{tag}-mlp"), "efss", "A")
        own(dataset, f"Bytes {byte_label}: sustained GE<1 (best/mean/worst)", "sustained_ge_lt1", indices, "A")
        if tag != "ches":
            add(dataset, "Byte 2: published N_GE1 (mean across three models)", "EstraNet", "14.3" if tag == "fixed" else "5.0", "estranet", "B")
            own(dataset, "Byte 2: first GE=0", "byte2", note="B")
            add(dataset, "Byte 2: first GE=0", "PART (best instance)", "8" if tag == "fixed" else "10", "part")
            own(dataset, "Byte 2: first GE=0", "part")
        if tag == "variable":
            add(dataset, "Byte 2: accuracy (%)", "GPAM", "95.94", "gpam", "C")
            own(dataset, "Byte 2: estimated SR1 (%)", "byte2_sr1_percent", note="C")
    dataset = "eshards-non-shuffled"
    add(dataset, "Bytes 0--15: published T_GE0 (best/mean/worst)", "MMB", published_triple("mmb-eshards"), "mmb", "D")
    own(dataset, "Bytes 0--15: first GE=0 (best/mean/worst)", "first_ge0", list(range(16)), "D")
    add("ascadv2", "All bytes: first GE<1", "GPAM (per-target best of seven)", "roughly 80", "gpam", "E")
    own("ascadv2", "All bytes: first GE<1", "first_all_ge_lt1", note="E")
    return rows


def latex(rows):
    def escape(text):
        return text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("<", r"\textless{}").replace(">", r"\textgreater{}")

    def compact(value):
        return "--".join(f"{float(part):.1f}".removesuffix(".0") for part in value.split("--"))

    def cells(row):
        if " / " in row["result"]:
            return " / ".join(compact(value) for value in row["result"].split(" / ")[1:])
        value = r"$\approx 80$" if row["result"] == "roughly 80" else escape(row["result"])
        if "best of seven" in row["method"]:
            value += " (best of 7 per target)"
        return value

    lines = [r"\begin{tabular}{lllcc}", r"\hline\noalign{\vskip 3pt}",
             r"Dataset & Metric & Prior work & Published & Ours \\[3pt]",
             r"\hline\noalign{\vskip 3pt}"]
    previous = None
    for row, own in zip(rows[::2], rows[1::2]):
        label = DATASETS[row["dataset"]] if row["dataset"] != previous else ""
        if previous is not None and row["dataset"] != previous:
            lines.append(r"\hline\noalign{\vskip 3pt}")
        if row["dataset"] == "ascadv2":
            label += r"$^*$"
        method = escape(row["method"].split(" (")[0])
        source = row["source"]
        if source == "efss":
            method = "EFSS"
            metric = r"Traces to GE $<1$ $\downarrow$ (mean/worst byte)"
        elif source == "estranet":
            metric = r"Traces to GE $=0$ $\downarrow$ (byte 2)"
        elif source == "part":
            metric = r"Traces to GE $=0$ $\downarrow$ (byte 2)"
        elif source == "mmb":
            metric = r"Traces to GE $=0$ $\downarrow$ (mean/worst byte)"
        elif row["dataset"] == "ascadv2":
            metric = r"Traces to GE $<1$ $\downarrow$ (all bytes)"
        else:
            metric = r"Single-trace success rate (\%) $\uparrow$ (byte 2)"
        method += r"~\cite{" + SOURCES[source]["bibkey"] + "}"
        lines.append(" & ".join((label, metric, method, cells(row), cells(own))) + r" \\[3pt]")
        previous = row["dataset"]
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("outputs"))
    parser.add_argument("--dest", type=Path, default=Path("outputs/tables"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = {ext: args.dest / f"prior-work-comparison.{ext}" for ext in ("tex", "csv", "json", "md")}
    if not args.overwrite and any(p.exists() for p in paths.values()):
        parser.error("output bundle exists; use --overwrite")
    results = {ds: load_dataset(args.results_root / ds, ds) for ds in DATASETS}
    rows = build_rows(results)
    args.dest.mkdir(parents=True, exist_ok=True)
    paths["tex"].write_text(latex(rows))
    with paths["csv"].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paths["json"].write_text(json.dumps(dict(rows=rows, runs=results, sources=SOURCES,
        published_byte_values=PUBLISHED_BYTES, generator_sha256=sha256_file(Path(__file__))), indent=2, allow_nan=False) + "\n")
    notes = NOTES + "\n## Sources\n"
    for name, source in SOURCES.items():
        notes += f"\n### {name}\n\n{source['url']}\n\n{source['locations']}\n\n{source['note']}\n"
        if "code" in source:
            notes += f"\nReleased code: {source['code']}\n"
    paths["md"].write_text(notes)
    for path in paths.values():
        print(f"saved {path}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(f"error: {error}") from error
