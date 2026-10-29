import argparse
import logging
from pathlib import Path
from typing import get_args

import numpy as np
from init_things import directories
from init_things.directories import append_directory_clargs
from init_things.mpl_constants import DPI
from matplotlib import pyplot as plt
from numpy.typing import DTypeLike
from utils.load_things import load_numpy_dataset

from uncropped_transformers.datasets import DATASET, PARTITION
from uncropped_transformers.parametric.snr import compute_snr


def run_compute_snr(
    dataset_id: DATASET,
    partition_id: PARTITION,
    dest: Path,
    chunk_size: int = 1024,
    use_progress_bar: bool = True,
    dtype: DTypeLike = np.float32,
    overwrite: bool = False,
    timestep_chunk_size: int | None = None,
):
    assert dest.exists()
    dataset = load_numpy_dataset(dataset_id, partition_id, target_byte=list(range(16)))
    if not overwrite:
        probe_traces, _, intermediate_values = dataset[0:1]
        expected_files = [
            dest / f"{key}.{partition_id}.npy" for key in intermediate_values
        ]
        del probe_traces, intermediate_values
        if expected_files and all(path.exists() for path in expected_files):
            logging.info(
                f"SNR files for {dataset_id}[{partition_id}] already exist. Skipping computation."
            )
            return
        if any(path.exists() for path in expected_files):
            logging.warning(
                "Partial SNR outputs exist. Recomputing all intermediates; existing files will be preserved. Use --overwrite to replace them."
            )
    snr_vals = compute_snr(
        dataset,
        chunk_size=chunk_size,
        use_progress_bar=use_progress_bar,
        dtype=dtype,
        timestep_chunk_size=timestep_chunk_size,
    )
    for int_var_key, int_var_snr in snr_vals.items():
        dest_file = dest / f"{int_var_key}.{partition_id}.npy"
        if dest_file.exists() and not overwrite:
            logging.warning(
                f"File {dest_file} already exists, so the new SNR will not be saved."
            )
            continue
        np.save(dest_file, int_var_snr)


def run_visualize_snr(dest: Path, dataset_id: DATASET, partition_id: PARTITION):
    assert dest.exists()
    var_names = list()
    snr_vals = dict()
    for file in dest.iterdir():
        if not file.name.endswith(".npy"):
            continue
        var_name, _partition_id, _ = file.name.split(".")
        if not _partition_id == partition_id:
            continue
        snr_val = np.load(file)
        var_names.append(var_name)
        snr_vals[var_name] = snr_val
    var_names.sort()
    var_count = len(var_names)
    if not var_count:
        raise ValueError(
            f"No SNR arrays found for partition {partition_id!r} in {dest}"
        )
    byte_count = max(snr_val.shape[0] for snr_val in snr_vals.values())
    fig, axes = plt.subplots(
        var_count,
        byte_count,
        sharex=True,
        sharey="row",
        squeeze=False,
        figsize=(2 * byte_count, 2 * var_count),
    )
    for row_idx, var_name in enumerate(var_names):
        snr_val = snr_vals[var_name]
        for col_idx in range(byte_count):
            ax = axes[row_idx, col_idx]
            if col_idx < snr_val.shape[0]:
                ax.plot(
                    snr_val[col_idx],
                    color="blue",
                    marker=".",
                    markersize=2,
                    linestyle=":",
                    rasterized=True,
                )
                ax.set_yscale("log")
            else:
                ax.axis("off")
    for row_idx in range(var_count):
        axes[row_idx, 0].set_ylabel(f"SNR of {var_names[row_idx]}")
    for col_idx in range(byte_count):
        axes[0, col_idx].set_title(f"Byte {col_idx}")
        axes[-1, col_idx].set_xlabel("Time")
    fig.suptitle(f"SNR of dataset {dataset_id}[{partition_id}]")
    fig.tight_layout()
    fig.savefig(dest / f"{dataset_id}.{partition_id}.pdf", dpi=DPI)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compute and cache all-byte SNR arrays, or plot saved SNR arrays without loading a dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=get_args(DATASET),
        help="Dataset for which to compute the SNR.",
    )
    parser.add_argument(
        "--partition",
        required=True,
        choices=get_args(PARTITION),
        help="Partition of the dataset for which to compute the SNR.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Directory for saved arrays and plots (defaults to <outputs-root>/<dataset>/snr).",
    )
    parser.add_argument(
        "--compute",
        default=False,
        action="store_true",
        help="Compute the SNR on the specified dataset.",
    )
    parser.add_argument(
        "--plot",
        default=False,
        action="store_true",
        help="Plot saved SNR arrays without loading a dataset; existing plots are replaced.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and replace existing SNR arrays.",
    )
    parser.add_argument(
        "--timestep-chunk-size",
        type=int,
        default=None,
        help="Timesteps per SNR block (default: full trace); smaller blocks reduce accumulator memory but reread the dataset for each block.",
    )
    append_directory_clargs(parser)
    args = parser.parse_args()
    directories.init_directories(vars(args), directories.load_directory_config())

    dataset_id: DATASET = args.dataset
    partition_id: PARTITION = args.partition
    dest: Path | None = args.dest
    compute: bool = args.compute
    plot: bool = args.plot

    assert dataset_id in get_args(DATASET)
    assert partition_id in get_args(PARTITION)
    if dest is None:
        assert directories.OUTPUTS_ROOT is not None
        dest = directories.get_output_dir(dataset_id) / "snr"
    else:
        assert isinstance(dest, Path)
    dest.mkdir(exist_ok=True, parents=True)
    assert isinstance(compute, bool)
    assert isinstance(plot, bool)

    if compute:
        logging.info(f"Computing SNR of dataset {dataset_id}[{partition_id}]")
        run_compute_snr(
            dataset_id,
            partition_id,
            dest,
            overwrite=args.overwrite,
            timestep_chunk_size=args.timestep_chunk_size,
        )
    if plot:
        logging.info(f"Visualizing SNR of dataset {dataset_id}[{partition_id}]")
        run_visualize_snr(dest, dataset_id, partition_id)


if __name__ == "__main__":
    main()
