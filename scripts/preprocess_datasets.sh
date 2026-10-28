#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 && $# -ne 2 ]]; then
    printf 'Usage: bash scripts/preprocess_datasets.sh {ascadv1-fixed|ascadv1-variable|ches-ctf-2018|ascadv2} [DATASET_ROOT]\n' >&2
    exit 1
fi
case "$1" in
    ascadv1-fixed|ascadv1-variable|ches-ctf-2018|ascadv2) ;;
    *) printf 'Unknown dataset: %s\n' "$1" >&2; exit 1 ;;
esac

python - "$@" <<'PY'
import sys

from uncropped_transformers.datasets.ascadv1 import ASCADv1_NumpyDataset
from uncropped_transformers.datasets.ascadv2 import ASCADv2_NumpyDataset
from uncropped_transformers.datasets.ches_ctf_2018 import CHESCTF2018_NumpyDataset

dataset_id = sys.argv[1]
default_roots = {
    "ascadv1-fixed": "./datasets/ascadv1_fixed",
    "ascadv1-variable": "./datasets/ascadv1_variable",
    "ches-ctf-2018": "./datasets/ches_ctf_2018",
    "ascadv2": "./datasets/ascadv2",
}
root = sys.argv[2] if len(sys.argv) == 3 else default_roots[dataset_id]
for partition in ("profile", "attack"):
    if dataset_id == "ches-ctf-2018":
        dataset = CHESCTF2018_NumpyDataset(
            root=root, partition=partition
        )
    elif dataset_id == "ascadv2":
        dataset = ASCADv2_NumpyDataset(root=root, partition=partition)
    else:
        variable_key = dataset_id == "ascadv1-variable"
        dataset = ASCADv1_NumpyDataset(
            root=root,
            partition=partition,
            variable_key=variable_key,
            cropped_traces=False,
        )
    if partition == "profile":
        dataset.get_trace_statistics(use_progress_bar=True)
    else:
        dataset.init_data()
PY
