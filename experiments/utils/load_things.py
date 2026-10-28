from pathlib import Path

import torch
from init_things import directories as _dirs
from torch.utils.data import DataLoader

from uncropped_transformers.datasets import (
    DATASET,
    PARTITION,
    Base_NumpyDataset,
    Base_TorchDataset,
)
from uncropped_transformers.training.supervised_lightning_module import SupervisedModule


def load_numpy_dataset(
    dataset_id: DATASET, partition_id: PARTITION, **kwargs
) -> Base_NumpyDataset:
    if dataset_id == "ascadv1-fixed":
        from uncropped_transformers.datasets.ascadv1 import ASCADv1_NumpyDataset

        dataset = ASCADv1_NumpyDataset(
            root=_dirs.ASCADV1_FIXED_ROOT,
            partition=partition_id,
            variable_key=False,
            cropped_traces=False,
            binary_trace_file=True,
            **kwargs,
        )
    elif dataset_id == "ascadv1-variable":
        from uncropped_transformers.datasets.ascadv1 import ASCADv1_NumpyDataset

        dataset = ASCADv1_NumpyDataset(
            root=_dirs.ASCADV1_VARIABLE_ROOT,
            partition=partition_id,
            variable_key=True,
            cropped_traces=False,
            binary_trace_file=True,
            **kwargs,
        )
    elif dataset_id == "ascadv2":
        from uncropped_transformers.datasets.ascadv2 import ASCADv2_NumpyDataset

        dataset = ASCADv2_NumpyDataset(
            root=_dirs.ASCADV2_ROOT, partition=partition_id, **kwargs
        )
    elif dataset_id == "ches-ctf-2018":
        from uncropped_transformers.datasets.ches_ctf_2018 import (
            CHESCTF2018_NumpyDataset,
        )

        dataset = CHESCTF2018_NumpyDataset(
            root=_dirs.CHES_CTF_2018_ROOT, partition=partition_id, **kwargs
        )

    else:
        raise ValueError(f"Unsupported dataset ID: {dataset_id!r}")
    return dataset


def load_torch_dataset(
    dataset_id: DATASET, partition_id: PARTITION, **kwargs
) -> Base_TorchDataset:
    if dataset_id == "ascadv1-fixed":
        from uncropped_transformers.datasets.ascadv1 import ASCADv1_TorchDataset

        dataset = ASCADv1_TorchDataset(
            root=_dirs.ASCADV1_FIXED_ROOT,
            partition=partition_id,
            variable_key=False,
            cropped_traces=False,
            binary_trace_file=True,
            **kwargs,
        )
    elif dataset_id == "ascadv1-variable":
        from uncropped_transformers.datasets.ascadv1 import ASCADv1_TorchDataset

        dataset = ASCADv1_TorchDataset(
            root=_dirs.ASCADV1_VARIABLE_ROOT,
            partition=partition_id,
            variable_key=True,
            cropped_traces=False,
            binary_trace_file=True,
            **kwargs,
        )
    elif dataset_id == "ascadv2":
        from uncropped_transformers.datasets.ascadv2 import ASCADv2_TorchDataset

        dataset = ASCADv2_TorchDataset(
            root=_dirs.ASCADV2_ROOT, partition=partition_id, **kwargs
        )
    elif dataset_id == "ches-ctf-2018":
        from uncropped_transformers.datasets.ches_ctf_2018 import (
            CHESCTF2018_TorchDataset,
        )

        dataset = CHESCTF2018_TorchDataset(
            root=_dirs.CHES_CTF_2018_ROOT, partition=partition_id, **kwargs
        )

    else:
        raise ValueError(f"Unsupported dataset ID: {dataset_id!r}")
    return dataset


def construct_loaders(
    train_sets: list[Base_TorchDataset],
    eval_sets: list[Base_TorchDataset],
    batch_size: int = 256,
    num_workers: int = 4,
) -> tuple[DataLoader, ...]:
    common_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    loaders = []
    for train_set in train_sets:
        loader = DataLoader(train_set, shuffle=True, drop_last=True, **common_kwargs)
        loaders.append(loader)
    for eval_set in eval_sets:
        loader = DataLoader(eval_set, shuffle=False, drop_last=False, **common_kwargs)
        loaders.append(loader)
    return loaders


def load_trained_model(
    ckpt_path: Path, profiling_set: Base_TorchDataset | None = None
) -> SupervisedModule:
    assert ckpt_path.exists()
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    names = ("trace_mean", "trace_std", "trace_min", "trace_rng")
    if not all(name in checkpoint["state_dict"] for name in names):
        raise ValueError(
            "Checkpoint is missing persistent normalization statistics; regenerate the checkpoint."
        )
    statistics = {name: checkpoint["state_dict"][name] for name in names}
    del checkpoint
    module = SupervisedModule.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
        checkpoint_statistics=statistics,
        strict=True,
    )
    return module
