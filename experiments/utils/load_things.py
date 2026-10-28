import hashlib
import importlib.metadata
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
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
    if dataset_id not in (
        "ascadv1-fixed",
        "ascadv1-variable",
        "ascadv2",
        "ches-ctf-2018",
    ):
        raise ValueError(f"Unsupported dataset ID: {dataset_id!r}")
    root = {
        "ascadv1-fixed": _dirs.ASCADV1_FIXED_ROOT,
        "ascadv1-variable": _dirs.ASCADV1_VARIABLE_ROOT,
        "ascadv2": _dirs.ASCADV2_ROOT,
        "ches-ctf-2018": _dirs.CHES_CTF_2018_ROOT,
    }[dataset_id]
    if root is None:
        raise ValueError(f"Dataset root is not configured for {dataset_id!r}")
    if dataset_id == "ascadv1-fixed":
        from uncropped_transformers.datasets.ascadv1 import ASCADv1_NumpyDataset

        dataset = ASCADv1_NumpyDataset(
            root=root,
            partition=partition_id,
            variable_key=False,
            cropped_traces=False,
            binary_trace_file=True,
            **kwargs,
        )
    elif dataset_id == "ascadv1-variable":
        from uncropped_transformers.datasets.ascadv1 import ASCADv1_NumpyDataset

        dataset = ASCADv1_NumpyDataset(
            root=root,
            partition=partition_id,
            variable_key=True,
            cropped_traces=False,
            binary_trace_file=True,
            **kwargs,
        )
    elif dataset_id == "ascadv2":
        from uncropped_transformers.datasets.ascadv2 import ASCADv2_NumpyDataset

        dataset = ASCADv2_NumpyDataset(root=root, partition=partition_id, **kwargs)
    elif dataset_id == "ches-ctf-2018":
        from uncropped_transformers.datasets.ches_ctf_2018 import (
            CHESCTF2018_NumpyDataset,
        )

        dataset = CHESCTF2018_NumpyDataset(root=root, partition=partition_id, **kwargs)

    else:
        raise ValueError(f"Unsupported dataset ID: {dataset_id!r}")
    return dataset


def load_torch_dataset(
    dataset_id: DATASET, partition_id: PARTITION, **kwargs
) -> Base_TorchDataset:
    if dataset_id not in (
        "ascadv1-fixed",
        "ascadv1-variable",
        "ascadv2",
        "ches-ctf-2018",
    ):
        raise ValueError(f"Unsupported dataset ID: {dataset_id!r}")
    root = {
        "ascadv1-fixed": _dirs.ASCADV1_FIXED_ROOT,
        "ascadv1-variable": _dirs.ASCADV1_VARIABLE_ROOT,
        "ascadv2": _dirs.ASCADV2_ROOT,
        "ches-ctf-2018": _dirs.CHES_CTF_2018_ROOT,
    }[dataset_id]
    if root is None:
        raise ValueError(f"Dataset root is not configured for {dataset_id!r}")
    if dataset_id == "ascadv1-fixed":
        from uncropped_transformers.datasets.ascadv1 import ASCADv1_TorchDataset

        dataset = ASCADv1_TorchDataset(
            root=root,
            partition=partition_id,
            variable_key=False,
            cropped_traces=False,
            binary_trace_file=True,
            **kwargs,
        )
    elif dataset_id == "ascadv1-variable":
        from uncropped_transformers.datasets.ascadv1 import ASCADv1_TorchDataset

        dataset = ASCADv1_TorchDataset(
            root=root,
            partition=partition_id,
            variable_key=True,
            cropped_traces=False,
            binary_trace_file=True,
            **kwargs,
        )
    elif dataset_id == "ascadv2":
        from uncropped_transformers.datasets.ascadv2 import ASCADv2_TorchDataset

        dataset = ASCADv2_TorchDataset(root=root, partition=partition_id, **kwargs)
    elif dataset_id == "ches-ctf-2018":
        from uncropped_transformers.datasets.ches_ctf_2018 import (
            CHESCTF2018_TorchDataset,
        )

        dataset = CHESCTF2018_TorchDataset(root=root, partition=partition_id, **kwargs)

    else:
        raise ValueError(f"Unsupported dataset ID: {dataset_id!r}")
    return dataset


def construct_loaders(
    train_sets: list[Base_TorchDataset],
    eval_sets: list[Base_TorchDataset],
    batch_size: int = 256,
    num_workers: int = 4,
) -> list[DataLoader]:
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


def source_root() -> Path:
    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip()
    )


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True)


def write_provenance(
    dest: Path,
    config: dict[str, Any],
    *,
    root: Path | None = None,
    trial_number: int | None = None,
) -> None:
    root = source_root() if root is None else root.resolve()
    untracked = git_output(
        root, "ls-files", "--others", "--exclude-standard", "-z"
    ).split("\0")
    source = [
        p
        for p in untracked
        if p
        and (
            (Path(p).parts[0] in ("src", "experiments") and p.endswith(".py"))
            or p == "pyproject.toml"
            or (
                Path(p).parts[0] in ("config", "configs")
                and p.endswith((".yaml", ".yml"))
            )
        )
    ]
    if source:
        raise ValueError(
            f"Untracked source cannot be reconstructed; track these files before training: {source}"
        )
    provenance_path = dest / "provenance.yaml"
    diff_path = dest / "source.diff"
    if provenance_path.exists() or diff_path.exists():
        if not (provenance_path.exists() and diff_path.exists()):
            raise ValueError(
                f"Incomplete original provenance at {dest}; refusing to overwrite it"
            )
        return
    diff = git_output(root, "diff", "HEAD")
    available = torch.cuda.is_available()
    argv = [sys.executable, *sys.argv]
    metadata = {
        "argv": argv,
        "command": shlex.join(argv),
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": dict(
            sorted(
                (d.metadata["Name"], d.version)
                for d in importlib.metadata.distributions()
                if d.metadata["Name"]
            )
        ),
        "torch": {
            "version": str(torch.__version__),
            "cuda": torch.version.cuda,
            "hip": torch.version.hip,
            "cudnn": torch.backends.cudnn.version(),
            "gpu_available": available,
            "gpu_names": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            if available
            else [],
        },
        "source": {
            "root": str(root),
            "commit": git_output(root, "rev-parse", "HEAD").strip(),
            "dirty": bool(git_output(root, "status", "--porcelain")),
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
            "untracked_source_policy": "reject src/experiments Python and config/configs YAML; other untracked files are not captured",
        },
        "config_commit": config.get("commit_hash"),
        "trial_number": trial_number,
    }
    dest.mkdir(parents=True, exist_ok=True)
    with diff_path.open("x") as f:
        f.write(diff)
    with provenance_path.open("x") as f:
        yaml.safe_dump(metadata, f)
