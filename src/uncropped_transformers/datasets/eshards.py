from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Literal,
    get_args,
)

import numpy as np
import torch
from numpy.typing import NDArray

from uncropped_transformers.utils import aes

from .base_dataset import Base_NumpyDataset, Base_TorchDataset
from .common import PARTITION
from .compute_trace_statistics import compute_trace_statistics

TARGET_BYTE = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
TARGET_VARIABLE = Literal["subbytes"]


@dataclass
class EShards_Config:
    root: Path
    partition: PARTITION
    target_byte: NDArray[np.integer]
    target_variable: NDArray[np.str_]

    def __init__(
        self,
        root: str | Path,
        partition: PARTITION,
        target_byte: TARGET_BYTE | Sequence[TARGET_BYTE] = 2,
        target_variable: TARGET_VARIABLE | Sequence[TARGET_VARIABLE] = "subbytes",
    ):
        if isinstance(root, str):
            root = Path(root)
        self.root = root
        self.partition = partition
        if isinstance(target_byte, int):
            target_byte = [target_byte]
        self.target_byte = np.array(target_byte)
        if isinstance(target_variable, str):
            target_variable = [target_variable]
        self.target_variable = np.array(target_variable)
        self.__post_init__()

    def __post_init__(self):
        assert self.root.exists()
        assert self.partition in get_args(PARTITION)
        assert all(x in get_args(TARGET_BYTE) for x in self.target_byte)
        assert all(x in get_args(TARGET_VARIABLE) for x in self.target_variable)
        self.num_labels = len(self.target_byte) * len(self.target_variable)
        self.num_classes = 256


class EShards_NumpyDataset(Base_NumpyDataset):
    traces: NDArray[np.float32] | None
    keys: NDArray[np.uint8] | None
    plaintexts: NDArray[np.uint8] | None
    ciphertexts: NDArray[np.uint8] | None
    int_var_keys = ("subbytes", "key", "plaintext", "ciphertext")

    def __init__(
        self,
        *,
        root: str | Path,
        partition: PARTITION,
        target_byte: TARGET_BYTE | list[TARGET_BYTE] = 0,
        target_variable: TARGET_VARIABLE | list[TARGET_VARIABLE] = "subbytes",
    ):
        self.config = EShards_Config(
            root=root,
            partition=partition,
            target_byte=target_byte,
            target_variable=target_variable,
        )

        from estraces.formats import ets_format

        self.path = self.config.root / "Nucleo_AES_masked_non_shuffled.ets"
        rows = slice(0, 80_000) if partition == "profile" else slice(80_000, 100_000)
        database = ets_format.read_ths_from_ets_file(self.path)
        self.traces = np.asarray(database.samples[rows, :], dtype=np.float32)
        self.keys = np.asarray(database.metadatas["key"][rows], dtype=np.uint8)
        self.plaintexts = np.asarray(
            database.metadatas["plaintext"][rows], dtype=np.uint8
        )
        self.ciphertexts = np.asarray(
            database.metadatas["ciphertext"][rows], dtype=np.uint8
        )
        self.trace_count = 80_000 if partition == "profile" else 20_000
        self.timestep_count = 1400
        self.byte_count = 16
        assert self.traces.shape == (self.trace_count, self.timestep_count)
        for array in (self.traces, self.keys, self.plaintexts, self.ciphertexts):
            array.flags.writeable = False
        self.trace_indices = np.arange(self.trace_count)

    def get_trace_statistics(
        self, use_progress_bar: bool = False
    ) -> dict[str, NDArray[np.floating]]:
        assert self.config.partition == "profile"
        cache_path = self.config.root / "eshards-non-shuffled.stats-cache.npz"
        if not cache_path.exists():
            trace_mean, trace_var, trace_min, trace_max = compute_trace_statistics(
                self, chunk_size=4096, use_progress_bar=use_progress_bar
            )
            np.savez(
                cache_path, mean=trace_mean, var=trace_var, min=trace_min, max=trace_max
            )
        cache = np.load(cache_path, allow_pickle=True)
        return cache

    def compute_intermediate_variables(
        self,
        key: NDArray[np.uint8],
        plaintext: NDArray[np.uint8],
        ciphertext: NDArray[np.uint8],
    ) -> dict[str, NDArray[np.uint8]]:
        subbytes = aes.SBOX[key ^ plaintext]
        intermediate_variables = {
            "key": key,
            "plaintext": plaintext,
            "ciphertext": ciphertext,
            "subbytes": subbytes,
        }
        return intermediate_variables

    @staticmethod
    def target_preds_to_key_preds(
        target_preds: NDArray[np.floating],
        intermediate_variables: dict[str, NDArray[np.integer]],
    ) -> NDArray[np.floating]:
        plaintext = intermediate_variables["plaintext"]
        assert len(target_preds.shape) == len(plaintext.shape) + 1
        assert target_preds.shape[:-1] == plaintext.shape
        key_candidates = np.arange(256, dtype=np.uint8)
        target_indices = aes.SBOX[key_candidates ^ plaintext[..., np.newaxis]]
        key_preds = np.take_along_axis(
            target_preds, target_indices.astype(np.intp), axis=-1
        )
        return key_preds

    def np_getitem(
        self, _idx: int | np.integer | slice | NDArray[np.integer] | Sequence[int]
    ) -> tuple[NDArray[np.float32], NDArray[np.integer], dict[str, NDArray[np.uint8]]]:
        if isinstance(_idx, slice):
            _idx = np.arange(*_idx.indices(len(self.trace_indices)))
        elif isinstance(_idx, (int, np.integer)):
            _idx = np.asarray(_idx)
        elif not isinstance(_idx, np.ndarray):
            _idx = np.array(_idx)
        if not ((0 <= _idx).all() and (_idx < len(self.trace_indices)).all()):
            raise IndexError
        assert self.traces is not None
        assert self.keys is not None
        assert self.plaintexts is not None
        assert self.ciphertexts is not None
        idx = self.trace_indices[_idx]
        trace = self.traces[idx, :]
        key = self.keys[idx, :]
        plaintext = self.plaintexts[idx, :]
        ciphertext = self.ciphertexts[idx, :]
        intermediate_variables = {
            name: value[..., self.config.target_byte]
            for name, value in self.compute_intermediate_variables(
                key, plaintext, ciphertext
            ).items()
        }
        target = np.concatenate(
            [
                intermediate_variables[target_variable]
                for target_variable in self.config.target_variable
            ],
            axis=-1,
        )
        return trace, target, intermediate_variables

    def __getitem__(
        self, *args, **kwargs
    ) -> tuple[NDArray[np.float32], NDArray[np.integer], dict[str, NDArray[np.uint8]]]:
        return self.np_getitem(*args, **kwargs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            + ", ".join(
                (
                    f"partition={self.config.partition}",
                    f"target_byte={self.config.target_byte}",
                    f"target_variable={self.config.target_variable}",
                )
            )
            + ")"
        )


class EShards_TorchDataset(Base_TorchDataset, EShards_NumpyDataset):
    def __init__(
        self,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        **dataset_config,
    ):
        Base_TorchDataset.__init__(
            self, transform=transform, target_transform=target_transform
        )
        EShards_NumpyDataset.__init__(self, **dataset_config)

    def __getitem__(
        self, _idx: int | np.integer | slice | NDArray[np.integer] | Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        trace, target, intermediate_variables = EShards_NumpyDataset.__getitem__(
            self, _idx
        )
        trace = torch.tensor(trace).unsqueeze(-2)
        target = torch.tensor(target, dtype=torch.long)
        intermediate_variables = {
            k: torch.tensor(v, dtype=torch.long)
            for k, v in intermediate_variables.items()
        }
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return trace, target, intermediate_variables

    def __len__(self) -> int:
        return EShards_NumpyDataset.__len__(self)
