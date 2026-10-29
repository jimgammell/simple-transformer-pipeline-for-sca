from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ascadv1 import ASCADv1_Config
    from .ascadv2 import ASCADv2_Config
    from .ches_ctf_2018 import CHESCTF2018_Config
    from .eshards import EShards_Config

import h5py
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset


class Base_NumpyDataset(ABC):
    database: h5py.File | None
    trace_count: int
    timestep_count: int

    @staticmethod
    @abstractmethod
    def target_preds_to_key_preds(
        target_preds: NDArray[np.floating],
        intermediate_variables: dict[str, NDArray[np.integer]],
    ) -> NDArray[np.floating]:
        pass

    @abstractmethod
    def compute_intermediate_variables(
        self, *args: NDArray[np.integer]
    ) -> dict[str, NDArray[np.uint8]]:
        pass

    @abstractmethod
    def np_getitem(
        self, _idx: int | np.integer | slice | NDArray[np.integer] | Sequence[int]
    ) -> tuple[NDArray[np.number], NDArray[np.integer], dict[str, NDArray[np.uint8]]]:
        pass

    @abstractmethod
    def __getitem__(
        self, _idx: int | np.integer | slice | NDArray[np.integer] | Sequence[int]
    ) -> tuple[NDArray[np.number], NDArray[np.integer], dict[str, NDArray[np.uint8]]]:
        pass

    def __len__(self) -> int:
        assert hasattr(self, "trace_count")
        return self.trace_count

    def __del__(self):
        database: h5py.File | None = getattr(self, "database", None)
        if database is not None:
            database.close()


class Base_TorchDataset(ABC, Dataset):
    timestep_count: int
    int_var_keys: tuple[str, ...]
    config: "ASCADv1_Config | ASCADv2_Config | CHESCTF2018_Config | EShards_Config"

    if TYPE_CHECKING:

        @staticmethod
        def target_preds_to_key_preds(
            target_preds: NDArray[np.floating],
            intermediate_variables: dict[str, NDArray[np.integer]],
        ) -> NDArray[np.floating]: ...

        def get_trace_statistics(
            self, use_progress_bar: bool = False
        ) -> dict[str, NDArray[np.floating]]: ...

    def __init__(
        self,
        *,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.transform = transform
        self.target_transform = target_transform

    @abstractmethod
    def __getitem__(
        self, _idx: int | np.integer | slice | NDArray[np.integer] | Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
