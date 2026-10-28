from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import h5py
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset


class Base_NumpyDataset(ABC):
    database: h5py.File | None
    trace_count: int
    timestep_count: int

    @abstractmethod
    def target_preds_to_key_preds(
        self,
        target: NDArray[np.floating],
        int_vars: dict[str, NDArray[np.uint8]],
        byte_idx: int | None = None,
    ) -> NDArray[np.floating]:
        pass

    @abstractmethod
    def compute_intermediate_variables(
        self, *args: NDArray[np.integer]
    ) -> dict[str, NDArray[np.integer]]:
        pass

    @abstractmethod
    def np_getitem(
        self, _idx: int | Sequence[int]
    ) -> tuple[
        NDArray[np.floating], NDArray[np.integer], dict[str, NDArray[np.integer]]
    ]:
        pass

    @abstractmethod
    def __getitem__(
        self, _idx: int | Sequence[int]
    ) -> tuple[
        NDArray[np.floating], NDArray[np.integer], dict[str, NDArray[np.integer]]
    ]:
        pass

    def __len__(self) -> int:
        assert hasattr(self, "trace_count")
        return self.trace_count

    def __del__(self):
        database: h5py.File | None = getattr(self, "database", None)
        if database is not None:
            database.close()


class Base_TorchDataset(ABC, Dataset):
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
        self, _idx: int | Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
