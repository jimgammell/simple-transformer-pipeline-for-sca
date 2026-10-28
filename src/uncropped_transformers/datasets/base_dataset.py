from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset


class Base_NumpyDataset(ABC):
    database: Optional[h5py.File]
    trace_count: int
    timestep_count: int

    @abstractmethod
    def target_preds_to_key_preds(
        self,
        target: NDArray[np.floating],
        int_vars: Dict[str, NDArray[np.uint8]],
        byte_idx: Optional[int] = None,
    ) -> NDArray[np.floating]:
        pass

    @abstractmethod
    def compute_intermediate_variables(
        self, *args: NDArray[np.integer]
    ) -> Dict[str, NDArray[np.integer]]:
        pass

    @abstractmethod
    def np_getitem(
        self, _idx: Union[int, Sequence[int]]
    ) -> Tuple[
        NDArray[np.floating], NDArray[np.integer], Dict[str, NDArray[np.integer]]
    ]:
        pass

    @abstractmethod
    def __getitem__(
        self, _idx: Union[int, Sequence[int]]
    ) -> Tuple[
        NDArray[np.floating], NDArray[np.integer], Dict[str, NDArray[np.integer]]
    ]:
        pass

    def __len__(self) -> int:
        assert hasattr(self, "trace_count")
        return self.trace_count

    def __del__(self):
        database: Optional[h5py.File] = getattr(self, "database", None)
        if database is not None:
            database.close()


class Base_TorchDataset(ABC, Dataset):
    def __init__(
        self,
        *,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.transform = transform
        self.target_transform = target_transform

    @abstractmethod
    def __getitem__(
        self, _idx: Union[int, Sequence[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
