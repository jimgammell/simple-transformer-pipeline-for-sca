# eShard Nucleo_AES_masked_{non_shuffled,shuffled}.ets: 100,000 traces x 1,400 samples, fixed key.
#  The first 80,000 rows are used for profiling and the last 20,000 rows for attacking, identically for both variants.
#  The shuffled variant's supplied plaintext is used as-is.

from typing import Literal, Union, List, Sequence, Dict, Tuple, Any, Optional, Callable, get_args
from pathlib import Path
from dataclasses import dataclass

import h5py
import numpy as np
from numpy.typing import NDArray
import torch

from .common import PARTITION
from .base_dataset import Base_NumpyDataset, Base_TorchDataset
from .compute_trace_statistics import compute_trace_statistics
from uncropped_transformers.utils import aes

TARGET_BYTE = Literal[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
]
TARGET_VARIABLE = Literal[
    'subbytes',
    'key',
    'plaintext',
    'ciphertext'
]

PROFILE_COUNT = 80_000
ATTACK_COUNT = 20_000

@dataclass
class EShards_Config:
    root: Union[str, Path]
    partition: PARTITION
    shuffled: bool
    target_byte: Union[TARGET_BYTE, Sequence[TARGET_BYTE]] = 2
    target_variable: Union[TARGET_VARIABLE, Sequence[TARGET_VARIABLE]] = 'subbytes'

    def __post_init__(self):
        if isinstance(self.root, str):
            self.root = Path(self.root)
        assert self.root.exists()
        assert self.partition in get_args(PARTITION)
        assert isinstance(self.shuffled, (bool, np.bool_))
        if isinstance(self.target_byte, int):
            self.target_byte = [self.target_byte]
        self.target_byte = np.array(self.target_byte)
        assert all(x in get_args(TARGET_BYTE) for x in self.target_byte)
        if isinstance(self.target_variable, str):
            self.target_variable = [self.target_variable]
        self.target_variable = np.array(self.target_variable)
        assert all(x in get_args(TARGET_VARIABLE) for x in self.target_variable)
        self.num_labels = len(self.target_byte)*len(self.target_variable)
        self.num_classes = 256

class EShards_NumpyDataset(Base_NumpyDataset):
    int_var_keys = get_args(TARGET_VARIABLE)

    def __init__(
            self,
            *,
            root: Union[str, Path],
            partition: PARTITION,
            shuffled: bool,
            target_byte: Union[TARGET_BYTE, List[TARGET_BYTE]] = 0,
            target_variable: Union[TARGET_VARIABLE, List[TARGET_VARIABLE]] = 'subbytes',
    ):
        self.config = EShards_Config(
            root=root,
            partition=partition,
            shuffled=shuffled,
            target_byte=target_byte,
            target_variable=target_variable,
        )

        variant = 'shuffled' if self.config.shuffled else 'non_shuffled'
        self.path = self.config.root / f'Nucleo_AES_masked_{variant}.ets'
        if self.config.partition == 'profile':
            rows = slice(0, PROFILE_COUNT)
            self.trace_count = PROFILE_COUNT
        elif self.config.partition == 'attack':
            rows = slice(PROFILE_COUNT, PROFILE_COUNT + ATTACK_COUNT)
            self.trace_count = ATTACK_COUNT
        else:
            assert False
        self.timestep_count = 1400
        self.byte_count = 16

        with h5py.File(self.path, 'r') as database:
            self.traces = database['traces'][rows, :].astype(np.float32, copy=False)
            self.keys = database['metadata/key'][rows, :]
            self.plaintexts = database['metadata/plaintext'][rows, :]
            self.ciphertexts = database['metadata/ciphertext'][rows, :]
        assert self.traces.shape == (self.trace_count, self.timestep_count)
        for array in (self.traces, self.keys, self.plaintexts, self.ciphertexts):
            array.flags.writeable = False
        self.trace_indices = np.arange(self.trace_count)

    def get_trace_statistics(self, use_progress_bar: bool = False) -> Dict[str, NDArray[np.floating]]:
        assert self.config.partition == 'profile'
        variant = 'shuffled' if self.config.shuffled else 'non-shuffled'
        cache_path = self.config.root / f'eshards-{variant}.stats-cache.npz'
        if not cache_path.exists():
            trace_mean, trace_var, trace_min, trace_max = compute_trace_statistics(self, chunk_size=4096, use_progress_bar=use_progress_bar)
            np.savez(cache_path, mean=trace_mean, var=trace_var, min=trace_min, max=trace_max)
        cache = np.load(cache_path, allow_pickle=True)
        return cache

    def compute_intermediate_variables(
            self,
            key: NDArray[np.uint8],
            plaintext: NDArray[np.uint8],
            ciphertext: NDArray[np.uint8]
    ) -> Dict[str, NDArray[np.uint8]]:
        subbytes = aes.SBOX[key ^ plaintext]
        intermediate_variables = {
            'key': key,
            'plaintext': plaintext,
            'ciphertext': ciphertext,
            'subbytes': subbytes
        }
        return intermediate_variables

    @staticmethod
    def target_preds_to_key_preds(
            target_preds: NDArray[np.floating],
            intermediate_variables: Dict[str, NDArray[np.uint8]],
    ) -> NDArray[np.floating]:
        plaintext = intermediate_variables['plaintext']
        assert len(target_preds.shape) == len(plaintext.shape) + 1
        assert target_preds.shape[:-1] == plaintext.shape
        key_candidates = np.arange(256, dtype=np.uint8)
        target_indices = aes.SBOX[key_candidates ^ plaintext[..., np.newaxis]]
        key_preds = np.take_along_axis(target_preds, target_indices.astype(np.intp), axis=-1)
        return key_preds

    def np_getitem(self, _idx: Union[int, slice, NDArray[np.integer], Sequence[int]]) -> Tuple[NDArray[np.floating], NDArray[np.integer], Dict[str, NDArray[np.integer]]]:
        if isinstance(_idx, slice):
            _idx = np.arange(*_idx.indices(len(self.trace_indices)))
        elif not isinstance(_idx, (int, np.integer)):
            _idx = np.asarray(_idx)
        if not ((0 <= np.asarray(_idx)).all() and (np.asarray(_idx) < len(self.trace_indices)).all()):
            raise IndexError
        idx = self.trace_indices[_idx]
        trace = self.traces[idx, :]
        key = self.keys[idx, :]
        plaintext = self.plaintexts[idx, :]
        ciphertext = self.ciphertexts[idx, :]
        # only the selected bytes are returned so that target predictions and key predictions line up
        intermediate_variables = {
            k: v[..., self.config.target_byte]
            for k, v in self.compute_intermediate_variables(key, plaintext, ciphertext).items()
        }
        target = np.concatenate([
            intermediate_variables[target_variable] for target_variable in self.config.target_variable
        ], axis=-1)
        return trace, target, intermediate_variables

    def __getitem__(self, *args, **kwargs) -> Any:
        return self.np_getitem(*args, **kwargs)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(' + ', '.join((
            f'partition={self.config.partition}',
            f'shuffled={self.config.shuffled}',
            f'target_byte={self.config.target_byte}',
            f'target_variable={self.config.target_variable}'
        )) + ')'

class EShards_TorchDataset(Base_TorchDataset, EShards_NumpyDataset):
    def __init__(
            self,
            transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            **dataset_config
    ):
        Base_TorchDataset.__init__(self, transform=transform, target_transform=target_transform)
        EShards_NumpyDataset.__init__(self, **dataset_config)

    def __getitem__(self, _idx: Union[int, slice, NDArray[np.integer], Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, NDArray[np.integer]]]:
        trace, target, intermediate_variables = EShards_NumpyDataset.__getitem__(self, _idx)
        trace = torch.tensor(trace).unsqueeze(-2)
        target = torch.tensor(target, dtype=torch.long)
        intermediate_variables = {k: torch.tensor(v, dtype=torch.long) for k, v in intermediate_variables.items()}
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return trace, target, intermediate_variables

    def __len__(self) -> int:
        return EShards_NumpyDataset.__len__(self)
