# Spook SCA CTF SW3 (masked software Clyde-128): 62,500 int16 samples per trace, official sharded NPZ layout.
#  profile: <root>/sw3/random_key/rkey_sw3_{N}_{counter}.npz (traces, nonces, umsk_keys; masks/seeds are never read).
#  attack:  <root>/sw3/fixed_key/key_0/fkey_sw3_K0_{N}_{counter}.npz (traces, nonces only).
#  Targets are the unmasked outputs of the first Clyde S-box layer, nonce ^ key -> (y0, y1, y2, y3), as in EstraNet.
#  Each y_r is a uint32 holding one bit of each of 32 parallel S-boxes; target byte 4*r+b is (y_r >> 8*b) & 255.
#  Traces are stored as int16 and are neither cropped, shifted, nor scaled.
#  Attack key scoring is deferred: the serialization of secret_sw3_K0.txt into 32-bit words is not established.

from typing import Literal, Union, List, Sequence, Dict, Tuple, Any, Optional, Callable, get_args
from pathlib import Path
from dataclasses import dataclass
import re

import numpy as np
from numpy.typing import NDArray
import torch

from .common import PARTITION
from .base_dataset import Base_NumpyDataset, Base_TorchDataset
from .compute_trace_statistics import compute_trace_statistics

TARGET_BYTE = Literal[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
]
TARGET_VARIABLE = Literal[
    'subbytes',
    'key',
    'nonce'
]

TIMESTEP_COUNT = 62_500
PROFILE_DIR = Path('sw3') / 'random_key'
ATTACK_DIR = Path('sw3') / 'fixed_key' / 'key_0'
PROFILE_PATTERN = re.compile(r'rkey_sw3_(\d+)_(\d+)\.npz')
ATTACK_PATTERN = re.compile(r'fkey_sw3_K0_(\d+)_(\d+)\.npz')

def sbox_layer(x: Sequence[NDArray[np.uint32]]) -> List[NDArray[np.uint32]]:
    # Clyde S-box layer, identical to sbox_layer in the EstraNet data utilities
    y1 = (x[0] & x[1]) ^ x[2]
    y0 = (x[3] & x[0]) ^ x[1]
    y3 = (y1 & x[3]) ^ x[0]
    y2 = (y0 & y1) ^ x[3]
    return [y0, y1, y2, y3]

def words_to_bytes(words: NDArray[np.uint32]) -> NDArray[np.uint8]:
    # [..., 4] uint32 words -> [..., 16] uint8, byte 4*r+b = (word_r >> 8*b) & 255 (explicit shifts, independent of host endianness)
    words = np.asarray(words, dtype=np.uint32)
    assert words.shape[-1] == 4
    out = np.empty(words.shape[:-1] + (16,), dtype=np.uint8)
    for r in range(4):
        for b in range(4):
            out[..., 4*r + b] = ((words[..., r] >> np.uint32(8*b)) & np.uint32(0xFF)).astype(np.uint8)
    return out

def list_shards(directory: Path, pattern: re.Pattern) -> List[Path]:
    # shards are ordered by their numeric counter (not lexicographically, e.g. 2 < 10)
    assert directory.is_dir(), f'Missing directory {directory}'
    matches = [(int(m.group(2)), p) for p in directory.iterdir() if (m := pattern.fullmatch(p.name)) is not None]
    assert len(matches) > 0, f'No shards matching {pattern.pattern} in {directory}'
    counters = [c for c, _ in matches]
    assert len(set(counters)) == len(counters), f'Duplicate shard counters in {directory}'
    return [p for _, p in sorted(matches)]

@dataclass
class Spook_Config:
    root: Union[str, Path]
    partition: PARTITION
    target_byte: Union[TARGET_BYTE, Sequence[TARGET_BYTE]] = 0
    target_variable: Union[TARGET_VARIABLE, Sequence[TARGET_VARIABLE]] = 'subbytes'

    def __post_init__(self):
        if isinstance(self.root, str):
            self.root = Path(self.root)
        assert self.root.exists()
        assert self.partition in get_args(PARTITION)
        if isinstance(self.target_byte, (int, np.integer)):
            self.target_byte = [self.target_byte]
        self.target_byte = np.array(self.target_byte)
        assert all(x in get_args(TARGET_BYTE) for x in self.target_byte)
        if isinstance(self.target_variable, str):
            self.target_variable = [self.target_variable]
        self.target_variable = np.array(self.target_variable)
        assert all(x in get_args(TARGET_VARIABLE) for x in self.target_variable)
        self.num_labels = len(self.target_byte)*len(self.target_variable)
        self.num_classes = 256

class Spook_NumpyDataset(Base_NumpyDataset):
    int_var_keys = get_args(TARGET_VARIABLE)

    def __init__(
            self,
            *,
            root: Union[str, Path],
            partition: PARTITION,
            target_byte: Union[TARGET_BYTE, List[TARGET_BYTE]] = 0,
            target_variable: Union[TARGET_VARIABLE, List[TARGET_VARIABLE]] = 'subbytes',
    ):
        self.config = Spook_Config(
            root=root,
            partition=partition,
            target_byte=target_byte,
            target_variable=target_variable,
        )

        if self.config.partition == 'profile':
            shard_paths = list_shards(self.config.root / PROFILE_DIR, PROFILE_PATTERN)
        elif self.config.partition == 'attack':
            raise NotImplementedError('Spook SW3 attack targets are deferred: the fixed key (secret_sw3_K0.txt) is not yet parsed into key words.')
        else:
            assert False
        self.timestep_count = TIMESTEP_COUNT
        self.byte_count = 16

        # first pass reads only the small metadata members (npz members are loaded lazily) to size the trace matrix
        nonces, keys, shard_sizes = [], [], []
        for shard_path in shard_paths:
            with np.load(shard_path) as shard:
                shard_nonces = shard['nonces']
                assert shard_nonces.ndim == 2 and shard_nonces.shape[1] == 4
                nonces.append(shard_nonces.astype(np.uint32, copy=False))
                if self.config.partition == 'profile':
                    shard_keys = shard['umsk_keys']
                    assert shard_keys.shape == shard_nonces.shape
                    keys.append(shard_keys.astype(np.uint32, copy=False))
                shard_sizes.append(len(shard_nonces))
        self.shard_sizes = np.array(shard_sizes)
        self.trace_count = int(self.shard_sizes.sum())
        self.nonces = np.concatenate(nonces, axis=0)
        self.keys = np.concatenate(keys, axis=0) if self.config.partition == 'profile' else None

        # second pass fills a preallocated int16 matrix shard by shard so the full set is never held twice
        self.traces = np.empty((self.trace_count, self.timestep_count), dtype=np.int16)
        start_idx = 0
        for shard_path, shard_size in zip(shard_paths, shard_sizes):
            with np.load(shard_path) as shard:
                shard_traces = shard['traces']
                assert shard_traces.dtype == np.int16
                assert shard_traces.shape == (shard_size, self.timestep_count)
                self.traces[start_idx:start_idx + shard_size] = shard_traces
            start_idx += shard_size
        assert start_idx == self.trace_count
        for array in (self.traces, self.nonces, self.keys):
            if array is not None:
                array.flags.writeable = False
        self.trace_indices = np.arange(self.trace_count)

    def get_trace_statistics(self, use_progress_bar: bool = False) -> Dict[str, NDArray[np.floating]]:
        assert self.config.partition == 'profile'
        cache_path = self.config.root / 'spook-sw3.stats-cache.npz'
        if not cache_path.exists():
            trace_mean, trace_var, trace_min, trace_max = compute_trace_statistics(self, chunk_size=4096, use_progress_bar=use_progress_bar)
            np.savez(cache_path, mean=trace_mean, var=trace_var, min=trace_min, max=trace_max)
        cache = np.load(cache_path, allow_pickle=True)
        return cache

    def compute_intermediate_variables(
            self,
            key: NDArray[np.uint32],
            nonce: NDArray[np.uint32]
    ) -> Dict[str, NDArray[np.uint8]]:
        # key, nonce: [..., 4] uint32 words
        sbox_in = [nonce[..., i] ^ key[..., i] for i in range(4)]
        sbox_out = np.stack(sbox_layer(sbox_in), axis=-1)
        intermediate_variables = {
            'key': words_to_bytes(key),
            'nonce': words_to_bytes(nonce),
            'subbytes': words_to_bytes(sbox_out)
        }
        return intermediate_variables

    @staticmethod
    def target_preds_to_key_preds(
            target_preds: NDArray[np.floating],
            intermediate_variables: Dict[str, NDArray[np.uint8]],
    ) -> NDArray[np.floating]:
        raise NotImplementedError(
            'Spook SW3 key scoring is deferred: the mapping from per-byte S-box output predictions to key candidates '
            'and the serialization of secret_sw3_K0.txt into key words have not been established.'
        )

    def np_getitem(self, _idx: Union[int, slice, NDArray[np.integer], Sequence[int]]) -> Tuple[NDArray[np.floating], NDArray[np.integer], Dict[str, NDArray[np.integer]]]:
        if self.keys is None:
            raise NotImplementedError('Spook SW3 attack targets are deferred: the fixed key (secret_sw3_K0.txt) is not yet parsed into key words.')
        if isinstance(_idx, slice):
            _idx = np.arange(*_idx.indices(len(self.trace_indices)))
        elif not isinstance(_idx, (int, np.integer)):
            _idx = np.asarray(_idx)
        if not ((0 <= np.asarray(_idx)).all() and (np.asarray(_idx) < len(self.trace_indices)).all()):
            raise IndexError
        idx = self.trace_indices[_idx]
        # int16 samples are exactly representable in float32
        trace = self.traces[idx, :].astype(np.float32)
        key = self.keys[idx, :]
        nonce = self.nonces[idx, :]
        # only the selected bytes are returned so that target predictions and key predictions line up
        intermediate_variables = {
            k: v[..., self.config.target_byte]
            for k, v in self.compute_intermediate_variables(key, nonce).items()
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
            f'target_byte={self.config.target_byte}',
            f'target_variable={self.config.target_variable}'
        )) + ')'

class Spook_TorchDataset(Base_TorchDataset, Spook_NumpyDataset):
    def __init__(
            self,
            transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            **dataset_config
    ):
        Base_TorchDataset.__init__(self, transform=transform, target_transform=target_transform)
        Spook_NumpyDataset.__init__(self, **dataset_config)

    def __getitem__(self, _idx: Union[int, slice, NDArray[np.integer], Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, NDArray[np.integer]]]:
        trace, target, intermediate_variables = Spook_NumpyDataset.__getitem__(self, _idx)
        trace = torch.tensor(trace).unsqueeze(-2)
        target = torch.tensor(target, dtype=torch.long)
        intermediate_variables = {k: torch.tensor(v, dtype=torch.long) for k, v in intermediate_variables.items()}
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return trace, target, intermediate_variables

    def __len__(self) -> int:
        return Spook_NumpyDataset.__len__(self)
