import numpy as np
import pytest
import torch

from uncropped_transformers.datasets.eshards import EShards_Config, EShards_TorchDataset
from uncropped_transformers.utils import aes


def test_selected_byte_labels_and_key_scores(tmp_path):
    dataset = EShards_TorchDataset.__new__(EShards_TorchDataset)
    dataset.config = EShards_Config(tmp_path, "attack", target_byte=[9, 2])
    dataset.transform = dataset.target_transform = None
    dataset.trace_count = 4
    dataset.trace_indices = np.array([3, 1, 0, 2])
    dataset.traces = np.arange(32, dtype=np.float32).reshape(4, 8)
    dataset.keys = np.arange(64, dtype=np.uint8).reshape(4, 16)
    dataset.plaintexts = dataset.keys ^ np.uint8(37)
    dataset.ciphertexts = np.zeros_like(dataset.keys)
    trace, target, metadata = dataset[0]
    assert trace.shape == (1, 8)
    torch.testing.assert_close(trace[0], torch.from_numpy(dataset.traces[3]))
    np.testing.assert_array_equal(metadata["plaintext"], dataset.plaintexts[3, [9, 2]])
    np.testing.assert_array_equal(target, aes.SBOX[np.array([37, 37])])
    _, batched_target, batched_metadata = dataset.np_getitem([0, 2])
    assert batched_target.shape == (2, 2)
    scores = np.broadcast_to(np.arange(256), (2, 2, 256)).astype(float)
    mapped = dataset.target_preds_to_key_preds(scores, batched_metadata)
    expected = aes.SBOX[
        batched_metadata["plaintext"][..., None] ^ np.arange(256, dtype=np.uint8)
    ]
    np.testing.assert_array_equal(mapped, expected)
    for variable in ["key", "plaintext", "ciphertext"]:
        with pytest.raises(AssertionError):
            EShards_Config(tmp_path, "attack", target_variable=variable)
