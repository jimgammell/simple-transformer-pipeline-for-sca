import numpy as np
import pytest
import torch

from uncropped_transformers.datasets.ascadv1 import ASCADv1_Config, ASCADv1_NumpyDataset
from uncropped_transformers.datasets.ascadv2 import ASCADv2_Config, ASCADv2_NumpyDataset
from uncropped_transformers.datasets.ches_ctf_2018 import (
    CHESCTF2018_Config,
    CHESCTF2018_NumpyDataset,
)
from uncropped_transformers.datasets.eshards import EShards_Config, EShards_TorchDataset
from uncropped_transformers.utils import aes


@pytest.mark.parametrize("kind", ["ascadv1_fixed", "ascadv1_variable", "ches"])
def test_selected_labels_and_key_mapping(tmp_path, kind):
    if kind == "ches":
        dataset = object.__new__(CHESCTF2018_NumpyDataset)
        dataset.config = CHESCTF2018_Config(
            tmp_path, "attack", [9, 2], ["plaintext", "subbytes"]
        )
        dataset.ciphertexts = np.zeros((3, 16), dtype=np.uint8)
    else:
        dataset = object.__new__(ASCADv1_NumpyDataset)
        variable = kind == "ascadv1_variable"
        dataset.config = ASCADv1_Config(
            tmp_path, "attack", [9, 2], ["plaintext", "subbytes"], variable_key=variable
        )
        dataset.masks = np.arange(3 * (18 if variable else 16), dtype=np.uint8).reshape(
            3, -1
        )
    dataset.trace_indices = np.array([2, 0, 1])
    dataset.traces = np.arange(24, dtype=np.int8).reshape(3, 8)
    dataset.keys = np.arange(48, dtype=np.uint8).reshape(3, 16)
    if kind == "ascadv1_fixed":
        dataset.keys[:] = dataset.keys[0]
    dataset.plaintexts = np.arange(48, 96, dtype=np.uint8).reshape(3, 16)
    for index in (0, [0, 2]):
        rows = dataset.trace_indices[index]
        trace, target, metadata = dataset.np_getitem(index)
        plaintext = dataset.plaintexts[rows]
        subbytes = aes.SBOX[plaintext ^ dataset.keys[rows]]
        np.testing.assert_array_equal(trace, dataset.traces[rows])
        np.testing.assert_array_equal(metadata["key"], dataset.keys[rows])
        np.testing.assert_array_equal(metadata["plaintext"], plaintext)
        np.testing.assert_array_equal(
            target,
            np.concatenate([plaintext[..., [9, 2]], subbytes[..., [9, 2]]], axis=-1),
        )
        assert target.shape[-1] == dataset.config.num_labels == 4
        scores = np.broadcast_to(np.arange(256), (*plaintext.shape, 256)).astype(float)
        np.testing.assert_array_equal(
            dataset.target_preds_to_key_preds(scores, metadata),
            aes.SBOX[plaintext[..., None] ^ np.arange(256, dtype=np.uint8)],
        )


@pytest.mark.parametrize("index", [0, slice(None)])
@pytest.mark.parametrize("target_bytes", [list(range(16)), [7, 2]])
def test_ascadv2_target_order_and_count(tmp_path, index, target_bytes):
    dataset = object.__new__(ASCADv2_NumpyDataset)
    dataset.config = ASCADv2_Config(
        root=tmp_path,
        partition="profile",
        target_byte=target_bytes,
        target_variable=["alpha", "beta", "masked_subbytes", "perm_indices"],
    )
    dataset.trace_indices = np.array([2, 0, 1])
    dataset.traces = np.arange(24, dtype=np.int8).reshape(3, 8)
    dataset.keys = np.arange(48, dtype=np.uint8).reshape(3, 16)
    dataset.plaintexts = np.arange(48, 96, dtype=np.uint8).reshape(3, 16)
    dataset.masks = np.arange(57, dtype=np.uint8).reshape(3, 19)
    trace, target, variables = dataset.np_getitem(index)
    rows = dataset.trace_indices[index]
    np.testing.assert_array_equal(trace, dataset.traces[rows])
    np.testing.assert_array_equal(variables["alpha"], dataset.masks[rows, 18, None])
    np.testing.assert_array_equal(variables["beta"], dataset.masks[rows, 17, None])
    np.testing.assert_array_equal(
        np.sort(variables["perm_indices"], axis=-1),
        np.broadcast_to(np.arange(16), variables["perm_indices"].shape),
    )
    subbytes = np.take_along_axis(
        aes.SBOX[dataset.keys[rows] ^ dataset.plaintexts[rows]],
        variables["perm_indices"],
        axis=-1,
    )
    np.testing.assert_array_equal(
        variables["masked_subbytes"],
        aes.mult_gf256(variables["alpha"], subbytes) ^ variables["beta"],
    )
    expected = np.concatenate(
        [
            variables["alpha"],
            variables["beta"],
            variables["masked_subbytes"][..., target_bytes],
            variables["perm_indices"],
        ],
        axis=-1,
    )
    np.testing.assert_array_equal(target, expected)
    assert target.shape[-1] == dataset.config.num_labels == 18 + len(target_bytes)


def test_eshards_selected_byte_labels_and_key_scores(tmp_path):
    dataset = object.__new__(EShards_TorchDataset)
    dataset.config = EShards_Config(tmp_path, "attack", target_byte=[9, 2])
    dataset.transform = dataset.target_transform = None
    dataset.trace_count = 3
    dataset.trace_indices = np.array([2, 0, 1])
    dataset.traces = np.arange(24, dtype=np.float32).reshape(3, 8)
    dataset.keys = np.arange(48, dtype=np.uint8).reshape(3, 16)
    dataset.plaintexts = np.arange(48, 96, dtype=np.uint8).reshape(3, 16)
    dataset.ciphertexts = np.zeros_like(dataset.keys)
    trace, target, metadata = dataset[0]
    assert trace.shape == (1, 8)
    torch.testing.assert_close(trace[0], torch.from_numpy(dataset.traces[2]))
    np.testing.assert_array_equal(metadata["plaintext"], dataset.plaintexts[2, [9, 2]])
    np.testing.assert_array_equal(
        target, aes.SBOX[dataset.keys[2, [9, 2]] ^ dataset.plaintexts[2, [9, 2]]]
    )
    _, batched_target, batched_metadata = dataset.np_getitem([0, 2])
    np.testing.assert_array_equal(
        batched_target,
        aes.SBOX[batched_metadata["key"] ^ batched_metadata["plaintext"]],
    )
    scores = np.broadcast_to(np.arange(256), (2, 2, 256)).astype(float)
    np.testing.assert_array_equal(
        dataset.target_preds_to_key_preds(scores, batched_metadata),
        aes.SBOX[
            batched_metadata["plaintext"][..., None] ^ np.arange(256, dtype=np.uint8)
        ],
    )
    for variable in ["key", "plaintext", "ciphertext"]:
        with pytest.raises(AssertionError):
            EShards_Config(tmp_path, "attack", target_variable=variable)
