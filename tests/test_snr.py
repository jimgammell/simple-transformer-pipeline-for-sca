from types import SimpleNamespace

import numba
import numpy as np
import pytest

from uncropped_transformers.datasets.ches_ctf_2018 import CHESCTF2018_NumpyDataset
from uncropped_transformers.parametric.snr import compute_snr


@pytest.fixture(autouse=True)
def limited_numba_threads():
    previous = numba.get_num_threads()
    numba.set_num_threads(min(previous, 2))
    try:
        yield
    finally:
        numba.set_num_threads(previous)


class ArrayDataset:
    def __init__(self, traces, labels):
        self.traces = np.asarray(traces)
        self.labels = {
            key: np.asarray(value, dtype=np.uint8) for key, value in labels.items()
        }
        self.trace_count, self.timestep_count = self.traces.shape
        self.requests = []

    def __getitem__(self, index):
        self.requests.append((index.start, index.stop))
        return (
            self.traces[index],
            None,
            {key: value[index] for key, value in self.labels.items()},
        )


def reference_snr(traces, labels):
    traces = np.asarray(traces, dtype=np.float64)
    result = []
    for byte in labels.T:
        groups = [traces[byte == label] for label in np.unique(byte)]
        weights = np.array([len(group) / len(traces) for group in groups])
        means = np.array([group.mean(axis=0) for group in groups])
        variances = np.array([group.var(axis=0, ddof=0) for group in groups])
        center = np.sum(weights[:, None] * means, axis=0)
        signal = np.sum(weights[:, None] * (means - center) ** 2, axis=0)
        noise = np.sum(weights[:, None] * variances, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            result.append(signal / noise)
    return np.array(result)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("chunk_size", [1, 4])
def test_empirical_population_snr_and_chunk_coverage(dtype, chunk_size):
    rng = np.random.default_rng(12)
    labels = np.array([[0, 9]] * 7 + [[3, 9]] * 3 + [[3, 200]] * 2, dtype=np.uint8)
    traces = rng.normal(size=(12, 5)) + labels[:, :1] / 3
    dataset = ArrayDataset(
        traces.astype(dtype), {"multi": labels, "scalar": labels[:, :1]}
    )
    actual = compute_snr(dataset, chunk_size=chunk_size, dtype=dtype)
    for key, value in dataset.labels.items():
        np.testing.assert_allclose(
            actual[key], reference_snr(dataset.traces, value), rtol=2e-6, atol=1e-7
        )
        assert actual[key].shape == (value.shape[1], 5)
    chunks = [
        (start, min(start + chunk_size, 12)) for start in range(0, 12, chunk_size)
    ]
    assert dataset.requests == [(0, min(chunk_size, 12))] + chunks * 4


@pytest.mark.parametrize(
    "traces", [np.ones((4, 2)), np.array([[0, 0], [0, 0], [1, 1], [1, 1]])]
)
def test_zero_noise_returns_nan_or_infinity(traces):
    dataset = ArrayDataset(traces, {"x": [[0], [0], [1], [1]]})
    with np.errstate(divide="raise", invalid="raise"):
        actual = compute_snr(dataset)["x"]
    np.testing.assert_allclose(actual, reference_snr(traces, dataset.labels["x"]))


def test_singleton_class_uses_population_not_sample_variance():
    traces = np.array([[0, 1], [2, 3], [4, 5], [10, 11]])
    dataset = ArrayDataset(traces, {"x": [[0], [0], [0], [255]]})
    np.testing.assert_allclose(
        compute_snr(dataset)["x"], reference_snr(traces, dataset.labels["x"]), rtol=1e-6
    )


@pytest.mark.parametrize("timestep_chunk_size", [2, 20])
def test_timestep_blocks_match_unblocked(timestep_chunk_size):
    dtype = np.float32
    traces = np.array(
        [[0, 1, 0, 2, 0], [2, 1, 0, 4, 1], [4, 1, 0, 6, 2], [10, 1, 1, 8, 3]]
    )
    dataset = ArrayDataset(
        traces,
        {
            "multi": [[0, 9], [0, 9], [0, 200], [255, 200]],
            "scalar": [[0], [0], [0], [255]],
        },
    )
    expected = compute_snr(dataset, chunk_size=3, dtype=dtype)
    actual = compute_snr(
        dataset, chunk_size=3, dtype=dtype, timestep_chunk_size=timestep_chunk_size
    )
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
        assert actual[key].dtype == expected[key].dtype


@pytest.mark.parametrize("chunk_size", [1, 4])
def test_real_numpy_dataset_singleton_chunks(chunk_size):
    dataset = object.__new__(CHESCTF2018_NumpyDataset)
    dataset.trace_count = 5
    dataset.timestep_count = 2
    dataset.trace_indices = np.arange(5)
    dataset.traces = np.arange(10, dtype=np.int8).reshape(5, 2)
    dataset.keys = np.array([[0], [0], [1], [1], [1]], dtype=np.uint8)
    dataset.plaintexts = dataset.keys.copy()
    dataset.ciphertexts = dataset.keys.copy()
    dataset.config = SimpleNamespace(target_variable=("x",), target_byte=[0])
    dataset.init_data = lambda: None
    dataset.compute_intermediate_variables = lambda key, plaintext, ciphertext: {
        "x": key
    }
    actual = compute_snr(dataset, chunk_size=chunk_size)["x"]
    np.testing.assert_allclose(
        actual, reference_snr(dataset.traces, dataset.keys), rtol=1e-6
    )
