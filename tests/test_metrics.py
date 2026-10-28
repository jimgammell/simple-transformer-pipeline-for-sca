import numpy as np
import pytest
import torch
from utils.artifacts import summarize_ranks

from uncropped_transformers.evaluation.mtd import accumulate_ranks, compute_mtd
from uncropped_transformers.evaluation.rank import Rank, get_rank


@pytest.mark.parametrize("tensor", [False, True])
def test_rank_uses_pessimistic_ties_and_accumulates_batches(tensor):
    logits = np.array(
        [[[4, 0, 0], [1, 1, 1]], [[0, 2, 2], [0, 0, 4]]], dtype=np.float32
    )
    targets = np.array([[0, 1], [1, 2]])
    predictions = torch.from_numpy(logits) if tensor else logits
    labels = torch.from_numpy(targets) if tensor else targets
    np.testing.assert_array_equal(get_rank(predictions, labels), [[1, 3], [2, 1]])
    metric = Rank()
    for index in range(2):
        metric.update(torch.from_numpy(logits[index]), torch.from_numpy(targets[index]))
    assert metric.compute().item() == 1.75
    metric.reset()
    metric.update(torch.tensor([[4.0, 0, 0]]), torch.tensor([0]))
    assert metric.compute().item() == 1


@pytest.mark.parametrize("tied,expected", [(False, 1), (True, 4)])
def test_accumulate_ranks_perfect_logits_and_ties(tied, expected):
    logits = torch.zeros(3, 2, 4)
    if not tied:
        logits[:, :, 0] = 10
    keys = {"key": torch.zeros((3, 2), dtype=torch.long)}
    ranks = accumulate_ranks(
        logits,
        keys,
        lambda predictions, variables: predictions,
        attack_count=2,
        traces_per_attack=9,
    )
    np.testing.assert_array_equal(ranks, np.full((2, 3, 2), expected))
    assert compute_mtd(ranks, "max") == (1 if not tied else 4)


def test_accumulated_evidence_can_lose_and_recover_rank():
    order = np.random.default_rng(seed=0).choice(3, size=3, replace=False)
    logits = torch.zeros(3, 1, 2)
    logits[order, 0, 0] = torch.tensor([2.0, -4.0, 5.0])
    ranks = accumulate_ranks(
        logits,
        {"key": torch.zeros((3, 1), dtype=torch.long)},
        lambda predictions, variables: predictions,
        attack_count=1,
        traces_per_attack=3,
    )
    np.testing.assert_array_equal(ranks, [[[1], [2], [1]]])
    assert compute_mtd(ranks, "max") == 3


def test_mtd_requires_sustained_recovery_and_censors_at_horizon_plus_one():
    ranks = np.array(
        [[[1, 2], [2, 1], [1, 2], [1, 1]], [[1, 1], [1, 1], [1, 1], [1, 2]]],
        dtype=float,
    )
    np.testing.assert_array_equal(compute_mtd(ranks, "none"), [[3, 4], [1, 5]])
    assert compute_mtd(ranks, "max") == 4.5
    assert compute_mtd(ranks, "mean") == 3.25
    with pytest.raises(ValueError, match="reduction"):
        compute_mtd(ranks, "invalid")


def test_rank_summaries_preserve_uint8_extremes_and_simultaneous_success():
    ranks = np.array([[[1, 256], [1, 1]], [[256, 1], [1, 1]]], dtype=float)
    summary = summarize_ranks(ranks)
    assert summary["ranks_zero_based"].dtype == np.uint8
    np.testing.assert_array_equal(
        summary["ranks_zero_based"].astype(np.int16) + 1, ranks
    )
    np.testing.assert_array_equal(summary["guessing_entropy"], [[128.5, 1], [128.5, 1]])
    np.testing.assert_array_equal(summary["success_rate"], [[0.5, 1], [0.5, 1]])
    np.testing.assert_array_equal(summary["full_key_success_rate"], [0, 1])
    assert compute_mtd(ranks, "max") == 2


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros((1, 2)),
        np.empty((0, 2, 1)),
        [[[0]]],
        [[[257]]],
        [[[1.5]]],
        [[[np.nan]]],
        [[[np.inf]]],
    ],
)
def test_rank_summaries_reject_invalid_inputs(invalid):
    with pytest.raises(ValueError, match="finite integer one-based ranks"):
        summarize_ranks(invalid)
