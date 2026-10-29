import numpy as np
import pytest
import torch

from uncropped_transformers.models.building_blocks.grey_box_heads import ASCADv2Head
from uncropped_transformers.utils import aes


@pytest.fixture(autouse=True)
def single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def one_hot_logits(values):
    return torch.nn.functional.one_hot(values, num_classes=256).double().log()


def test_one_hot_recovery_unmasks_and_inverts_execution_order():
    subbytes = torch.arange(16) * 17
    permutation = torch.roll(torch.arange(16), 3)
    masked = ((subbytes << 1) ^ ((subbytes >> 7) * 0x11B)) ^ 0x53
    labels = torch.cat([torch.tensor([2, 0x53]), masked[permutation], permutation])
    recovered = ASCADv2Head()(one_hot_logits(labels)[None]).exp()
    assert recovered.shape == (1, 16, 256)
    torch.testing.assert_close(recovered[0], one_hot_logits(subbytes).exp())


def sparse_logits(probabilities):
    result = torch.full((256,), float("-inf"), dtype=torch.float64)
    for value, probability in probabilities.items():
        result[value] = torch.tensor(probability, dtype=torch.float64).log()
    return result


def test_soft_recovery_matches_enumeration_of_all_predicted_masks_and_slots():
    alphas = {2: 0.7, 3: 0.3}
    betas = {0: 0.4, 0x53: 0.6}
    masked = [{i * 11: 0.2, i * 11 + 3: 0.8} for i in range(16)]
    permutations = [torch.roll(torch.arange(16), shift) for shift in (3, 7)]
    slot_weights = (0.35, 0.65)
    permutation_logits = [
        sparse_logits(
            {
                int(p[i]): weight
                for p, weight in zip(permutations, slot_weights, strict=True)
            }
        )
        for i in range(16)
    ]
    logits = torch.stack(
        [sparse_logits({0: 0.9, 2: 0.07, 3: 0.03}), sparse_logits(betas)]
        + [sparse_logits(p) for p in masked]
        + permutation_logits
    )[None]
    actual = ASCADv2Head()(logits).exp()[0]
    expected = torch.zeros((16, 256), dtype=torch.float64)
    for slot in range(16):
        for permutation, p_slot in zip(permutations, slot_weights, strict=True):
            for alpha, p_alpha in alphas.items():
                for beta, p_beta in betas.items():
                    for value, p_value in masked[slot].items():
                        subbytes = int(
                            aes.mult_gf256(
                                np.array(aes.GF256_INV[alpha]), np.array(value ^ beta)
                            )
                        )
                        expected[permutation[slot], subbytes] += (
                            p_slot * p_alpha * p_beta * p_value
                        )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.sum(dim=-1), torch.ones(16, dtype=torch.float64))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dense_recovery_is_normalized_chunk_independent_and_ignores_invalid_classes(
    dtype,
):
    generator = torch.Generator().manual_seed(37)
    logits = torch.randn(9, 34, 256, generator=generator, dtype=dtype)
    head = ASCADv2Head()
    expected = torch.cat([head(row[None]) for row in logits])
    logits[:, 0, 0] = 1e6
    logits[:, 18:34, 16:] = 1e6
    actual = head(logits)
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.exp().sum(dim=-1), torch.ones(9, 16, dtype=dtype))


def test_legacy_eighteen_head_output_stays_in_execution_order():
    subbytes = torch.arange(16) * 17
    masked = ((subbytes << 1) ^ ((subbytes >> 7) * 0x11B)) ^ 0x53
    labels = torch.cat([torch.tensor([2, 0x53]), masked])
    actual = ASCADv2Head()(one_hot_logits(labels)[None]).exp()
    torch.testing.assert_close(actual[0], one_hot_logits(subbytes).exp())


def test_nearly_one_hot_predictions_recover_logical_bytes():
    subbytes = torch.arange(16) * 17
    permutation = torch.roll(torch.arange(16), 5)
    masked = ((subbytes << 1) ^ ((subbytes >> 7) * 0x11B)) ^ 0x53
    labels = torch.cat([torch.tensor([2, 0x53]), masked[permutation], permutation])
    logits = one_hot_logits(labels).exp().clamp_min(1e-7).log()[None]
    recovered = ASCADv2Head()(logits).exp()[0]
    torch.testing.assert_close(recovered.argmax(dim=-1), subbytes)
    assert (recovered.gather(-1, subbytes[:, None]) > 0.9998).all()


def test_uniform_beta_makes_all_recovered_subbytes_uniform():
    logits = torch.randn(1, 34, 256, dtype=torch.float64)
    logits[:, 1, :] = 0
    recovered = ASCADv2Head()(logits).exp()
    torch.testing.assert_close(recovered, torch.full_like(recovered, 1 / 256))


def test_inconsistent_permutation_marginals_use_normalized_slot_mixture():
    slots = torch.arange(16)
    logits = one_hot_logits(torch.cat([torch.tensor([1, 0]), slots, slots]))
    weights = (slots[:, None] + slots[None, :] + 1).double()
    probabilities = weights / weights.sum(dim=-1, keepdim=True)
    logits[18:34, :16] = probabilities.log()
    recovered = ASCADv2Head()(logits[None]).exp()[0]
    assert not torch.allclose(
        probabilities.sum(dim=0), torch.ones(16, dtype=torch.float64)
    )
    expected = torch.zeros((16, 256), dtype=torch.float64)
    for byte in range(16):
        for slot in range(16):
            expected[byte, slot] = (
                probabilities[slot, byte] / probabilities[:, byte].sum()
            )
    torch.testing.assert_close(recovered, expected)


def test_recovery_has_finite_gradients_for_soft_predictions():
    logits = torch.randn(1, 34, 256, dtype=torch.float64, requires_grad=True)
    result = ASCADv2Head()(logits)
    result[:, :, 17].sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[:, 0, 1:].abs().sum() > 0).item()
    assert (logits.grad[:, 1:18, :].abs().sum() > 0).item()
    assert (logits.grad[:, 18:34, :16].abs().sum() > 0).item()
    assert not logits.grad[:, 0, 0].any()
    assert not logits.grad[:, 18:34, 16:].any()


@pytest.mark.parametrize("invalid", ["alpha", "permutation", "missing-byte"])
def test_recovery_rejects_empty_valid_support(invalid):
    logits = torch.zeros(1, 34, 256, dtype=torch.float64)
    if invalid == "alpha":
        logits[:, 0, 1:] = float("-inf")
    elif invalid == "permutation":
        logits[:, 18, :16] = float("-inf")
    else:
        logits[:, 18:34, 7] = float("-inf")
    with pytest.raises(ValueError, match="support"):
        ASCADv2Head()(logits)


@pytest.mark.parametrize("shape", [(1, 17, 256), (1, 34, 128), (34, 256)])
def test_recovery_rejects_incompatible_head_layout(shape):
    with pytest.raises(ValueError, match="logits"):
        ASCADv2Head()(torch.zeros(shape))
