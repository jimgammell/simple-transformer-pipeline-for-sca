import numpy as np
import pytest
import torch

from uncropped_transformers.datasets.ascadv2 import ASCADv2_NumpyDataset
from uncropped_transformers.evaluation.mtd import accumulate_ranks
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


def fixture():
    plaintext = np.arange(32, dtype=np.uint8).reshape(2, 16)
    key = np.arange(32, 64, dtype=np.uint8).reshape(2, 16)
    permutation = np.roll(np.arange(16), 5)
    subbytes = aes.SBOX[plaintext ^ key]
    masked = aes.mult_gf256(np.array(2, dtype=np.uint8), subbytes[:, permutation]) ^ 83
    labels = np.concatenate(
        [np.tile([2, 83], (2, 1)), masked, np.tile(permutation, (2, 1))], axis=1
    )
    logits = torch.full((2, 34, 256), -30.0, dtype=torch.float64)
    logits.scatter_(-1, torch.tensor(labels.astype(np.int64))[..., None], 0.0)
    return logits, {"plaintext": torch.tensor(plaintext), "key": torch.tensor(key)}


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


def test_simulated_zero_key_accumulation_recovers_all_bytes():
    logits, metadata = fixture()
    original = {k: v.clone() for k, v in metadata.items()}
    ranks = accumulate_ranks(
        logits, metadata, ASCADv2_NumpyDataset.target_preds_to_key_preds, attack_count=2
    )
    np.testing.assert_array_equal(ranks, np.ones((2, 2, 16)))
    for name in metadata:
        torch.testing.assert_close(metadata[name], original[name])


def test_soft_evidence_ignores_actual_masks_and_matches_candidate_reindex():
    logits, metadata = fixture()
    callback = ASCADv2_NumpyDataset.target_preds_to_key_preds
    probabilities = torch.log_softmax(logits, dim=-1).numpy()
    raw = {k: v.numpy() for k, v in metadata.items()}
    effective = raw["plaintext"] ^ raw["key"]
    actual = callback(probabilities, {"plaintext": effective})
    poisoned = {
        "plaintext": effective,
        "key": np.full((2, 16), 111),
        "alpha": np.zeros((2, 1), dtype=np.uint8),
        "beta": np.full((2, 1), 201),
        "perm_indices": np.zeros((2, 16), dtype=np.uint8),
        "masked_subbytes": np.zeros((2, 16), dtype=np.uint8),
    }
    np.testing.assert_array_equal(actual, callback(probabilities, poisoned))
    subbytes = ASCADv2Head()(logits).numpy()
    candidates = np.arange(256, dtype=np.uint8)
    original_scores = np.take_along_axis(
        subbytes,
        aes.SBOX[raw["plaintext"][..., None] ^ candidates],
        axis=-1,
    )
    reindexed = np.take_along_axis(
        original_scores,
        raw["key"][..., None] ^ candidates,
        axis=-1,
    )
    np.testing.assert_allclose(actual, reindexed, atol=1e-10)
    np.testing.assert_array_equal(actual.sum(axis=0).argmax(axis=-1), np.zeros(16))
    corrupted = {k: torch.tensor(v) for k, v in poisoned.items()}
    corrupted["plaintext"] = metadata["plaintext"]
    corrupted["key"] = metadata["key"]
    np.testing.assert_array_equal(
        accumulate_ranks(logits, corrupted, callback, attack_count=1),
        accumulate_ranks(logits, metadata, callback, attack_count=1),
    )


@pytest.mark.parametrize("shape", [(1, 17, 256), (1, 34, 128), (34, 256)])
def test_recovery_rejects_incompatible_head_layout(shape):
    with pytest.raises(ValueError, match="logits"):
        ASCADv2Head()(torch.zeros(shape))
