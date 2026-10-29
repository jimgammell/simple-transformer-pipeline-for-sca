from types import SimpleNamespace

import numpy as np
import pytest
import torch

from uncropped_transformers.datasets.ascadv2 import ASCADv2_Config, ASCADv2_NumpyDataset
from uncropped_transformers.evaluation.mtd import accumulate_ranks
from uncropped_transformers.utils import aes


def dataset(tmp_path):
    result = ASCADv2_NumpyDataset.__new__(ASCADv2_NumpyDataset)
    result.config = ASCADv2_Config(
        tmp_path,
        "attack",
        list(range(16)),
        ["alpha", "beta", "masked_subbytes", "perm_indices"],
    )
    return result


def fixture():
    plaintext = np.arange(32, dtype=np.uint8).reshape(2, 16)
    key = np.arange(32, 64, dtype=np.uint8).reshape(2, 16)
    permutation = np.roll(np.arange(16), 5)
    subbytes = aes.SBOX[plaintext ^ key]
    masked = aes.mult_gf256(np.array(2, dtype=np.uint8), subbytes[:, permutation]) ^ 83
    labels = np.concatenate(
        [
            np.tile([2, 83], (2, 1)),
            masked,
            np.tile(permutation, (2, 1)),
        ],
        axis=1,
    )
    logits = torch.full((2, 34, 256), -30.0, dtype=torch.float64)
    logits.scatter_(-1, torch.tensor(labels.astype(np.int64))[..., None], 0.0)
    return logits, {"plaintext": torch.tensor(plaintext), "key": torch.tensor(key)}


def test_simulated_zero_key_accumulation_recovers_all_bytes(tmp_path):
    logits, metadata = fixture()
    original = {k: v.clone() for k, v in metadata.items()}
    ranks = accumulate_ranks(
        logits, metadata, dataset(tmp_path).target_preds_to_key_preds, attack_count=2
    )
    np.testing.assert_array_equal(ranks, np.ones((2, 2, 16)))
    for name in metadata:
        torch.testing.assert_close(metadata[name], original[name])


def test_attack_uses_attack_dataset_callback_for_both_evaluations(
    tmp_path, monkeypatch
):
    from pathlib import Path

    with monkeypatch.context() as import_patch:
        import_patch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
        import attack

    attack_set = dataset(tmp_path)
    logits, metadata = fixture()
    metric = SimpleNamespace(target_preds_to_key_preds=None)
    module = SimpleNamespace(
        config=SimpleNamespace(
            mtd_kwargs={
                "target_preds_to_key_preds": None,
                "attack_count": 2,
            }
        ),
        metrics={"test/mtd": metric},
        to=lambda device: None,
    )

    class Trainer:
        def __init__(self, **kwargs):
            pass

        def test(self, model, **kwargs):
            assert (
                metric.target_preds_to_key_preds == attack_set.target_preds_to_key_preds
            )
            return [{"test/mtd": 1.0}]

        def predict(self, model, **kwargs):
            return [(logits, metadata)]

    monkeypatch.setattr(attack, "load_torch_dataset", lambda *a, **kw: attack_set)
    monkeypatch.setattr(attack, "construct_loaders", lambda *a: [None])
    monkeypatch.setattr(attack, "load_trained_model", lambda *a: module)
    monkeypatch.setattr(attack, "Trainer", Trainer)
    results = attack.run_attack_performance_evalutaion(
        tmp_path / "model.ckpt", "ascadv2"
    )
    np.testing.assert_array_equal(results["ranks_zero_based"], np.zeros((2, 2, 16)))


def test_soft_evidence_ignores_actual_masks_and_matches_candidate_reindex(tmp_path):
    from uncropped_transformers.models.building_blocks.grey_box_heads import ASCADv2Head

    logits, metadata = fixture()
    callback = dataset(tmp_path).target_preds_to_key_preds
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


def test_callback_pickle_contains_no_dataset_state(tmp_path):
    import pickle

    ds = dataset(tmp_path)
    ds.traces = np.zeros((100, 1000), dtype=np.int8)
    callback = ds.target_preds_to_key_preds
    assert not hasattr(callback, "__self__")
    payload = pickle.dumps(callback)
    assert len(payload) < 256
    assert b"traces" not in payload
    assert pickle.loads(payload) is ASCADv2_NumpyDataset.target_preds_to_key_preds


@pytest.mark.parametrize("invalid", ["bytes", "byte-order", "head-order"])
def test_rejects_unsupported_evaluation_layout(tmp_path, invalid, monkeypatch):
    from pathlib import Path

    with monkeypatch.context() as import_patch:
        import_patch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
        import attack

    ds = dataset(tmp_path)
    if invalid == "bytes":
        ds.config.target_byte = np.arange(15)
    elif invalid == "byte-order":
        ds.config.target_byte = np.arange(16)[::-1]
    elif invalid == "head-order":
        ds.config.target_variable = ds.config.target_variable[::-1]
    monkeypatch.setattr(attack, "load_torch_dataset", lambda *a, **kw: ds)
    with pytest.raises(ValueError, match="ASCADv2"):
        attack.run_attack_performance_evalutaion(tmp_path / "model.ckpt", "ascadv2")


def test_callback_rejects_wrong_shape(tmp_path):
    logits, metadata = fixture()
    with pytest.raises(ValueError, match="ASCADv2"):
        dataset(tmp_path).target_preds_to_key_preds(
            logits[:, :18].numpy(), {k: v.numpy() for k, v in metadata.items()}
        )


def test_non_ascadv2_mapping_still_uses_actual_rank_keys():
    from uncropped_transformers.datasets.ascadv1 import ASCADv1_NumpyDataset

    plaintext = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8)
    key = torch.tensor([[17, 93], [17, 93]], dtype=torch.uint8)
    logits = torch.full((2, 2, 256), -30.0)
    labels = torch.from_numpy(aes.SBOX[(plaintext ^ key).numpy()].astype(np.int64))
    logits.scatter_(-1, labels[..., None], 0.0)
    ranks = accumulate_ranks(
        logits,
        {"plaintext": plaintext, "key": key},
        ASCADv1_NumpyDataset.target_preds_to_key_preds,
        attack_count=1,
    )
    np.testing.assert_array_equal(ranks, np.ones((1, 2, 2)))
