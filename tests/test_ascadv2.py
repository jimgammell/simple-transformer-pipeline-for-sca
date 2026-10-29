from pathlib import Path

import numpy as np
import pytest
import yaml
from utils.training_config import SupervisedTrainingConfig

from uncropped_transformers.datasets.ascadv2 import (
    ASCADv2_Config,
    ASCADv2_NumpyDataset,
    repr_target,
)


@pytest.mark.parametrize("index", [0, slice(None)])
@pytest.mark.parametrize("target_bytes", [list(range(16)), [7, 2]])
def test_synthetic_target_order_and_count(tmp_path, index, target_bytes):
    dataset = object.__new__(ASCADv2_NumpyDataset)
    dataset.config = ASCADv2_Config(
        root=tmp_path,
        partition="profile",
        target_byte=target_bytes,
        target_variable=["alpha", "beta", "masked_subbytes", "perm_indices"],
    )
    dataset.trace_indices = np.arange(3)
    dataset.traces = np.zeros((3, 8), dtype=np.int8)
    dataset.keys = np.arange(48, dtype=np.uint8).reshape(3, 16)
    dataset.plaintexts = dataset.keys ^ np.uint8(19)
    dataset.masks = np.arange(57, dtype=np.uint8).reshape(3, 19)
    _, target, variables = dataset.np_getitem(index)
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
    assert dataset.config.num_classes == 256


@pytest.mark.parametrize(
    ("variable", "expected"),
    [("alpha", 1), ("beta", 1), ("perm_indices", 16), ("subbytes", 2)],
)
def test_individual_target_counts(tmp_path, variable, expected):
    config = ASCADv2_Config(
        root=tmp_path, partition="profile", target_byte=[7, 2], target_variable=variable
    )
    assert config.num_labels == expected
    assert repr_target("perm_indices") == r"$\pi$"


def test_saved_recipe_and_existing_configs_validate():
    config_dir = Path(__file__).resolve().parents[1] / "config"
    recipe = SupervisedTrainingConfig(
        **yaml.safe_load((config_dir / "ascadv2.yaml").read_text())
    )
    assert recipe.data.target_variable == [
        "alpha",
        "beta",
        "masked_subbytes",
        "perm_indices",
    ]
    assert recipe.search_space == {}
    assert recipe.training.total_steps == 25000
    assert recipe.training.batch_size == 512
    assert recipe.training.base_lr == 0.0009
    assert recipe.training.weight_decay == 0.05
    assert recipe.training.lr_decay_multiplier == 0.01
    assert recipe.training.precision == "bf16-mixed"
    assert recipe.training.compile is False
    assert recipe.model.embedding_dim == 512
    assert recipe.model.trunk_blocks == 4
    assert recipe.model.patch_size == 500
    for name in ("ascadv1_fixed.yaml", "ascadv1_variable.yaml", "ches_ctf_2018.yaml"):
        config = SupervisedTrainingConfig(
            **yaml.safe_load((config_dir / name).read_text())
        )
        assert isinstance(config.data.target_variable, str)
