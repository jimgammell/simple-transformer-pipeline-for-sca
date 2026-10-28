import importlib
from pathlib import Path

import lightning
import pytest
import torch
import yaml
from test_model import batch as batch
from test_model import module_factory as module_factory
from utils.training_config import SupervisedTrainingConfig


@pytest.fixture
def training_entrypoint(monkeypatch, tmp_path):
    original_load = yaml.load

    def load_directory_config(stream, Loader):
        if getattr(stream, "name", "") == str(
            Path(__file__).resolve().parents[1] / "config" / "directories.yaml"
        ):
            return {
                "downloads-cache-root": str(tmp_path / "cache"),
                "outputs-root": str(tmp_path / "outputs"),
            }
        return original_load(stream, Loader=Loader)

    monkeypatch.setattr(yaml, "load", load_directory_config)
    return importlib.import_module("train")


def test_same_seed_repeats_cpu_dropout_updates(module_factory, batch):
    def run(seed):
        lightning.seed_everything(seed)
        module = module_factory(dropout=0.25).train()
        configured = module.configure_optimizers()
        optimizer = configured["optimizer"]
        scheduler = configured["lr_scheduler"]["scheduler"]
        initial = [p.detach().clone() for p in module.model.parameters()]
        losses = []
        for _ in range(3):
            optimizer.zero_grad()
            trace, target, _ = module.prepare_batch(batch)
            loss = module.compute_loss(module.model(trace), target).mean()
            losses.append(loss.detach().clone())
            loss.backward()
            optimizer.step()
            scheduler.step()
        weights = [p.detach().clone() for p in module.model.parameters()]
        assert any(not torch.equal(a, b) for a, b in zip(initial, weights, strict=True))
        return torch.stack(losses), weights

    losses, weights = run(123)
    repeated_losses, repeated_weights = run(123)
    other_losses, other_weights = run(124)
    assert torch.equal(losses, repeated_losses)
    assert all(
        torch.equal(a, b) for a, b in zip(weights, repeated_weights, strict=True)
    )
    assert not torch.equal(losses, other_losses)
    assert any(
        not torch.equal(a, b) for a, b in zip(weights, other_weights, strict=True)
    )


def test_resolved_config_replays_overrides_and_refuses_changed_identity(
    training_entrypoint, tmp_path
):
    recipe = Path(__file__).resolve().parents[1] / "config" / "ascadv1_fixed.yaml"
    config = SupervisedTrainingConfig(**yaml.safe_load(recipe.read_text())).model_dump()
    training_entrypoint._apply_overrides(
        config,
        [
            "--training.seed",
            "42",
            "--training.compile",
            "false",
            "--training.precision",
            "32-true",
        ],
    )
    resolved = SupervisedTrainingConfig(**config)
    assert resolved.training.seed == 42
    assert resolved.training.compile is False
    assert resolved.training.precision == "32-true"
    dest = tmp_path / "trial"
    training_entrypoint._write_or_check_config(
        dest, resolved.model_dump(), require_identity=True
    )
    saved = (dest / "config.yaml").read_bytes()
    replay = SupervisedTrainingConfig(**yaml.safe_load(saved))
    assert replay == resolved
    training_entrypoint._write_or_check_config(
        dest, replay.model_dump(), require_identity=True
    )
    assert (dest / "config.yaml").read_bytes() == saved
    replay.training.seed += 1
    with pytest.raises(ValueError, match="Configuration mismatch"):
        training_entrypoint._write_or_check_config(
            dest, replay.model_dump(), require_identity=True
        )
    assert (dest / "config.yaml").read_bytes() == saved
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "latest.ckpt").touch()
    with pytest.raises(ValueError, match="Occupied trial destination"):
        training_entrypoint._write_or_check_config(
            occupied, resolved.model_dump(), require_identity=True
        )


@pytest.mark.parametrize(
    "overrides",
    [
        ["training.seed", "1"],
        ["--training.seed"],
        ["--missing.seed", "1"],
        ["--training.missing", "1"],
    ],
)
def test_config_overrides_reject_unknown_or_incomplete_paths(
    training_entrypoint, overrides
):
    with pytest.raises(ValueError):
        training_entrypoint._apply_overrides({"training": {"seed": 0}}, overrides)
