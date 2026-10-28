import lightning
import numpy as np
import pytest
import torch

from uncropped_transformers.models import Model
from uncropped_transformers.training import SupervisedModule


@pytest.fixture
def module_factory():
    def create(head="tied", fourier=False, dropout=0.0):
        return SupervisedModule(
            model_constructor=Model,
            model_kwargs=dict(
                input_length=16,
                output_count=2,
                grey_box_head=None,
                position_embedding="rope",
                pooling="token",
                head=head,
                fnn_style="mlp",
                patch_size=4,
                use_fourier_embed=fourier,
                fourier_embed_num_bands=2 if fourier else None,
                fourier_embed_sigma=1.0 if fourier else None,
                embedding_dim=16,
                expansion_factor=2,
                trunk_blocks=1,
                head_count=2,
                input_dropout_rate=dropout,
                input_droppatch_rate=0.0,
                hidden_dropout_rate=dropout,
                use_bias=False,
            ),
            leakage_model="id",
            num_labels=2,
            total_steps=4,
            lr_warmup_steps=0,
            lr_const_steps=0,
            base_lr=0.001,
            lr_decay_multiplier=0.1,
            weight_decay=0.01,
            label_smoothing=0.0,
            mtd_kwargs=dict(
                target_preds_to_key_preds=identity_key_predictions,
                int_var_keys=["key"],
                attack_count=2,
                traces_per_attack=3,
            ),
            trace_statistics=dict(
                mean=np.full((1, 16), 2.0, dtype=np.float32),
                var=np.full((1, 16), 4.0, dtype=np.float32),
                min=np.full((1, 16), -1.0, dtype=np.float32),
                max=np.full((1, 16), 5.0, dtype=np.float32),
            ),
            additive_gaussian_noise=0.0,
            mixup_alpha=0.0,
            preprocessing="standardize",
            random_roll_scale=0.0,
            random_lpf_scale=0.0,
        )

    return create


def identity_key_predictions(predictions, intermediate_variables):
    return predictions


@pytest.fixture
def batch():
    return (
        torch.linspace(-2, 3, 48).reshape(3, 1, 16),
        torch.tensor([[0, 1], [2, 3], [4, 5]]),
        {"key": torch.zeros((3, 2), dtype=torch.long)},
    )


@pytest.mark.parametrize(
    "head,fourier", [("tied", False), ("untied", False), ("tied", True)]
)
def test_model_supervised_update_has_finite_gradients(
    module_factory, batch, head, fourier
):
    module = module_factory(head=head, fourier=fourier)
    trace, target, _ = module.prepare_batch(batch)
    logits = module.model(trace)
    assert logits.shape == (3, 2, 256)
    assert torch.isfinite(logits).all()
    losses = module.compute_loss(logits, target)
    assert losses.shape == (3, 2)
    assert torch.isfinite(losses).all()
    optimizer = module.configure_optimizers()["optimizer"]
    before = [p.detach().clone() for p in module.model.parameters()]
    losses.mean().backward()
    gradients = [p.grad for p in module.model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g) > 0 for g in gradients)
    optimizer.step()
    assert any(
        not torch.equal(a, b)
        for a, b in zip(before, module.model.parameters(), strict=True)
    )


@pytest.mark.parametrize("preprocessing", ["standardize", "normalize"])
def test_checkpoint_preserves_statistics_and_eval_predictions(
    module_factory, batch, tmp_path, preprocessing
):
    module = module_factory().eval()
    module.config.preprocessing = preprocessing
    module.hparams["preprocessing"] = preprocessing
    trace = module.prepare_batch(batch)[0]
    with torch.no_grad():
        expected = module.model(trace)
    checkpoint = dict(
        state_dict=module.state_dict(),
        hyper_parameters=dict(module.hparams),
        **{"pytorch-lightning_version": lightning.__version__},
    )
    path = tmp_path / "model.ckpt"
    torch.save(checkpoint, path)
    statistics = {
        name: value
        for name, value in torch.load(path, map_location="cpu", weights_only=False)[
            "state_dict"
        ].items()
        if name in ("trace_mean", "trace_std", "trace_min", "trace_rng")
    }
    restored = SupervisedModule.load_from_checkpoint(
        path,
        map_location="cpu",
        weights_only=False,
        checkpoint_statistics=statistics,
        strict=True,
    ).eval()
    for name in ("trace_mean", "trace_std", "trace_min", "trace_rng"):
        assert torch.equal(getattr(module, name), getattr(restored, name))
    restored_trace = restored.prepare_batch(batch)[0]
    assert torch.equal(trace, restored_trace)
    with torch.no_grad():
        assert torch.equal(expected, restored.model(restored_trace))
    del checkpoint["state_dict"]["trace_std"]
    torch.save(checkpoint, path)
    with pytest.raises((ValueError, RuntimeError), match="statistic|trace_std"):
        SupervisedModule.load_from_checkpoint(
            path,
            map_location="cpu",
            weights_only=False,
            checkpoint_statistics={
                name: getattr(module, name)
                for name in ("trace_mean", "trace_std", "trace_min", "trace_rng")
            },
            strict=True,
        )
