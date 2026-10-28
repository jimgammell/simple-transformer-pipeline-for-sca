from collections.abc import Callable, Sized
from functools import partial
from typing import Literal

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from uncropped_transformers.training.supervised_lightning_module import SupervisedModule

ATTRIBUTION_METHOD = Literal[
    "gradvis",
    "input-x-gradient",
]


class Attributor:
    def __init__(self, module: SupervisedModule):
        self.module = module
        self.module.eval()

    def _get_loss(
        self, trace: torch.Tensor, target: torch.Tensor, head_idx: int | None = None
    ) -> torch.Tensor:
        logits = self.module.model(trace)
        loss = self.module.compute_loss(logits, target)
        if head_idx is not None:
            loss = loss[:, head_idx]
        return loss

    def compute_gradvis(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]],
        smoothing_std: float = 0.0,
        smoothing_count: int = 1,
    ) -> torch.Tensor:
        _trace, target, intermediate_values = self.module.prepare_batch(batch)
        batch_size, *_, feature_count = _trace.shape
        *_, head_count = target.shape
        grads = torch.zeros(
            (batch_size, head_count, feature_count),
            dtype=_trace.dtype,
            device=_trace.device,
        )
        for _ in range(smoothing_count):
            for head_idx in range(head_count):
                trace = _trace.clone().detach().requires_grad_(True)
                if smoothing_std > 0:
                    trace = trace + smoothing_std * torch.randn_like(trace)
                loss = self._get_loss(trace, target, head_idx=head_idx)
                grad = torch.autograd.grad(outputs=loss.sum(), inputs=trace)[0]
                grads[:, head_idx, :] += grad.detach().view(batch_size, feature_count)
        attribution = grads.abs() / smoothing_count
        return attribution

    def compute_input_x_gradient(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        _trace, target, _ = self.module.prepare_batch(batch)
        batch_size, *_, feature_count = _trace.shape
        *_, head_count = target.shape
        attrs = torch.zeros(
            (batch_size, head_count, feature_count),
            dtype=_trace.dtype,
            device=_trace.device,
        )
        for head_idx in range(head_count):
            trace = _trace.clone().detach().requires_grad_(True)
            loss = self._get_loss(trace, target, head_idx=head_idx)
            grad = torch.autograd.grad(outputs=loss.sum(), inputs=trace)[0]
            attrs[:, head_idx, :] = (
                (grad * trace).detach().view(batch_size, feature_count).abs()
            )
        return attrs

    def aggregate_attributions(
        self,
        attr_fn: Callable[[torch.Tensor], torch.Tensor],
        dataloader: DataLoader,
        show_progress_bar: bool = False,
    ) -> torch.Tensor:
        if show_progress_bar:
            progress_bar = tqdm(
                total=len(dataloader.dataset)
                if isinstance(dataloader.dataset, Sized)
                else None
            )
        attrs = None
        old_count = 0
        for batch in dataloader:
            attr = attr_fn(batch)
            if attrs is None:
                *_, head_count, feature_count = attr.shape
                dtype = self.module.dtype
                assert isinstance(dtype, torch.dtype)
                attrs = torch.zeros(
                    (head_count, feature_count),
                    dtype=dtype,
                    device=self.module.device,
                )
            new_count = old_count + len(attr)
            attrs = (old_count / new_count) * attrs + (
                len(attr) / new_count
            ) * attr.mean(dim=0)
            old_count = new_count
            if show_progress_bar:
                progress_bar.update(len(attr))
        if attrs is None:
            raise ValueError("Cannot aggregate attributions from an empty dataloader")
        return attrs

    def __call__(
        self,
        attr_method: ATTRIBUTION_METHOD,
        dataloader: DataLoader,
        show_progress_bar: bool = False,
        **attr_kwargs,
    ) -> torch.Tensor:
        if attr_method == "gradvis":
            attr_fn = partial(self.compute_gradvis, **attr_kwargs)
        elif attr_method == "input-x-gradient":
            attr_fn = partial(self.compute_input_x_gradient, **attr_kwargs)
        else:
            raise ValueError(f"Unsupported attribution method: {attr_method!r}")
        attr = self.aggregate_attributions(
            attr_fn, dataloader, show_progress_bar=show_progress_bar
        )
        return attr
