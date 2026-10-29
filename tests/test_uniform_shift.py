import numpy as np
import torch
from test_model import module_factory as module_factory


def test_uniform_shift_independence_reflection_and_gating(module_factory):
    module = module_factory().train()
    module.config.random_roll_distribution = "uniform"
    module.config.random_roll_scale = 1.0
    module.trace_mean.zero_()
    module.trace_std.fill_(1)
    length = module.trace_mean.shape[-1]
    trace = torch.arange(length).float().repeat(96, 2, 1)
    labels = torch.zeros(96, 16, dtype=torch.long)
    metadata = {"key": labels.clone()}
    batch = trace, labels, metadata
    torch.manual_seed(42)
    shifts = torch.randint(-1, 2, (96,))
    torch.manual_seed(42)
    result, returned_labels, returned_metadata = module.prepare_batch(batch)
    assert returned_labels is labels
    assert returned_metadata is metadata
    assert set(shifts.tolist()) == {-1, 0, 1}
    assert torch.unique(result[:, 0], dim=0).shape[0] == 3
    for row, shift in enumerate(shifts.tolist()):
        padded = np.pad(trace[row].numpy(), ((0, 0), (1, 1)), mode="reflect")
        expected = torch.from_numpy(padded[:, 1 - shift : 1 - shift + length])
        torch.testing.assert_close(result[row], expected)
    torch.manual_seed(42)
    torch.testing.assert_close(module.prepare_batch(batch)[0], result)
    module.eval()
    torch.testing.assert_close(module.prepare_batch(batch)[0], trace)
    module.train()
    module.config.random_roll_scale = 0.0
    torch.testing.assert_close(module.prepare_batch(batch)[0], trace)


def test_gaussian_default_stays_batch_shared(module_factory):
    module = module_factory().train()
    assert module.config.random_roll_distribution == "gaussian"
    module.config.random_roll_scale = 1.0
    trace = torch.arange(module.trace_mean.numel()).float().repeat(16, 1, 1)
    labels = torch.zeros(16, 16, dtype=torch.long)
    np.random.seed(42)
    result = module.prepare_batch((trace, labels, {}))[0]
    assert torch.unique(result, dim=0).shape[0] == 1
