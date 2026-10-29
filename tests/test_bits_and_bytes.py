import operator

import numpy as np
import pytest
import torch

from uncropped_transformers.models.building_blocks.bits_and_bytes import (
    SoftGF256Inv,
    SoftGF256Mult,
    SoftXOR,
)
from uncropped_transformers.utils import aes


def gf_mult(x, y):
    result = 0
    for _ in range(8):
        if y & 1:
            result ^= x
        x <<= 1
        if x & 0x100:
            x ^= 0x11B
        y >>= 1
    return result


def hard_logits(values):
    values = torch.as_tensor(values, dtype=torch.long)
    logits = torch.full((*values.shape, 256), -torch.inf, dtype=torch.float64)
    return logits.scatter_(-1, values.unsqueeze(-1), 0.0)


def sparse_probs(entries):
    probs = torch.zeros(256, dtype=torch.float64)
    for value, weight in entries:
        probs[value] = weight
    return probs / probs.sum()


def exact_probs(px, py, operation):
    result = torch.zeros(256, dtype=torch.float64)
    for x in px.nonzero().flatten().tolist():
        for y in py.nonzero().flatten().tolist():
            result[operation(x, y)] += px[x] * py[y]
    return result


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_aes_multiplication_matches_independent_polynomial_arithmetic():
    assert gf_mult(0x57, 0x13) == 0xFE
    values = np.arange(256, dtype=np.uint8)
    expected = np.array([[gf_mult(x, y) for y in range(256)] for x in range(256)])
    np.testing.assert_array_equal(
        aes.mult_gf256(values[:, None], values[None, :]), expected
    )
    assert aes.mult_gf256(np.array(0x57), np.array(0x13)) == 0xFE


@pytest.mark.parametrize(
    "module,operation", [(SoftXOR, operator.xor), (SoftGF256Mult, gf_mult)]
)
def test_hard_one_hot_arithmetic(module, operation):
    operation_module = module()
    for start in range(0, 256, 16):
        xs = list(range(start, start + 16))
        ys = [((x * 73) + 19) % 256 for x in xs]
        actual = operation_module(hard_logits(xs), hard_logits(ys))
        expected = hard_logits([operation(x, y) for x, y in zip(xs, ys, strict=True)])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for xs, ys in [([0] * 16, list(range(16))), (list(range(16)), [0] * 16)]:
        torch.testing.assert_close(
            operation_module(hard_logits(xs), hard_logits(ys)),
            hard_logits([operation(x, y) for x, y in zip(xs, ys, strict=True)]),
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        operation_module(hard_logits([0x57]), hard_logits([0x13])),
        hard_logits([operation(0x57, 0x13)]),
        rtol=0,
        atol=0,
    )


def test_all_nonzero_inverses_and_zero_convention():
    inverses = [0] + [
        next(y for y in range(1, 256) if gf_mult(x, y) == 1) for x in range(1, 256)
    ]
    np.testing.assert_array_equal(aes.GF256_INV, inverses)
    actual = SoftGF256Inv()(hard_logits(list(range(256))))
    torch.testing.assert_close(actual, hard_logits(inverses), rtol=0, atol=0)
    multiply = SoftGF256Mult()
    for start in range(1, 256, 16):
        xs = list(range(start, min(start + 16, 256)))
        torch.testing.assert_close(
            multiply(hard_logits(xs), hard_logits([inverses[x] for x in xs])),
            hard_logits([1] * len(xs)),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    "module,operation", [(SoftXOR, operator.xor), (SoftGF256Mult, gf_mult)]
)
def test_sparse_probabilities_match_exact_enumeration(module, operation):
    px = sparse_probs([(0, 2), (1, 3), (0x57, 5)])
    py = sparse_probs([(0, 7), (0x13, 2), (255, 1)])
    actual = module()(px.log(), py.log()).exp()
    torch.testing.assert_close(
        actual, exact_probs(px, py, operation), rtol=1e-12, atol=1e-14
    )
    torch.testing.assert_close(actual.sum(), torch.tensor(1.0, dtype=torch.float64))
    if module is SoftGF256Mult:
        torch.testing.assert_close(actual[0], px[0] + py[0] - px[0] * py[0])


@pytest.mark.parametrize(
    "module,operation", [(SoftXOR, operator.xor), (SoftGF256Mult, gf_mult)]
)
def test_batch_channel_broadcasting(module, operation):
    px = torch.stack(
        [
            sparse_probs([(0, 1), (7, 3)]),
            sparse_probs([(3, 2), (255, 1)]),
        ]
    ).unsqueeze(1)
    py = torch.stack(
        [
            sparse_probs([(0, 2), (9, 1)]),
            sparse_probs([(2, 1), (17, 4)]),
            sparse_probs([(0, 1), (33, 2)]),
            sparse_probs([(19, 3), (88, 1)]),
        ]
    ).reshape(2, 2, 256)
    actual = module()(px.log(), py.log()).exp()
    expected = torch.stack(
        [exact_probs(px[b, 0], py[b, c], operation) for b in range(2) for c in range(2)]
    ).reshape(2, 2, 256)
    assert actual.shape == (2, 2, 256)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("module", [SoftXOR, SoftGF256Mult])
def test_extreme_finite_logits_are_normalized_float64(module):
    x = torch.linspace(-10000, 10000, 256, dtype=torch.float64)
    y = x.flip(0) + 12345
    actual = module()(x, y)
    assert actual.dtype == torch.float64
    assert actual.device.type == "cpu"
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual.logsumexp(-1), torch.zeros((), dtype=torch.float64), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("module", [SoftXOR, SoftGF256Mult])
def test_finite_logits_have_finite_nonzero_gradients(module):
    generator = torch.Generator().manual_seed(71)
    x = (
        torch.randn(2, 1, 256, generator=generator, dtype=torch.float64) * 3
    ).requires_grad_()
    y = (
        torch.randn(2, 2, 256, generator=generator, dtype=torch.float64) * 3
    ).requires_grad_()
    actual = module()(x, y)
    weights = torch.randn(actual.shape, generator=generator, dtype=torch.float64)
    (actual * weights).sum().backward()
    for logits in (x, y):
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert torch.count_nonzero(logits.grad) > 0


def test_inverse_preserves_finite_logits_and_gradients():
    generator = torch.Generator().manual_seed(72)
    logits = torch.randn(
        2, 2, 256, generator=generator, dtype=torch.float64, requires_grad=True
    )
    inverse = SoftGF256Inv()
    actual = inverse(logits)
    torch.testing.assert_close(inverse(actual), logits, rtol=0, atol=0)
    torch.testing.assert_close(actual.logsumexp(-1), logits.logsumexp(-1))
    actual.square().sum().backward()
    torch.testing.assert_close(logits.grad, 2 * logits)
