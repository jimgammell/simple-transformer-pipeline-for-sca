"""
Benchmark transformer compute cost vs. key hyperparameters.

Sweeps patch_count (with fixed input_length=2^18, so patch_size varies inversely),
layer_count, and embedding_dim, measuring per-forward+backward-pass FLOPs, wall-clock
time, parameter count, and peak VRAM.

With fixed input_length, increasing patch_count shrinks patch_size: the transformer
sees a longer sequence (more FLOPs) but the patchifier projection has fewer parameters.

Usage:
    python experiments/benchmark_cost_scaling.py [--dest PATH]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from lightning.fabric import Fabric
from lightning.fabric.utilities.throughput import measure_flops as lightning_measure_flops
from init_things import OUTPUTS_ROOT

from uncropped_transformers.models.model import Model

# ── Fixed benchmark parameters ────────────────────────────────────────────────
BATCH_SIZE = 256
FIXED_INPUT_LEN = 2**18  # 262 144; patch_size = FIXED_INPUT_LEN // patch_count
OUTPUT_COUNT = 16  # number of output variables predicted (as in practice)
OUTPUT_DIM = 256
EXPANSION_FACTOR = 4
N_WARMUP = 3
N_ITERS = 10

# ── Base config ───────────────────────────────────────────────────────────────
BASE_PATCH_COUNT = 64
BASE_LAYER_COUNT = 8
BASE_EMBEDDING_DIM = 512

# ── Sweeps (factors of 2 around base) ────────────────────────────────────────
PATCH_COUNTS = [8, 16, 32, 64, 128, 256]
LAYER_COUNTS = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
EMBEDDING_DIMS = [128, 256, 384, 512, 640, 768, 896, 1024, 1280, 1536]


# ── Model helpers ─────────────────────────────────────────────────────────────


def build_model(patch_count: int, layer_count: int, embedding_dim: int) -> Model:
    patch_size = FIXED_INPUT_LEN // patch_count
    assert patch_size % 2 == 0, (
        f"patch_size={patch_size} must be even (got patch_count={patch_count})"
    )
    model = Model(
        input_length=FIXED_INPUT_LEN,
        output_dim=OUTPUT_DIM,
        output_count=OUTPUT_COUNT,
        grey_box_head=None,
        position_embedding="rope",
        pooling="token",
        head="tied",
        fnn_style="mlp",
        patch_size=patch_size,
        use_fourier_embed=False,
        fourier_embed_num_bands=None,
        fourier_embed_sigma=None,
        embedding_dim=embedding_dim,
        expansion_factor=EXPANSION_FACTOR,
        trunk_blocks=layer_count,
        head_count=None,  # defaults to embedding_dim // 64
        input_dropout_rate=0.0,
        input_droppatch_rate=0.0,
        hidden_dropout_rate=0.0,
        use_bias=False,
    )
    return model.train()


def make_input(patch_count: int) -> torch.Tensor:
    return torch.randn(BATCH_SIZE, 1, FIXED_INPUT_LEN)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _forward_backward(model: torch.nn.Module, x: torch.Tensor, fabric: Fabric) -> torch.Tensor:
    out = model(x)
    loss = out.sum()
    fabric.backward(loss)
    model.zero_grad(set_to_none=True)
    return loss


# ── Metric measurements ───────────────────────────────────────────────────────


def measure_flops(model: torch.nn.Module, x: torch.Tensor, fabric: Fabric) -> int:
    """Forward/backward matrix and convolution FLOPs; multiply-add = 2.

    Registered fused-attention backward formulas include recomputation.
    Softmax, normalization, activation, elementwise, RNG and optimizer work
    are excluded. Count eager execution separately from compiled timing.
    """
    flops = lightning_measure_flops(model, lambda: _forward_backward(model, x, fabric))
    if flops <= 0:
        raise RuntimeError("FLOP coverage is empty")
    return int(flops)


def measure_vram_mb(model: torch.nn.Module, x: torch.Tensor, fabric: Fabric) -> float:
    """Peak VRAM (MB) over one forward + backward pass."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _forward_backward(model, x, fabric)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def measure_wall_time_ms(model: torch.nn.Module, x: torch.Tensor, fabric: Fabric) -> float:
    """Median wall-clock time (ms) per forward + backward pass over N_ITERS."""
    for _ in range(N_WARMUP):
        _forward_backward(model, x, fabric)
        torch.cuda.synchronize()

    times = []
    for _ in range(N_ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _forward_backward(model, x, fabric)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000)


# ── Per-config orchestration ──────────────────────────────────────────────────


def benchmark_config(patch_count: int, layer_count: int, embedding_dim: int, compile_model: bool = True) -> dict:
    patch_size = FIXED_INPUT_LEN // patch_count
    base = {
        "patch_count": patch_count,
        "layer_count": layer_count,
        "embedding_dim": embedding_dim,
        "patch_size": patch_size,
    }
    try:
        torch.manual_seed(0)
        fabric = Fabric(accelerator="cuda", devices=1, precision="bf16-mixed")
        eager_model = build_model(patch_count, layer_count, embedding_dim)
        model = fabric.setup_module(eager_model)
        x = fabric.to_device(make_input(patch_count))
        params = count_params(model)
        flops = measure_flops(model, x, fabric)
        if compile_model:
            eager_model.compile()
        wall_ms = measure_wall_time_ms(model, x, fabric)
        vram_mb = measure_vram_mb(model, x, fabric)

        del model, eager_model, x
        torch.cuda.empty_cache()

        return {
            **base,
            "param_count": params,
            "flops": flops,
            "wall_time_ms": wall_ms,
            "vram_mb": vram_mb,
            "oom": False,
        }

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            **base,
            "param_count": None,
            "flops": None,
            "wall_time_ms": float("nan"),
            "vram_mb": float("nan"),
            "oom": True,
        }


def _fmt(r: dict) -> str:
    if r["oom"]:
        return "OOM"
    return (
        f"params={r['param_count']:>12,}  "
        f"flops={r['flops']:.3e}  "
        f"vram={r['vram_mb']:>8.1f} MB  "
        f"time={r['wall_time_ms']:>7.1f} ms"
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    output_dir: Path | None = args.dest
    if output_dir is None:
        output_dir = OUTPUTS_ROOT / "compute_benchmark"
    assert isinstance(output_dir, Path)
    output_dir.mkdir(exist_ok=True, parents=True)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    results = []

    # ── Sweep: patch_count ────────────────────────────────────────────────────
    print(
        f"=== Sweeping patch_count  "
        f"(layer_count={BASE_LAYER_COUNT}, embedding_dim={BASE_EMBEDDING_DIM}) ==="
    )
    for pc in PATCH_COUNTS:
        ps = FIXED_INPUT_LEN // pc
        print(f"  patch_count={pc:>4}  patch_size={ps:>6} ... ", end="", flush=True)
        r = benchmark_config(pc, BASE_LAYER_COUNT, BASE_EMBEDDING_DIM, args.compile)
        r["sweep_var"] = "patch_count"
        results.append(r)
        print(_fmt(r))

    # ── Sweep: layer_count ────────────────────────────────────────────────────
    print(
        f"\n=== Sweeping layer_count  "
        f"(patch_count={BASE_PATCH_COUNT}, embedding_dim={BASE_EMBEDDING_DIM}) ==="
    )
    for lc in LAYER_COUNTS:
        print(f"  layer_count={lc:>3} ... ", end="", flush=True)
        r = benchmark_config(BASE_PATCH_COUNT, lc, BASE_EMBEDDING_DIM, args.compile)
        r["sweep_var"] = "layer_count"
        results.append(r)
        print(_fmt(r))

    # ── Sweep: embedding_dim ──────────────────────────────────────────────────
    print(
        f"\n=== Sweeping embedding_dim  "
        f"(patch_count={BASE_PATCH_COUNT}, layer_count={BASE_LAYER_COUNT}) ==="
    )
    for ed in EMBEDDING_DIMS:
        print(f"  embedding_dim={ed:>5} ... ", end="", flush=True)
        r = benchmark_config(BASE_PATCH_COUNT, BASE_LAYER_COUNT, ed, args.compile)
        r["sweep_var"] = "embedding_dim"
        results.append(r)
        print(_fmt(r))

    # ── Pack results into arrays and save as .npz ─────────────────────────────
    def _arr(key, dtype=None):
        vals = [r[key] for r in results]
        return np.array(vals, dtype=dtype)

    out_path = output_dir / "results.npz"
    np.savez(
        out_path,
        compile=np.array(args.compile),
        precision=np.array("bf16-mixed"),
        flop_convention=np.array("matrix/convolution forward+backward; multiply-add=2; excludes elementwise and optimizer; eager accounting, warmed execution timing"),
        sweep_var=_arr("sweep_var"),
        patch_count=_arr("patch_count", np.int64),
        layer_count=_arr("layer_count", np.int64),
        embedding_dim=_arr("embedding_dim", np.int64),
        patch_size=_arr("patch_size", np.int64),
        param_count=_arr("param_count", np.float64),  # float to allow NaN on OOM
        flops=_arr("flops", np.float64),
        wall_time_ms=_arr("wall_time_ms", np.float64),
        vram_mb=_arr("vram_mb", np.float64),
        oom=_arr("oom", bool),
    )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
