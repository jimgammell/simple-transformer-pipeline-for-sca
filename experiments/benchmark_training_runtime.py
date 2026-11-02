"""Three native baseline epochs, including loaders, validation and checkpoint I/O.

Runtime step-budget policy is pending review; execution is disabled until resolved.
"""

# ruff: noqa: E402 -- process timing intentionally begins before heavyweight imports
import time

PROCESS_START = time.perf_counter()

import argparse
import math
from pathlib import Path

import lightning
import torch
import yaml
from init_things.directories import init_directories, load_directory_config
from utils.load_things import git_output, source_root, write_provenance
from utils.training_config import SupervisedTrainingConfig



class EpochTimer(lightning.Callback):
    def __init__(self):
        self.starts = []
        self.steps_per_epoch = None
        self.accumulation = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.starts.append(time.perf_counter())
        self.steps_per_epoch = math.ceil(trainer.num_training_batches / trainer.accumulate_grad_batches)
        self.accumulation = trainer.accumulate_grad_batches


def estimate_runtime(startup_first, second, third, total_steps, steps_per_epoch):
    full_epochs, tail_steps = divmod(total_steps, steps_per_epoch)
    steady = (second + third) / 2
    # Fractional final epoch approximation is explicit: validation/checkpoint fixed cost
    # is not separately identifiable from only three complete epochs.
    equivalent_epochs = full_epochs + tail_steps / steps_per_epoch
    estimate = startup_first + max(0, equivalent_epochs-1) * steady
    return dict(full_epochs=full_epochs, tail_optimizer_steps=tail_steps,
                steps_per_epoch=steps_per_epoch, equivalent_epochs=equivalent_epochs,
                estimated_full_recipe_seconds=estimate,
                whole_epoch_upper_estimate_seconds=startup_first + max(0, math.ceil(equivalent_epochs)-1)*steady,
                steady_epoch_seconds=steady,
                caveat="linear fractional-tail estimate, not exact; includes validation/checkpoint cost proportionally; late-training cache/metric behavior may differ")


def main():
    raise RuntimeError("Runtime benchmark pending review: choose the training-step budget and scheduler horizon before execution")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", required=True, type=Path)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--directory-config", required=True, type=Path)
    args = parser.parse_args()
    # Explicit machine data bindings; no environment or import-path overrides.
    directories = yaml.safe_load(args.directory_config.read_text())
    init_directories(directories, load_directory_config())
    raw = yaml.safe_load(args.config_file.read_text())
    raw["training"].update(compile=True, precision="bf16-mixed", seed=0)
    raw["commit_hash"] = git_output(source_root(), "rev-parse", "HEAD").strip()
    config = SupervisedTrainingConfig(**raw)
    args.dest.mkdir(parents=True, exist_ok=False)
    (args.dest / "config.yaml").write_text(yaml.safe_dump(config.model_dump()))
    (args.dest / "machine-directories.yaml").write_text(yaml.safe_dump(directories))
    write_provenance(args.dest, config.model_dump())
    timer = EpochTimer()
    torch.cuda.reset_peak_memory_stats()
    finished = time.perf_counter()
    if len(timer.starts) != 3 or timer.steps_per_epoch is None or timer.accumulation is None:
        raise RuntimeError(f"Expected three complete epochs; observed {len(timer.starts)}")
    first = timer.starts[1] - PROCESS_START
    second = timer.starts[2] - timer.starts[1]
    third = finished - timer.starts[2]
    result = dict(dataset=config.data.id, precision=config.training.precision, compile=True,
                  batch_size=config.training.batch_size, accumulation=timer.accumulation,
                  effective_batch_size=config.training.batch_size * timer.accumulation,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  total_recipe_steps=config.training.total_steps, benchmark_epochs=3,
                  scheduler_horizon_steps=config.training.total_steps,
                  startup_through_first_epoch_seconds=first,
                  epoch_two_seconds=second, epoch_three_plus_teardown_seconds=third,
                  total_process_seconds=finished-PROCESS_START,
                  timing_boundaries="process start to epoch 2 start; epoch 2 to epoch 3 start; epoch 3 start through native training return (validation, checkpoint and summary included)",
                  cache_policy="uses caller compiler and OS caches; no cache eviction or deletion",
                  **estimate_runtime(first, second, third, config.training.total_steps, timer.steps_per_epoch))
    (args.dest / "timing.yaml").write_text(yaml.safe_dump(result))


if __name__ == "__main__":
    main()
