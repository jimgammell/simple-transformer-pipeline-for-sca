import shutil
from pathlib import Path
from typing import Literal

import lightning
import pandas as pd
import torch
import yaml
from lightning import LightningModule, Trainer
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader


def _is_training_complete(ckpt_path: Path, total_steps: int) -> bool:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt.get("global_step", 0) >= total_steps


def _write_training_summary(dest: Path, selected_checkpoint: Path) -> None:
    checkpoint = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
    summary = dict(
        selected_checkpoint=selected_checkpoint.name,
        selected_global_step=int(checkpoint["global_step"]),
    )
    (dest / "training_summary.yaml").write_text(yaml.safe_dump(summary))


def train_supervised_model(
    *,
    dest: Path,
    training_module: LightningModule,
    train_loader: DataLoader,
    val_loader: DataLoader,
    total_steps: int,
    grad_clip_val: float | None = None,
    accumulate_grad_batches: int = 1,
    early_stop_metric: str | None = "val_rank",
    early_stop_mode: Literal["min", "max"] = "max",
    aux_callbacks: list[lightning.Callback] | None = None,
    precision: _PRECISION_INPUT = "bf16-mixed",
):
    latest_ckpt_path = dest / "latest.ckpt"
    if early_stop_metric is not None:
        best_ckpt_name = f"best_{early_stop_metric}".replace("/", "_")
    else:
        best_ckpt_name = "latest"

    # Skip training entirely if already complete
    if latest_ckpt_path.exists() and _is_training_complete(
        latest_ckpt_path, total_steps
    ):
        print(
            f"Training already complete (global_step >= {total_steps}). Skipping training."
        )
    else:
        callbacks = []
        if early_stop_metric is not None:
            early_stop_callback = ModelCheckpoint(
                monitor=early_stop_metric,
                mode=early_stop_mode,
                save_top_k=1,
                dirpath=dest,
                filename=best_ckpt_name,
                verbose=True,
                enable_version_counter=False,
                save_on_train_epoch_end=True,
            )
            callbacks.append(early_stop_callback)
        final_checkpoint_callback = ModelCheckpoint(
            dirpath=dest,
            filename="latest",
            enable_version_counter=False,
            every_n_epochs=1,
        )
        callbacks.append(final_checkpoint_callback)
        if aux_callbacks is not None:
            callbacks += aux_callbacks

        # Back up existing metrics before CSVLogger overwrites them on resume
        metrics_csv = dest / "metrics.csv"
        metrics_backup = dest / "metrics.backup.csv"
        is_resuming = latest_ckpt_path.exists()
        if is_resuming and metrics_csv.exists():
            shutil.copy2(metrics_csv, metrics_backup)

        logger = CSVLogger(save_dir=dest, name="", version="")
        trainer = Trainer(
            accelerator="gpu",
            precision=precision,
            logger=logger,
            callbacks=callbacks,
            max_steps=total_steps,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=grad_clip_val,
            default_root_dir=dest,
        )
        if not is_resuming:
            latest_ckpt_path = None
        trainer.fit(
            training_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=latest_ckpt_path,
            weights_only=False,
        )

        # Merge old and new metrics after resume
        if metrics_backup.exists():
            old_df = pd.read_csv(metrics_backup)
            new_df = pd.read_csv(metrics_csv)
            merged = pd.concat([old_df, new_df], ignore_index=True)
            merged.to_csv(metrics_csv, index=False)
            metrics_backup.unlink()

    _write_training_summary(dest, dest / f"{best_ckpt_name}.ckpt")
