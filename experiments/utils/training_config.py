from typing import Annotated, Any, Literal

from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from pydantic import BaseModel, Field, StrictBool, field_validator, model_validator

from uncropped_transformers.datasets import DATASET
from uncropped_transformers.models.model import (
    FNN_STYLE,
    GREY_BOX_HEAD,
    HEAD,
    POOLING,
    POSITION_EMBEDDING,
)
from uncropped_transformers.training.hyperparameter_tuning import ParamConfig
from uncropped_transformers.training.supervised_lightning_module import (
    LEAKAGE_MODEL,
    PREPROCESSING,
)


class DataConfig(BaseModel):
    id: DATASET
    target_byte: list[int]
    target_variable: str | list[str]
    preprocessing: PREPROCESSING
    random_roll_scale: Annotated[float, Field(ge=0)]
    random_roll_distribution: Literal["gaussian", "uniform"] = "gaussian"
    random_lpf_scale: Annotated[float, Field(ge=0)]
    val_prop: Annotated[float, Field(gt=0, lt=1)]

    @field_validator("target_byte", mode="before")
    @classmethod
    def coerce_target_byte_to_list(cls, v: Any) -> Any:
        if isinstance(v, int):
            return [v]
        return v


class TrainingConfig(BaseModel):
    total_steps: Annotated[int, Field(gt=0)]
    lr_warmup_frac: Annotated[float, Field(ge=0, le=1)]
    lr_const_frac: Annotated[float, Field(ge=0, le=1)]
    batch_size: Annotated[int, Field(gt=0)]
    base_lr: Annotated[float, Field(gt=0)]
    lr_decay_multiplier: Annotated[float, Field(ge=0, le=1)]
    weight_decay: Annotated[float, Field(ge=0)]
    label_smoothing: Annotated[float, Field(ge=0, lt=1)]
    mixup_alpha: Annotated[float, Field(ge=0)]
    additive_gaussian_noise: Annotated[float, Field(ge=0)]
    grad_clip_val: Annotated[float, Field(gt=0)] | None = None
    accumulate_grad_batches: Annotated[int, Field(gt=0)]
    early_stop_metric: str
    early_stop_mode: Literal["min", "max"]
    seed: int
    compile: StrictBool
    num_workers: Annotated[int, Field(ge=0)]
    precision: _PRECISION_INPUT = "bf16-mixed"
    cudnn_benchmark: StrictBool = True
    float32_matmul_precision: Literal["highest", "high", "medium"] = "high"


class MTDConfig(BaseModel):
    attack_count: Annotated[int, Field(gt=0)]
    traces_per_attack: Annotated[int, Field(gt=0)]


class ModelConfig(BaseModel):
    grey_box_head: GREY_BOX_HEAD | None
    position_embedding: POSITION_EMBEDDING
    pooling: POOLING
    head: HEAD
    fnn_style: FNN_STYLE
    patch_size: Annotated[int, Field(gt=0)]
    use_fourier_embed: StrictBool
    fourier_embed_num_bands: Annotated[int, Field(gt=0)] | None
    fourier_embed_sigma: Annotated[float, Field(gt=0)] | None
    embedding_dim: Annotated[int, Field(gt=0)]
    expansion_factor: Annotated[int, Field(gt=0)]
    trunk_blocks: Annotated[int, Field(gt=0)]
    head_count: Annotated[int, Field(gt=0)] | None
    input_dropout_rate: Annotated[float, Field(ge=0)]
    input_droppatch_rate: Annotated[float, Field(ge=0)]
    hidden_dropout_rate: Annotated[float, Field(ge=0)]
    use_bias: StrictBool
    leakage_model: LEAKAGE_MODEL

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "ModelConfig":
        if self.use_fourier_embed:
            assert self.fourier_embed_num_bands is not None, (
                "fourier_embed_num_bands required when use_fourier_embed=True"
            )
            assert self.fourier_embed_sigma is not None, (
                "fourier_embed_sigma required when use_fourier_embed=True"
            )
        else:
            assert self.fourier_embed_num_bands is None, (
                "fourier_embed_num_bands must be null when use_fourier_embed=False"
            )
            assert self.fourier_embed_sigma is None, (
                "fourier_embed_sigma must be null when use_fourier_embed=False"
            )
        if self.head_count is not None:
            assert self.embedding_dim % self.head_count == 0, (
                f"embedding_dim ({self.embedding_dim}) must be divisible by head_count ({self.head_count})"
            )
        return self


class SupervisedTrainingConfig(BaseModel):
    data: DataConfig
    training: TrainingConfig
    mtd: MTDConfig
    model: ModelConfig
    search_space: dict[str, dict[str, ParamConfig]]
    commit_hash: str | None = None
