from typing import Literal

import torch

LEAKAGE_MODEL = Literal["bit", "id", "hw"]
PHASE = Literal["train", "val", "test"]
PREPROCESSING = Literal["standardize", "normalize"]

BATCH = tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
