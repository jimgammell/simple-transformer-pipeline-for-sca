from typing import Dict, Literal, Tuple

import torch

LEAKAGE_MODEL = Literal["bit", "id", "hw"]
PHASE = Literal["train", "val", "test"]
PREPROCESSING = Literal["standardize", "normalize"]

BATCH = Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]
