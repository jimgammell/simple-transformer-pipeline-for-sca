import logging
import random
import shutil
from binascii import crc32
from typing import Any

import lightning
import numpy as np
import torch
from matplotlib import pyplot as plt

from . import directories
from .mpl_constants import DPI, WIDTH

__all__ = [
    "DPI",
    "OUTPUTS_ROOT",
    "SEED",
    "WIDTH",
    "directories",
    "init",
    "set_seed",
    "str_to_seed",
]

SEED: int = 0


def set_seed(seed: int | None = None):
    global SEED
    if seed is not None:
        assert isinstance(seed, int) and 0 <= seed < 0xFFFFFFFF
        SEED = seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    lightning.seed_everything(SEED)  # redundant but can't hurt


def str_to_seed(s: str) -> int:
    return (SEED + crc32(s.encode())) & 0xFFFFFFFF


def init(clargs: dict[str, Any] | None = None):
    logging.basicConfig(level=logging.INFO)
    latex_available = shutil.which("latex") is not None
    if not latex_available:
        logging.warning(
            "Latex installation not found. Falling back to non-Latex plotting, which might look ugly."
        )
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "figure.titlesize": 8,
            "font.family": "serif",
            "text.usetex": latex_available,
            **(
                {
                    "text.latex.preamble": r"\usepackage{times} \usepackage{amsmath} \usepackage{amssymb}"
                }
                if latex_available
                else {}
            ),
        }
    )
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    set_seed()
    directory_config = directories.load_directory_config()
    directories.init_directories(clargs, directory_config)


init()
OUTPUTS_ROOT = (
    directories.OUTPUTS_ROOT
)  # since global variables change after the init function is run
