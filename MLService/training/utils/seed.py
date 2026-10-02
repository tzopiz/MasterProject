"""
Reproducibility helpers for training (torch / numpy / random + DataLoader workers).
"""

from __future__ import annotations

import os
import random
from functools import partial
from typing import Callable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed RNGs and deterministic cuDNN convolutions, not cross-device bitwise identity."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int, base_seed: int) -> None:
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def make_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    """Distinct reproducible worker seeds, serializable for multiprocessing spawn."""
    return partial(_seed_worker, base_seed=base_seed)
