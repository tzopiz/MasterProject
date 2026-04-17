"""
Reproducibility helpers for training (torch / numpy / random + DataLoader workers).
"""

from __future__ import annotations

import os
import random
from typing import Callable, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix seeds for stdlib random, NumPy, and torch (CPU + CUDA if available)."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_worker_init_fn(base_seed: int) -> Optional[Callable[[int], None]]:
    """
    Factory for DataLoader worker_init_fn.

    Each worker gets a distinct seed derived from base_seed so augmentations
    differ across workers while remaining reproducible for a given base_seed.
    """

    def _worker_init(worker_id: int) -> None:
        s = base_seed + worker_id
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

    return _worker_init
