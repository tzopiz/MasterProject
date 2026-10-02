"""The configured DataLoader initializer must survive spawn serialization."""

import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.utils.seed import make_worker_init_fn


def test_worker_seed_can_be_pickled_and_replayed():
    init = pickle.loads(pickle.dumps(make_worker_init_fn(42)))
    init(1)
    first = (random.random(), np.random.random(), torch.rand(1).item())
    init(1)
    assert first == (random.random(), np.random.random(), torch.rand(1).item())
    init(2)
    assert first != (random.random(), np.random.random(), torch.rand(1).item())


def test_seed_configures_cudnn_reproducibility():
    from training.utils.seed import set_seed

    previous = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        set_seed(42)
        assert not torch.backends.cudnn.benchmark
        assert torch.backends.cudnn.deterministic
    finally:
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = previous


def test_spawn_loader_runs_with_configured_initializer():
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(
        TensorDataset(torch.arange(4)),
        batch_size=2,
        num_workers=1,
        multiprocessing_context="spawn",
        worker_init_fn=make_worker_init_fn(42),
    )
    assert torch.cat([row[0] for row in loader]).tolist() == [0, 1, 2, 3]
