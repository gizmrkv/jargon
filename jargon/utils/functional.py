import datetime
import os
import random
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def random_split(dataset: Tensor, proportions: Iterable[float]) -> List[Tensor]:
    """Split a dataset into multiple datasets randomly.

    Parameters
    ----------
    dataset : Tensor
        The dataset to split.
    proportions : Iterable[float]
        The proportions of the split datasets.

    Returns
    -------
    List[Tensor]
        The split datasets.

    Examples
    --------
    >>> dataset = torch.randn(100, 10)
    >>> d1, d2, d3 = random_split(dataset, [0.6, 0.2, 0.2])
    >>> len(d1), len(d2), len(d3)
    (60, 20, 20)
    """
    indices = np.random.permutation(len(dataset))

    proportions_sum = sum(proportions)
    split_sizes = [int(r / proportions_sum * len(dataset)) for r in proportions]
    split_sizes_argmax = split_sizes.index(max(split_sizes))
    split_sizes[split_sizes_argmax] += len(dataset) - sum(split_sizes)

    split_indices = np.split(indices, np.cumsum(split_sizes)[:-1])

    split_dataset = [dataset[torch.tensor(idx)] for idx in split_indices]

    return split_dataset


def init_weights(m: nn.Module) -> None:
    """Initialize the weights of the module.

    Parameters
    ----------
    m : nn.Module
        The module to initialize.

    Examples
    --------
    >>> net = nn.Sequential(nn.Linear(10, 32), nn.Linear(32, 5))
    >>> net.apply(init_weights)
    Sequential(
      (0): Linear(in_features=10, out_features=32, bias=True)
      (1): Linear(in_features=32, out_features=5, bias=True)
    )
    """
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.RNN, nn.LSTM, nn.GRU)):
        if isinstance(m.weight_ih_l0, torch.Tensor):
            nn.init.kaiming_uniform_(m.weight_ih_l0)
        if isinstance(m.weight_hh_l0, torch.Tensor):
            nn.init.kaiming_uniform_(m.weight_hh_l0)
        if isinstance(m.bias_ih_l0, torch.Tensor):
            nn.init.zeros_(m.bias_ih_l0)
        if isinstance(m.bias_hh_l0, torch.Tensor):
            nn.init.zeros_(m.bias_hh_l0)
    elif isinstance(m, nn.Embedding):
        nn.init.kaiming_uniform_(m.weight)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def fix_seed(seed: int) -> None:
    """
    Fix random seed for reproducibility of random operations.

    Args:
        seed (int): Seed value for random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_log_dir() -> Path:
    """make log directory and return its path

    Returns
    -------
    Path
        log directory path
    """
    dt = datetime.datetime.now()
    name = dt.strftime("%Y-%m-%d %H-%M-%S-%f")
    log_dir = Path("jargon-logs", name)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir
