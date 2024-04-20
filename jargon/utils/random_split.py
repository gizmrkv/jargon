from typing import Iterable, List

import numpy as np
import torch
from torch import Tensor


def random_split(dataset: Tensor, proportions: Iterable[float]) -> List[Tensor]:
    indices = np.random.permutation(len(dataset))

    proportions_sum = sum(proportions)
    split_sizes = [int(r / proportions_sum * len(dataset)) for r in proportions]
    split_sizes_argmax = split_sizes.index(max(split_sizes))
    split_sizes[split_sizes_argmax] += len(dataset) - sum(split_sizes)

    split_indices = np.split(indices, np.cumsum(split_sizes)[:-1])

    split_dataset = [dataset[torch.tensor(idx)] for idx in split_indices]

    return split_dataset
