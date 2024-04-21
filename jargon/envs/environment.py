from abc import ABC, abstractmethod
from typing import Iterator, Set

from torch import nn

from jargon.utils.torchdict import TensorDict


class Environment(ABC, nn.Module):
    agents: Set[str]

    modes: Set[str]
    mode: str

    @abstractmethod
    def rollout(self) -> Iterator[TensorDict]:
        pass
