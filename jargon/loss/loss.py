from abc import ABC, abstractmethod

from torch import Tensor, nn

from jargon.utils.torchdict import TensorDict


class Loss(ABC, nn.Module):
    @abstractmethod
    def forward(self, batch: TensorDict) -> Tensor:
        pass
