from abc import ABC, abstractmethod
from typing import Optional, Tuple

from torch import Tensor, nn

from jargon.utils.torchdict import TensorDict


class Net(ABC, nn.Module):
    @abstractmethod
    def forward(
        self, input: Tensor, info: Optional[TensorDict] = None
    ) -> Tuple[Tensor, Optional[TensorDict]]:
        pass
