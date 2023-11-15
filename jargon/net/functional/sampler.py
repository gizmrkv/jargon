from typing import Tuple

from torch import Tensor, nn
from torch.distributions import Categorical


class Sampler(nn.Module):
    def __init__(self, model: nn.Module, num_elems: int, num_attrs: int) -> None:
        super().__init__()
        self.model = model
        self.num_elems = num_elems
        self.num_attrs = num_attrs

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        logits = self.model(x)
        logits = logits.reshape(-1, self.num_attrs, self.num_elems)
        if self.training:
            distr = Categorical(logits=x)
            x = distr.sample()
        else:
            x = logits.argmax(dim=-1)
        return x, logits
