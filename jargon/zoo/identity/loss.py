from torch import Tensor
from torch.nn import functional as F

from jargon.core import Batch


class Loss:
    def __init__(
        self,
        num_elems: int,
        num_attrs: int,
    ) -> None:
        self.num_elems = num_elems
        self.num_attrs = num_attrs

    def __call__(self, batch: Batch) -> Tensor:
        output: Tensor = batch.output  # type: ignore
        target: Tensor = batch.target  # type: ignore

        output = output.reshape(-1, self.num_elems)
        target = target.reshape(-1)
        loss = F.cross_entropy(output, target, reduction="none")
        loss = loss.reshape(-1, self.num_attrs).mean(-1)
        return loss
