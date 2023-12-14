from typing import Any, Dict

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
        # [batch, num_attrs, num_elems; float]
        output = batch.get_tensor("output")
        # [batch, num_attrs; int]
        target = batch.get_tensor("target")

        # [batch * num_attrs, num_elems; float]
        output = output.reshape(-1, self.num_elems)
        # [batch * num_attrs; int]
        target = target.reshape(-1)
        # [batch * num_attrs; float]
        loss = F.cross_entropy(output, target, reduction="none")
        # [batch; float]
        loss = loss.reshape(-1, self.num_attrs).mean(-1)
        return loss

    def metrics(self, batch: Batch) -> Dict[str, Any]:
        loss = self(batch)
        return {"loss/mean": loss.mean().item(), "loss/std": loss.std().item()}
