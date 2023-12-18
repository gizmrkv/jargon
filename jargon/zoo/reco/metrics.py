from typing import Any, Dict

from torch import Tensor
from torch.distributions import Categorical

from jargon.core import Batch


class Metrics:
    def __init__(self, num_elems: int, num_attrs: int) -> None:
        self.num_elems = num_elems
        self.num_attrs = num_attrs

    def __call__(self, batch: Batch) -> Dict[str, Any]:
        # [batch, num_attrs, num_elems; float]
        output = batch.get_tensor("output")
        # [batch, num_attrs; int]
        target = batch.get_tensor("target")

        # [batch, num_attrs; bool]
        acc_flag = output.argmax(-1) == target
        # [batch; float]
        acc_comp = acc_flag.all(-1).float()
        # [batch; float]
        acc_part = acc_flag.float().mean(-1)

        metrics = {
            "acc/comp.mean": acc_comp.mean().item(),
            "acc/part.mean": acc_part.mean().item(),
            # "acc/comp.std": acc_comp.std().item(),
            # "acc/part.std": acc_part.std().item(),
        }
        return metrics
