from typing import Any, Dict

from torch import Tensor
from torch.distributions import Categorical

from jargon.core import Batch


class Metrics:
    def __init__(self, num_elems: int, num_attrs: int) -> None:
        self.num_elems = num_elems
        self.num_attrs = num_attrs

    def __call__(self, batch: Batch) -> Dict[str, Any]:
        output: Tensor = batch.output  # type: ignore
        target: Tensor = batch.target  # type: ignore

        acc_flag = (
            output.reshape(-1, self.num_attrs, self.num_elems).argmax(-1) == target
        )
        acc_comp = acc_flag.all(-1).float()
        acc_part = acc_flag.float()

        distr = Categorical(logits=output)
        entropy: Tensor = distr.entropy()

        metrics = {
            "acc/comp.mean": acc_comp.mean().item(),
            "acc/comp.std": acc_comp.std().item(),
            "acc/part.mean": acc_part.mean().item(),
            "acc/part.std": acc_part.std().item(),
            "entropy.mean": entropy.mean().item(),
            "entropy.std": entropy.std().item(),
        }
        return metrics
