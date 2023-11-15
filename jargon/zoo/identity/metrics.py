from typing import Any, Callable, Dict

from torch import Tensor
from torch.distributions import Categorical

from jargon.core import Batch


class Metrics:
    def __init__(
        self, num_elems: int, num_attrs: int, loss_fn: Callable[[Batch], Tensor]
    ) -> None:
        self.num_elems = num_elems
        self.num_attrs = num_attrs
        self.loss_fn = loss_fn

    def __call__(self, batch: Batch) -> Dict[str, Any]:
        output: Tensor = batch.output  # type: ignore
        target: Tensor = batch.target  # type: ignore

        loss = self.loss_fn(batch)

        acc_flag = (
            output.reshape(-1, self.num_attrs, self.num_elems).argmax(-1) == target
        )
        acc_comp = acc_flag.all(-1).float()
        acc_part = acc_flag.float()

        distr = Categorical(logits=output)
        entropy: Tensor = distr.entropy()

        metrics = {
            "loss.mean": loss.mean().item(),
            "acc_comp.mean": acc_comp.mean().item(),
            "acc_part.mean": acc_part.mean().item(),
            "entropy.mean": entropy.mean().item(),
            "loss.std": loss.std().item(),
            "acc_comp.std": acc_comp.std().item(),
            "acc_part.std": acc_part.std().item(),
            "entropy.std": entropy.std().item(),
        }
        return metrics
