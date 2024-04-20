from jargon.utils.torchdict import TensorDict

from .loss import Loss


class LossSelector:
    def __init__(self, **select_loss: Loss):
        self.select_loss = select_loss

    def forward(self, batches: TensorDict) -> TensorDict:
        return TensorDict(
            **{
                agent: loss(batches.get_tensor_dict(agent))
                for agent, loss in self.select_loss.items()
            }
        )
