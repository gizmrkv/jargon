from torch import Tensor
from torch.nn import functional as F

from jargon.utils.torchdict import TensorDict

from .loss import Loss


class TextCrossEntropyLoss(Loss):
    def __init__(self, vocab_size: int, seq_length: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_length = seq_length

    def forward(self, batch: TensorDict) -> Tensor:
        target = batch.get_tensor("target")
        info = batch.get_tensor_dict("info")
        logits = info.get_tensor("logits")

        target = target.view(-1)
        logits = logits.view(-1, self.vocab_size)
        loss = F.cross_entropy(logits, target, reduction="none")
        loss = loss.view(-1, self.seq_length).mean(-1)
        return loss
