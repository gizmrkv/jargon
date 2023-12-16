from typing import Any, Dict

import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from jargon.core import Batch
from jargon.net.loss import pg_loss


class Loss:
    def __init__(
        self,
        num_elems: int,
        num_attrs: int,
        vocab_size: int,
        max_len: int,
        entropy_loss_weight: float = 0.0,
        length_loss_weight: float = 0.0,
    ) -> None:
        self.num_elems = num_elems
        self.num_attrs = num_attrs
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.entropy_loss_weight = entropy_loss_weight
        self.length_loss_weight = length_loss_weight

    def receiver_loss(self, batch: Batch) -> Tensor:
        # [batch, num_attrs, num_elems; float]
        logits = batch.get_tensor("output_logits")
        # [batch, num_attrs; int]
        target = batch.get_tensor("target")

        # [batch * num_attrs, num_elems; float]
        logits = logits.reshape(-1, self.num_elems)
        # [batch * num_attrs; int]
        target = target.reshape(-1)
        # [batch * num_attrs; float]
        loss_r = F.cross_entropy(logits, target, reduction="none")
        # [batch; float]
        loss_r = loss_r.reshape(-1, self.num_attrs).mean(-1)

        return loss_r

    def sender_loss(self, batch: Batch, receiver_loss: Tensor) -> Tensor:
        # [batch, max_len; int]
        message = batch.get_tensor("message")
        # [batch, max_len, vocab_size; float]
        msg_logits = batch.get_tensor("message_logits")
        # [batch, max_len; int]
        msg_mask = batch.get_tensor("message_mask")

        distr = Categorical(logits=msg_logits)
        # [batch, max_len; float]
        log_prob = distr.log_prob(message)
        # [batch, max_len; float]
        log_prob = log_prob * msg_mask
        # [batch; float]
        log_prob = log_prob.sum(-1)

        # [batch; float]
        reward = -receiver_loss.detach()
        # [batch; float]
        loss_s = pg_loss(log_prob, reward)
        return loss_s

    def entropy_loss(self, batch: Batch) -> Tensor:
        # [batch, max_len, vocab_size; float]
        msg_logits = batch.get_tensor("message_logits")
        # [batch, max_len; int]
        msg_mask = batch.get_tensor("message_mask")

        distr = Categorical(logits=msg_logits)
        # [batch, max_len; float]
        loss_ent = distr.entropy() * msg_mask
        # [batch; float]
        loss_ent = loss_ent.mean(-1)

        return -loss_ent

    def length_loss(self, batch: Batch) -> Tensor:
        # [batch, max_len; int]
        message = batch.get_tensor("message")
        # [batch, max_len, vocab_size; float]
        msg_logits = batch.get_tensor("message_logits")
        # [batch; int]
        msg_length = batch.get_tensor("message_length")
        # [batch, max_len; int]
        msg_mask = batch.get_tensor("message_mask")

        distr = Categorical(logits=msg_logits)
        # [batch, max_len; float]
        log_prob = distr.log_prob(message)
        # [batch, max_len; float]
        log_prob = log_prob * msg_mask
        # [batch; float]
        log_prob = log_prob.sum(-1)

        # [batch; float]
        length = msg_length.float() / message.shape[-1]
        # [batch; float]
        loss_len = pg_loss(log_prob, -length)
        return loss_len

    def __call__(self, batch: Batch) -> Tensor:
        # [batch; float]
        loss_r = self.receiver_loss(batch)
        # [batch; float]
        loss_s = self.sender_loss(batch, loss_r)

        # [batch; float]
        loss = loss_r + loss_s

        if self.entropy_loss_weight > 0.0:
            # [batch; float]
            loss_ent = self.entropy_loss(batch)
            loss += self.entropy_loss_weight * loss_ent

        if self.length_loss_weight > 0.0:
            # [batch; float]
            loss_len = self.length_loss(batch)
            loss += self.length_loss_weight * loss_len

        return loss

    def metrics(self, batch: Batch) -> Dict[str, Any]:
        loss = self(batch)
        loss_r = self.receiver_loss(batch)
        loss_s = self.sender_loss(batch, loss_r)
        loss_ent = self.entropy_loss(batch)
        loss_len = self.length_loss(batch)

        return {
            "loss/total.mean": loss.mean().item(),
            "loss/receiver.mean": loss_r.mean().item(),
            "loss/sender.mean": loss_s.mean().item(),
            "loss/entropy.mean": loss_ent.mean().item(),
            "loss/length.mean": loss_len.mean().item(),
            # "loss/total.std": loss.std().item(),
            # "loss/receiver.std": loss_r.std().item(),
            # "loss/sender.std": loss_s.std().item(),
            # "loss/entropy.std": loss_ent.std().item(),
            # "loss/length.std": loss_len.std().item(),
        }
