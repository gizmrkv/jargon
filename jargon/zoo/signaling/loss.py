import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor, nn, optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset

from jargon.core import Batch, Trainer
from jargon.game import SignalingGame
from jargon.net import MLP, MultiDiscreteMLP, Receiver, Sender
from jargon.net.loss import pg_loss
from jargon.utils import BaseLogger, fix_seed, init_weights, random_split
from jargon.utils.analysis import topographic_similarity


class Loss:
    def __init__(
        self,
        num_elems: int,
        num_attrs: int,
        entropy_loss_weight: float = 0.0,
        length_loss_weight: float = 0.0,
    ) -> None:
        self.num_elems = num_elems
        self.num_attrs = num_attrs
        self.entropy_loss_weight = entropy_loss_weight
        self.length_loss_weight = length_loss_weight

    def receiver_loss(self, batch: Batch) -> Tensor:
        logits: Tensor = batch.output_logits  # type: ignore
        target: Tensor = batch.target  # type: ignore

        logits = logits.reshape(-1, self.num_elems)
        target = target.reshape(-1)
        loss_r = F.cross_entropy(logits, target, reduction="none")
        loss_r = loss_r.reshape(-1, self.num_attrs).mean(-1)
        return loss_r

    def sender_loss(self, batch: Batch, receiver_loss: Tensor) -> Tensor:
        logits: Tensor = batch.output_logits  # type: ignore
        target: Tensor = batch.target  # type: ignore
        message: Tensor = batch.message  # type: ignore
        msg_logits: Tensor = batch.message_logits  # type: ignore
        msg_mask: Tensor = batch.message_mask  # type: ignore

        distr = Categorical(logits=msg_logits)
        log_prob = distr.log_prob(message)
        log_prob = log_prob * msg_mask

        log_prob = log_prob.sum(-1)

        reward = -receiver_loss.detach()
        reward = reward - reward.mean()
        reward_std = reward.std()
        reward = reward / (reward_std + 1e-8)

        loss_s = -log_prob * reward
        return loss_s

    def entropy_loss(self, batch: Batch) -> Tensor:
        msg_logits: Tensor = batch.message_logits  # type: ignore
        msg_mask: Tensor = batch.message_mask  # type: ignore

        distr = Categorical(logits=msg_logits)
        loss_ent = distr.entropy() * msg_mask
        loss_ent = loss_ent.mean(-1)
        return -loss_ent

    def length_loss(self, batch: Batch) -> Tensor:
        message: Tensor = batch.message  # type: ignore
        msg_logits: Tensor = batch.message_logits  # type: ignore
        msg_length: Tensor = batch.message_length  # type: ignore
        msg_mask: Tensor = batch.message_mask  # type: ignore

        distr = Categorical(logits=msg_logits)
        log_prob = distr.log_prob(message)
        log_prob = log_prob * msg_mask
        log_prob = log_prob.sum(-1)

        length = msg_length.float() / message.shape[-1]
        loss_len = pg_loss(log_prob.unsqueeze(1), -length.unsqueeze(1)).squeeze(1)
        return loss_len

    def __call__(self, batch: Batch) -> Tensor:
        loss_r = self.receiver_loss(batch)
        loss_s = self.sender_loss(batch, loss_r)

        loss = loss_r + loss_s

        if self.entropy_loss_weight > 0.0:
            loss_ent = self.entropy_loss(batch)
            loss += self.entropy_loss_weight * loss_ent

        if self.length_loss_weight > 0.0:
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
            "loss/total.std": loss.std().item(),
            "loss/receiver.mean": loss_r.mean().item(),
            "loss/receiver.std": loss_r.std().item(),
            "loss/sender.mean": loss_s.mean().item(),
            "loss/sender.std": loss_s.std().item(),
            "loss/entropy.mean": loss_ent.mean().item(),
            "loss/entropy.std": loss_ent.std().item(),
            "loss/length.mean": loss_len.mean().item(),
            "loss/length.std": loss_len.std().item(),
        }
