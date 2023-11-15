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

    def __call__(self, batch: Batch) -> Tensor:
        output: Tensor = batch.output  # type: ignore
        target: Tensor = batch.target  # type: ignore
        message: Tensor = batch.message  # type: ignore
        msg_logits: Tensor = batch.message_logits  # type: ignore
        msg_length: Tensor = batch.message_length  # type: ignore
        msg_mask: Tensor = batch.message_mask  # type: ignore

        # output = output.reshape(-1, num_elems)
        # target = target.unsqueeze(1).expand(-1, max_len, -1).reshape(-1)
        # loss_r = F.cross_entropy(output, target, reduction="none")
        # loss_r = loss_r.reshape(-1, max_len, num_attrs).mean(-1)

        output = output[:, -1, :].reshape(-1, self.num_elems)
        target = target.reshape(-1)
        loss_r = F.cross_entropy(output, target, reduction="none")
        loss_r = loss_r.reshape(-1, self.num_attrs).mean(-1)

        distr = Categorical(logits=msg_logits)
        log_prob = distr.log_prob(message)
        log_prob = log_prob * msg_mask

        # loss_r = loss_r[:, -1].unsqueeze(-1)
        # loss_r = loss_r.unsqueeze(-1)
        # log_prob = log_prob.sum(-1).unsqueeze(-1)
        # loss_s = pg_loss(log_prob, -loss_r.detach(), discount_factor)
        # loss_s = loss_s.mean(-1)
        # loss_r = loss_r.mean(-1)

        log_prob = log_prob.sum(-1)
        reward = -loss_r.detach()
        reward = reward - reward.mean()
        loss_s = -log_prob * reward

        loss = loss_s + loss_r

        if self.entropy_loss_weight > 0.0:
            loss_ent = self.entropy_loss_weight * distr.entropy() * msg_mask
            loss_ent = loss_ent.mean(-1)
            loss -= loss_ent

        if self.length_loss_weight > 0.0:
            length = msg_length.float() / message.shape[-1]
            loss_len = self.length_loss_weight * pg_loss(
                log_prob.sum(-1).unsqueeze(1), -length.unsqueeze(1)
            ).squeeze(1)
            loss += loss_len

        return loss
