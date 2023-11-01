from typing import Dict, Mapping, Set

import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import functional as F

from jargon.core import Batch
from jargon.game import MultiSignalingGame
from jargon.net.loss import pg_loss


class ExplorationLoss:
    def __init__(
        self,
        num_elems: int,
        num_attrs: int,
        vocab_size: int,
        max_len: int,
        game: MultiSignalingGame,
        discount_factor: float,
        exploration_targets: Mapping[str, Set[str]],
    ) -> None:
        self.num_elems = num_elems
        self.num_attrs = num_attrs
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.game = game
        self.discount_factor = discount_factor
        self.exploration_targets = exploration_targets

    def communication_losses(self, batch: Batch) -> Dict[str, Dict[str, Tensor]]:
        outputs: Batch = batch.outputs  # type: ignore
        target: Tensor = batch.target  # type: ignore

        target_seq = target.unsqueeze(1).expand(-1, self.max_len, -1).reshape(-1)

        losses: Dict[str, Dict[str, Tensor]] = {k: {} for k in self.game.senders}
        for name_s, outputs_r in outputs.items():
            for name_r, output in outputs_r.items():  # type: ignore
                output = output.reshape(-1, self.num_elems)  # type: ignore
                loss_r = F.cross_entropy(output, target_seq, reduction="none")
                loss_r = loss_r.reshape(-1, self.max_len, self.num_attrs).mean(-1)
                losses[name_s][name_r] = loss_r
        return losses

    def sender_losses(
        self, batch: Batch, communication_losses: Dict[str, Dict[str, Tensor]]
    ) -> Dict[str, Tensor]:
        losses = {}
        for name_s in self.game.senders:
            targets_r = self.exploration_targets[name_s]
            losses_r = torch.stack(
                [communication_losses[name_s][r] for r in targets_r], dim=-1
            )
            loss_r = losses_r.mean(-1)
            msg_logits: Tensor = batch.messages_logits[name_s]  # type: ignore
            distr = Categorical(logits=msg_logits)
            log_prob = distr.log_prob(batch.messages[name_s])
            log_prob = log_prob * batch.messages_mask[name_s]

            loss_r = loss_r[:, -1].unsqueeze(-1)
            log_prob = log_prob.sum(-1).unsqueeze(-1)

            loss_s = pg_loss(log_prob, -loss_r.detach(), self.discount_factor)
            losses[name_s] = loss_s
        return losses

    def receiver_losses(
        self, batch: Batch, communication_losses: Dict[str, Dict[str, Tensor]]
    ) -> Dict[str, Tensor]:
        losses = {}
        for name_r in self.game.receivers:
            targets_s = self.exploration_targets[name_r]
            losses_r = torch.stack(
                [communication_losses[s][name_r] for s in targets_s], dim=-1
            )
            loss_r = losses_r.mean(-1)
            losses[name_r] = loss_r
        return losses

    def __call__(self, batch: Batch) -> Tensor:
        communication_losses = self.communication_losses(batch)
        sender_losses = self.sender_losses(batch, communication_losses)
        receiver_losses = self.receiver_losses(batch, communication_losses)

        sender_loss = torch.stack(list(sender_losses.values()), dim=-1).mean(-1)
        receiver_loss = torch.stack(list(receiver_losses.values()), dim=-1).mean(-1)
        total_loss = sender_loss + receiver_loss
        return total_loss
