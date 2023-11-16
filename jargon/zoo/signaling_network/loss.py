import math
from typing import Dict, Mapping, Set

import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import functional as F

from jargon.core import Batch
from jargon.game import SignalingNetworkGame
from jargon.net.loss import pg_loss


class Loss:
    def __init__(
        self,
        num_elems: int,
        num_attrs: int,
        vocab_size: int,
        max_len: int,
        game: SignalingNetworkGame,
        adaptation_targets: Mapping[str, Set[str]],
        entropy_loss_weight: float = 0.0,
        length_loss_weight: float = 0.0,
        imitation_targets: Mapping[str, Set[str]] | None = None,
        imitation_triggers: Mapping[str, Set[str]] | None = None,
        imitation_threshold: float = 0.99,
    ) -> None:
        self.num_elems = num_elems
        self.num_attrs = num_attrs
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.game = game
        self.adaptation_targets = adaptation_targets
        self.entropy_loss_weight = entropy_loss_weight
        self.length_loss_weight = length_loss_weight
        self.imitation_targets = imitation_targets or {}
        self.imitation_triggers = imitation_triggers or {}
        self.imitation_threshold = imitation_threshold

    def receiver_communication_losses(
        self, batch: Batch
    ) -> Dict[str, Dict[str, Tensor]]:
        outputs_logits: Batch = batch.outputs_logits  # type: ignore
        target: Tensor = batch.target  # type: ignore
        target = target.reshape(-1)

        losses: Dict[str, Dict[str, Tensor]] = {k: {} for k in self.game.senders}
        for name_s, output_logits_r in outputs_logits.items():
            for name_r, logits in output_logits_r.items():  # type: ignore
                logits = logits.reshape(-1, self.num_elems)  # type: ignore
                loss_r = F.cross_entropy(logits, target, reduction="none")
                loss_r = loss_r.reshape(-1, self.num_attrs).mean(-1)
                losses[name_s][name_r] = loss_r
        return losses

    def sender_communication_loss(
        self, batch: Batch, receiver_communication_losses: Dict[str, Dict[str, Tensor]]
    ) -> Dict[str, Tensor]:
        losses: Dict[str, Tensor] = {}
        for name_s in self.game.senders:
            targets_r = self.adaptation_targets[name_s]

            losses_r = torch.stack(
                [receiver_communication_losses[name_s][r] for r in targets_r], dim=-1
            )
            reward = -losses_r.detach().mean(dim=-1)
            reward = reward - reward.mean(dim=0)
            reward = reward.unsqueeze(1)

            msg_logits: Tensor = batch.messages_logits[name_s]  # type: ignore
            distr = Categorical(logits=msg_logits)
            log_prob = distr.log_prob(batch.messages[name_s])
            log_prob = log_prob * batch.messages_mask[name_s]

            log_prob = log_prob.sum(-1).unsqueeze(1)
            loss_s = pg_loss(log_prob, reward).squeeze(1)
            losses[name_s] = loss_s

        return losses

    def sender_imitation_losses(self, batch: Batch) -> Dict[str, Dict[str, Tensor]]:
        losses: Dict[str, Dict[str, Tensor]] = {k: {} for k in self.game.senders}
        for name_s, targets_s in self.imitation_targets.items():
            target: Tensor = batch.target  # type: ignore
            logits: Tensor = batch.messages_logits[name_s]  # type: ignore
            logits = logits.reshape(-1, self.vocab_size)
            for target_s in targets_s:
                trigger_list = []
                for trigger_r in self.imitation_triggers[name_s]:
                    output: Tensor = batch.outputs[target_s][trigger_r]  # type: ignore
                    mask = output == target
                    trigger_list.append(mask)

                trigger = torch.cat(trigger_list, dim=-1).float().mean(dim=-1)
                trigger = (trigger >= self.imitation_threshold).float()

                message: Tensor = batch.messages[target_s].reshape(-1)  # type: ignore
                loss = F.cross_entropy(logits, message, reduction="none")
                loss = loss.reshape(-1, self.max_len).mean(-1)
                loss = loss * trigger
                losses[name_s][target_s] = loss

        return losses

    def sender_entropy_loss(self, batch: Batch) -> Dict[str, Tensor]:
        losses: Dict[str, Tensor] = {}
        for name_s in self.game.senders:
            msg_logits: Tensor = batch.messages_logits[name_s]
            distr = Categorical(logits=msg_logits)
            entropy = distr.entropy() * batch.messages_mask[name_s]
            entropy = entropy.sum(dim=-1)
            losses[name_s] = -entropy

        return losses

    def sender_length_loss(self, batch: Batch) -> Dict[str, Tensor]:
        losses: Dict[str, Tensor] = {}
        for name_s in self.game.senders:
            msg_length = batch.messages_length[name_s].float() / self.max_len

            distr = Categorical(logits=batch.messages_logits[name_s])
            log_prob = distr.log_prob(batch.messages[name_s])
            log_prob = log_prob * batch.messages_mask[name_s]
            log_prob = log_prob.sum(-1).unsqueeze(1)

            loss_len = pg_loss(log_prob, -msg_length.unsqueeze(1)).squeeze(1)
            losses[name_s] = loss_len

        return losses

    def sender_loss(
        self,
        sender_communication_losses: Dict[str, Tensor],
        sender_imitation_losses: Dict[str, Dict[str, Tensor]],
        sender_entropy_losses: Dict[str, Tensor] | None = None,
        sender_length_losses: Dict[str, Tensor] | None = None,
    ) -> Dict[str, Tensor]:
        losses = {}
        for name_s in self.game.senders:
            loss_s = torch.stack(
                [
                    sender_communication_losses[name_s],
                    *sender_imitation_losses[name_s].values(),
                ],
                dim=-1,
            )
            loss_s = loss_s.sum(dim=-1)

            if sender_entropy_losses is not None:
                loss_s += self.entropy_loss_weight * sender_entropy_losses[name_s]

            if sender_length_losses is not None:
                loss_s += self.length_loss_weight * sender_length_losses[name_s]

            losses[name_s] = loss_s

        return losses

    def receiver_loss(
        self, receiver_communication_losses: Dict[str, Dict[str, Tensor]]
    ) -> Dict[str, Tensor]:
        losses = {}
        for name_r in self.game.receivers:
            losses_r = []
            for target_s in self.adaptation_targets[name_r]:
                losses_r.append(receiver_communication_losses[target_s][name_r])

            loss_r = torch.stack(losses_r, dim=-1).sum(dim=-1)
            losses[name_r] = loss_r
        return losses

    def __call__(self, batch: Batch) -> Tensor:
        receiver_communication_losses = self.receiver_communication_losses(batch)
        sender_communication_losses = self.sender_communication_loss(
            batch, receiver_communication_losses
        )
        sender_imitation_losses = self.sender_imitation_losses(batch)

        if not math.isclose(self.entropy_loss_weight, 0.0):
            sender_entropy_losses = self.sender_entropy_loss(batch)
        else:
            sender_entropy_losses = None

        if not math.isclose(self.length_loss_weight, 0.0):
            sender_length_losses = self.sender_length_loss(batch)
        else:
            sender_length_losses = None

        sender_losses = self.sender_loss(
            sender_communication_losses,
            sender_imitation_losses,
            sender_entropy_losses,
            sender_length_losses,
        )
        receiver_losses = self.receiver_loss(receiver_communication_losses)

        loss_s = torch.stack(list(sender_losses.values()), dim=-1).sum(dim=-1)
        loss_r = torch.stack(list(receiver_losses.values()), dim=-1).sum(dim=-1)
        loss = loss_s + loss_r
        return loss

    def metrics(self, batch: Batch) -> Dict[str, float]:
        loss = self(batch)
        return {"loss/mean": loss.mean().item(), "loss/std": loss.std().item()}
