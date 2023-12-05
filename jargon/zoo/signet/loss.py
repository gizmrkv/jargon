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
        instantly: bool = False,
        discount_factor: float = 0.99,
    ) -> None:
        self.num_elems = num_elems
        self.num_attrs = num_attrs
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.game = game
        self.adaptation_targets = adaptation_targets
        self.entropy_loss_weight = entropy_loss_weight
        self.length_loss_weight = length_loss_weight
        self.instantly = instantly
        self.discount_factor = discount_factor

    def receiver_communication_losses(
        self, batch: Batch
    ) -> Dict[str, Dict[str, Tensor]]:
        outputs_logits: Batch = batch.outputs_logits  # type: ignore
        target: Tensor = batch.target  # type: ignore
        if self.instantly:
            target = target.unsqueeze(1).repeat(1, self.max_len, 1)
        target = target.reshape(-1)

        losses: Dict[str, Dict[str, Tensor]] = {k: {} for k in self.game.senders}
        for name_s, output_logits_r in outputs_logits.items():
            for name_r, logits in output_logits_r.items():  # type: ignore
                logits = logits.reshape(-1, self.num_elems)  # type: ignore
                loss_r = F.cross_entropy(logits, target, reduction="none")
                if self.instantly:
                    loss_r = loss_r.reshape(-1, self.max_len, self.num_attrs).mean(-1)
                else:
                    loss_r = loss_r.reshape(-1, self.num_attrs).mean(-1)
                losses[name_s][name_r] = loss_r
        return losses

    def sender_communication_loss(
        self, batch: Batch, receiver_communication_losses: Dict[str, Dict[str, Tensor]]
    ) -> Dict[str, Tensor]:
        losses: Dict[str, Tensor] = {}
        for name_s in self.game.senders:
            message: Tensor = batch.messages[name_s]  # type: ignore
            msg_logits: Tensor = batch.messages_logits[name_s]  # type: ignore
            msg_mask: Tensor = batch.messages_mask[name_s]  # type: ignore

            distr = Categorical(logits=msg_logits)
            log_prob = distr.log_prob(message)
            log_prob = log_prob * msg_mask
            if not self.instantly:
                log_prob = log_prob.sum(-1)

            targets_r = self.adaptation_targets[name_s]
            targets_r.intersection_update(set(receiver_communication_losses[name_s]))

            loss_r = torch.stack(
                [receiver_communication_losses[name_s][r] for r in targets_r], dim=-1
            ).mean(dim=-1)
            reward = -loss_r.detach()
            losses[name_s] = pg_loss(log_prob, reward)

        return losses

    def sender_entropy_loss(self, batch: Batch) -> Dict[str, Tensor]:
        losses: Dict[str, Tensor] = {}
        for name_s in self.game.senders:
            msg_logits: Tensor = batch.messages_logits[name_s]  # type: ignore
            msg_mask: Tensor = batch.messages_mask[name_s]  # type: ignore
            distr = Categorical(logits=msg_logits)
            entropy = distr.entropy() * msg_mask
            entropy = entropy.mean(dim=-1)
            losses[name_s] = -entropy

        return losses

    def sender_length_loss(self, batch: Batch) -> Dict[str, Tensor]:
        losses: Dict[str, Tensor] = {}
        for name_s in self.game.senders:
            message: Tensor = batch.messages[name_s]  # type: ignore
            msg_logits: Tensor = batch.messages_logits[name_s]  # type: ignore
            msg_length: Tensor = batch.messages_length[name_s]
            msg_mask: Tensor = batch.messages_mask[name_s]  # type: ignore

            distr = Categorical(logits=msg_logits)
            log_prob = distr.log_prob(message)
            log_prob = log_prob * msg_mask
            log_prob = log_prob.sum(-1)

            length = msg_length.float() / message.shape[-1]
            loss_len = pg_loss(log_prob, -length)
            losses[name_s] = loss_len

        return losses

    def sender_loss(
        self,
        sender_communication_losses: Dict[str, Tensor],
        sender_entropy_losses: Dict[str, Tensor] | None = None,
        sender_length_losses: Dict[str, Tensor] | None = None,
    ) -> Dict[str, Tensor]:
        losses = {}
        for name_s in self.game.senders:
            loss_s = sender_communication_losses[name_s]

            if self.instantly:
                loss_s = loss_s.sum(-1)

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

            if self.instantly:
                loss_r = loss_r.sum(-1)

            losses[name_r] = loss_r
        return losses

    def __call__(self, batch: Batch) -> Tensor:
        receiver_communication_losses = self.receiver_communication_losses(batch)
        sender_communication_losses = self.sender_communication_loss(
            batch, receiver_communication_losses
        )

        if math.isclose(self.entropy_loss_weight, 0.0):
            sender_entropy_losses = None
        else:
            sender_entropy_losses = self.sender_entropy_loss(batch)

        if math.isclose(self.length_loss_weight, 0.0):
            sender_length_losses = None
        else:
            sender_length_losses = self.sender_length_loss(batch)

        sender_losses = self.sender_loss(
            sender_communication_losses,
            sender_entropy_losses,
            sender_length_losses,
        )
        receiver_losses = self.receiver_loss(receiver_communication_losses)

        loss_s = torch.stack(list(sender_losses.values()), dim=-1).sum(dim=-1)
        loss_r = torch.stack(list(receiver_losses.values()), dim=-1).sum(dim=-1)
        loss = loss_s + loss_r
        return loss

    def metrics(self, batch: Batch) -> Dict[str, float]:
        receiver_communication_losses = self.receiver_communication_losses(batch)
        sender_communication_losses = self.sender_communication_loss(
            batch, receiver_communication_losses
        )
        sender_entropy_losses = self.sender_entropy_loss(batch)
        sender_length_losses = self.sender_length_loss(batch)
        sender_losses = self.sender_loss(
            sender_communication_losses,
            sender_entropy_losses,
            sender_length_losses,
        )
        receiver_losses = self.receiver_loss(receiver_communication_losses)

        metrics: Dict[str, float] = {}
        for name_s, losses in receiver_communication_losses.items():
            for name_r, loss in losses.items():
                metrics |= {
                    f"loss/com.{name_s}->{name_r}.mean": loss.mean().item(),
                    f"loss/com.{name_s}->{name_r}.std": loss.std().item(),
                }

        for name_s, loss in sender_communication_losses.items():
            metrics |= {
                f"loss/com.{name_s}.mean": loss.mean().item(),
                f"loss/com.{name_s}.std": loss.std().item(),
            }
        for name_s, loss in sender_entropy_losses.items():
            metrics |= {
                f"loss/ent.{name_s}.mean": loss.mean().item(),
                f"loss/ent.{name_s}.std": loss.std().item(),
            }
        for name_s, loss in sender_length_losses.items():
            metrics |= {
                f"loss/len.{name_s}.mean": loss.mean().item(),
                f"loss/len.{name_s}.std": loss.std().item(),
            }
        for name_s, loss in sender_losses.items():
            metrics |= {
                f"loss/{name_s}.mean": loss.mean().item(),
                f"loss/{name_s}.std": loss.std().item(),
            }
        for name_r, loss in receiver_losses.items():
            metrics |= {
                f"loss/{name_r}.mean": loss.mean().item(),
                f"loss/{name_r}.std": loss.std().item(),
            }

        return metrics
