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
    ) -> None:
        self.num_elems = num_elems
        self.num_attrs = num_attrs
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.game = game
        self.adaptation_targets = adaptation_targets
        self.entropy_loss_weight = entropy_loss_weight
        self.length_loss_weight = length_loss_weight

    def receiver_communication_losses(
        self, batch: Batch
    ) -> Dict[str, Dict[str, Tensor]]:
        outputs_logits = batch.get_batch("outputs_logits")
        # [batch, num_attrs; int]
        target = batch.get_tensor("target")
        # [batch * num_attrs; int]
        target = target.reshape(-1)

        losses: Dict[str, Dict[str, Tensor]] = {k: {} for k in self.game.senders}
        for name_s, output_logits_r in outputs_logits.items():
            for name_r, logits in output_logits_r.items():
                # [batch * num_attrs, num_elems; float]
                logits = logits.reshape(-1, self.num_elems)
                # [batch * num_attrs; float]
                loss_r = F.cross_entropy(logits, target, reduction="none")
                # [batch; float]
                loss_r = loss_r.reshape(-1, self.num_attrs).mean(-1)
                losses[name_s][name_r] = loss_r
        return losses

    def sender_communication_loss(
        self, batch: Batch, receiver_communication_losses: Dict[str, Dict[str, Tensor]]
    ) -> Dict[str, Tensor]:
        messages = batch.get_batch("messages")
        msgs_logits = batch.get_batch("messages_logits")
        msgs_mask = batch.get_batch("messages_mask")

        losses: Dict[str, Tensor] = {}
        for name_s in self.game.senders:
            # [batch, max_len; int]
            message = messages.get_tensor(name_s)
            # [batch, max_len, vocab_size; float]
            msg_logits = msgs_logits.get_tensor(name_s)
            # [batch, max_len; int]
            msg_mask = msgs_mask.get_tensor(name_s)

            distr = Categorical(logits=msg_logits)
            # [batch, max_len; float]
            log_prob = distr.log_prob(message)
            # [batch, max_len; float]
            log_prob = log_prob * msg_mask
            # [batch; float]
            log_prob = log_prob.sum(-1)

            targets_r = self.adaptation_targets[name_s]
            targets_r.intersection_update(set(receiver_communication_losses[name_s]))
            # [batch; float]
            loss_r = torch.stack(
                [receiver_communication_losses[name_s][r] for r in targets_r], dim=-1
            ).mean(dim=-1)
            # [batch; float]
            reward = -loss_r.detach()
            # [batch; float]
            losses[name_s] = pg_loss(log_prob, reward)

        return losses

    def sender_entropy_loss(self, batch: Batch) -> Dict[str, Tensor]:
        msgs_logits = batch.get_batch("messages_logits")
        msgs_mask = batch.get_batch("messages_mask")
        losses: Dict[str, Tensor] = {}
        for name_s in self.game.senders:
            # [batch, max_len, vocab_size; float]
            msg_logits = msgs_logits.get_tensor(name_s)
            # [batch, max_len; int]
            msg_mask = msgs_mask.get_tensor(name_s)

            distr = Categorical(logits=msg_logits)
            # [batch, max_len; float]
            entropy = distr.entropy() * msg_mask
            # [batch; float]
            entropy = entropy.mean(dim=-1)
            losses[name_s] = -entropy

        return losses

    def sender_length_loss(self, batch: Batch) -> Dict[str, Tensor]:
        messages = batch.get_batch("messages")
        msgs_logits = batch.get_batch("messages_logits")
        msgs_length = batch.get_batch("messages_length")
        msgs_mask = batch.get_batch("messages_mask")
        losses: Dict[str, Tensor] = {}
        for name_s in self.game.senders:
            # [batch, max_len; int]
            message = messages.get_tensor(name_s)
            # [batch, max_len, vocab_size; float]
            msg_logits = msgs_logits.get_tensor(name_s)
            # [batch; int]
            msg_length = msgs_length.get_tensor(name_s)
            # [batch, max_len; int]
            msg_mask = msgs_mask.get_tensor(name_s)

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
