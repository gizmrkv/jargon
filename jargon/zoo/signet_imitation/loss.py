from typing import Dict, Mapping, Set

import torch
from torch import Tensor
from torch.nn import functional as F

from jargon.core import Batch
from jargon.zoo.signet.loss import Loss


class ImitationLoss:
    def __init__(
        self,
        loss: Loss,
        imitation_targets: Mapping[str, Set[str]] | None = None,
        imitation_triggers: Mapping[str, Set[str]] | None = None,
        imitation_threshold: float = 0.99,
    ) -> None:
        self.loss = loss
        self.imitation_targets = imitation_targets or {}
        self.imitation_triggers = imitation_triggers or {}
        self.imitation_threshold = imitation_threshold

    def sender_imitation_losses(self, batch: Batch) -> Dict[str, Dict[str, Tensor]]:
        losses: Dict[str, Dict[str, Tensor]] = {k: {} for k in self.loss.game.senders}
        for name_s, targets_s in self.imitation_targets.items():
            target: Tensor = batch.target  # type: ignore
            logits: Tensor = batch.messages_logits[name_s]  # type: ignore
            logits = logits.reshape(-1, self.loss.vocab_size)
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
                loss = loss.reshape(-1, self.loss.max_len).mean(-1)
                loss = loss * trigger
                losses[name_s][target_s] = loss

        return losses

    def __call__(self, batch: Batch) -> Tensor:
        loss_total = self.loss(batch)
        loss_imi = self.sender_imitation_losses(batch)
        for losses in loss_imi.values():
            for loss in losses.values():
                loss_total += loss

        return loss_total

    def metrics(self, batch: Batch) -> Dict[str, float]:
        metrics = self.loss.metrics(batch)
        sender_imitation_losses = self.sender_imitation_losses(batch)

        for name_s1, losses in sender_imitation_losses.items():
            for name_s2, loss in losses.items():
                metrics |= {
                    f"loss/imi.{name_s1}->{name_s2}.mean": loss.mean().item(),
                    f"loss/imi.{name_s1}->{name_s2}.std": loss.std().item(),
                }

        return metrics
