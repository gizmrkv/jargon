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

    def sender_imitation_losses(
        self, batch: Batch, imitation_threshold: float
    ) -> Dict[str, Dict[str, Tensor]]:
        # [batch, num_attrs; int]
        input = batch.get_tensor("input")
        # [batch, num_attrs; int]
        target = batch.get_tensor("target")
        outputs = batch.get_batch("outputs")
        messages = batch.get_batch("messages")
        losses: Dict[str, Dict[str, Tensor]] = {k: {} for k in self.loss.game.senders}
        for name_s, targets_s in self.imitation_targets.items():
            for target_s in targets_s:
                outputs_s = outputs.get_batch(target_s)
                triggers_r = self.imitation_triggers[name_s].intersection(outputs_s)
                assert len(triggers_r) > 0, "No triggers found"
                trigger_list = []
                for trigger_r in triggers_r:
                    # [batch, num_attrs; int]
                    output = outputs.get_batch(target_s).get_tensor(trigger_r)
                    # [batch, num_attrs; bool]
                    mask = output == target
                    trigger_list.append(mask)

                # [batch, num_attrs * num_triggers; bool]
                trigger = torch.cat(trigger_list, dim=-1)
                # [batch; float]
                trigger = trigger.float().mean(dim=-1)
                # [batch; bool]
                trigger = (trigger >= imitation_threshold).float()

                # [batch, max_len; int]
                tgt_message = messages.get_tensor(target_s)
                # [batch, max_len, vocab_size; float]
                _, logits = self.loss.game.senders[name_s](input, message=tgt_message)
                # [batch * max_len; int]
                tgt_message = tgt_message.reshape(-1)
                # [batch * max_len, vocab_size; float]
                logits = logits.reshape(-1, self.loss.vocab_size)

                # [batch * max_len; float]
                loss = F.cross_entropy(logits, tgt_message, reduction="none")
                # [batch; float]
                loss = loss.reshape(-1, self.loss.max_len).mean(-1)
                # [batch; float]
                loss = loss * trigger
                losses[name_s][target_s] = loss

        return losses

    def __call__(self, batch: Batch) -> Tensor:
        loss_total = self.loss(batch)
        loss_imi = self.sender_imitation_losses(batch, self.imitation_threshold)
        for losses in loss_imi.values():
            for loss in losses.values():
                loss_total += loss

        return loss_total

    def metrics(self, batch: Batch) -> Dict[str, float]:
        metrics = self.loss.metrics(batch)
        sender_imitation_losses = self.sender_imitation_losses(batch, -1.0)

        for name_s1, losses in sender_imitation_losses.items():
            for name_s2, loss in losses.items():
                metrics |= {
                    f"loss/imi.{name_s1}->{name_s2}.mean": loss.mean().item(),
                    f"loss/imi.{name_s1}->{name_s2}.std": loss.std().item(),
                }

        return metrics
