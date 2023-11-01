from typing import Dict

import numpy as np
from numpy.typing import NDArray
from torch import Tensor
from torch.distributions import Categorical

from jargon.core import Batch
from jargon.utils.analysis import language_similarity, topographic_similarity
from jargon.zoo.multi_signaling.loss import ExplorationLoss


def accuracy_metrics(batch: Batch, num_elems: int, num_attrs: int) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for name_s, outputs in batch.outputs.items():  # type: ignore
        for name_r, output in outputs.items():  # type: ignore
            output = output[:, -1, :]  # type: ignore
            acc_flag = (
                output.reshape(-1, num_attrs, num_elems).argmax(-1) == batch.target  # type: ignore
            )
            acc_comp = acc_flag.all(-1).float()
            acc_part = acc_flag.float()
            metrics |= {
                f"acc/comp.{name_s}->{name_r}.mean": acc_comp.mean().item(),
                f"acc/comp.{name_s}->{name_r}.std": acc_comp.std().item(),
                f"acc/part.{name_s}->{name_r}.mean": acc_part.mean().item(),
                f"acc/part.{name_s}->{name_r}.std": acc_part.std().item(),
            }

    return metrics


def message_metrics(batch: Batch, vocab_size: int, max_len: int) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for name_s in batch.messages:  # type: ignore
        input: Tensor = batch.input  # type: ignore
        message: Tensor = batch.messages[name_s]  # type: ignore
        msg_logits: Tensor = batch.messages_logits[name_s]  # type: ignore
        msg_mask: Tensor = batch.messages_mask[name_s]  # type: ignore
        msg_length: Tensor = batch.messages_length[name_s]  # type: ignore

        distr = Categorical(logits=msg_logits.reshape(-1, max_len, vocab_size))
        entropy = distr.entropy() * msg_mask

        msg_length = msg_length.float()

        unique = message.unique(dim=0).shape[0] / message.shape[0]

        metrics |= {
            f"msg/entropy.{name_s}.mean": entropy.mean().item(),
            f"msg/entropy.{name_s}.std": entropy.std().item(),
            f"msg/length.{name_s}.mean": msg_length.mean().item(),
            f"msg/length.{name_s}.std": msg_length.std().item(),
            f"msg/unique.{name_s}": unique,
        }

    return metrics


def loss_metrics(batch: Batch, loss: ExplorationLoss) -> Dict[str, float]:
    loss_com = loss.communication_losses(batch)
    loss_sen = loss.sender_losses(batch, loss_com)
    loss_rec = loss.receiver_losses(batch, loss_com)

    metrics: Dict[str, float] = {}
    for name, l in (loss_sen | loss_rec).items():
        metrics |= {
            f"loss/{name}.mean": l.mean().item(),
            f"loss/{name}.std": l.std().item(),
        }

    return metrics


def topsim_metrics(batch: Batch) -> Dict[str, float]:
    input: Tensor = batch.input  # type: ignore
    metrics: Dict[str, float] = {}
    for name_s, message in batch.messages.items():  # type: ignore
        topsim = topographic_similarity(
            input.cpu().numpy(),
            message.cpu().numpy(),  # type: ignore
            y_processor=drop_padding,  # type: ignore
        )
        metrics[f"topsim/{name_s}"] = topsim

    return metrics


def langsim_metrics(batch: Batch) -> Dict[str, float]:
    names_s = list(batch.messages)  # type: ignore
    metrics = {}
    for i in range(len(names_s)):
        for j in range(i + 1, len(names_s)):
            m1 = batch.messages[names_s[i]].cpu().numpy()  # type: ignore
            m2 = batch.messages[names_s[j]].cpu().numpy()  # type: ignore
            ls = language_similarity(m1, m2, processor=drop_padding)
            metrics[f"langsim/{names_s[i]}-{names_s[j]}"] = ls

    return metrics


def drop_padding(x: NDArray[np.int32]) -> NDArray[np.int32]:
    i = np.argwhere(x == 0)
    return x if len(i) == 0 else x[: i[0, 0]]
