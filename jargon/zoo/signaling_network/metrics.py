import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from moviepy.editor import ImageSequenceClip
from numpy.typing import NDArray
from torch import Tensor
from torch.distributions import Categorical

from jargon.core import Batch
from jargon.utils.analysis import language_similarity, topographic_similarity
from jargon.zoo.signaling_network.loss import Loss

matplotlib.use("Agg")


class Metrics:
    def __init__(
        self,
        num_elems: int,
        num_attrs: int,
        vocab_size: int,
        max_len: int,
        senders: Sequence[str],
        receivers: Sequence[str],
        log_dir: Path,
    ) -> None:
        self.num_elems = num_elems
        self.num_attrs = num_attrs
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.senders = list(senders)
        self.receivers = list(receivers)
        self.log_dir = log_dir

        self.acc_df_dir = log_dir / "acc"
        self.langsim_df_dir = log_dir / "langsim"

        os.makedirs(self.acc_df_dir, exist_ok=True)
        os.makedirs(self.langsim_df_dir, exist_ok=True)

        self.heatmap_kwargs = {
            "vmin": 0,
            "vmax": 1,
            "cmap": "viridis",
            "annot": True,
            "fmt": ".2f",
            "cbar": True,
            "square": True,
        }

    def __call__(self, batch: Batch) -> Dict[str, float]:
        return accuracy_metrics(batch) | message_metrics(
            batch, self.vocab_size, self.max_len
        )

    def heavy_test(
        self, batch: Batch, metrics: Dict[str, Any], epoch: int
    ) -> Dict[str, Any]:
        heavy_metrics = topsim_metrics(batch) | langsim_metrics(batch)

        acc_df = accuracy_dataframe(metrics, self.senders, self.receivers)
        langsim_df = langsim_dataframe(heavy_metrics, self.senders)
        dataframe_to_image(acc_df, epoch, self.acc_df_dir, self.heatmap_kwargs)
        dataframe_to_image(langsim_df, epoch, self.langsim_df_dir, self.heatmap_kwargs)

        return heavy_metrics

    def frames_to_movies(self) -> Dict[str, Path]:
        movie_paths = {}
        for df_dir, df_name in [
            (self.acc_df_dir, "acc"),
            (self.langsim_df_dir, "langsim"),
        ]:
            frame_paths = list(df_dir.glob("*.png"))
            frame_paths = sorted([f.as_posix() for f in frame_paths])
            path = df_dir.joinpath("video.mp4")
            clip = ImageSequenceClip(frame_paths, fps=10)
            clip.write_videofile(path.as_posix())
            movie_paths[df_name] = path

        return movie_paths


def accuracy_metrics(batch: Batch) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for name_s, outputs in batch.outputs.items():  # type: ignore
        for name_r, output in outputs.items():  # type: ignore
            acc_flag = output == batch.target  # type: ignore
            acc_comp = acc_flag.all(-1).float()
            acc_part = acc_flag.float()
            metrics |= {
                f"acc/comp.{name_s}->{name_r}.mean": acc_comp.mean().item(),
                f"acc/comp.{name_s}->{name_r}.std": acc_comp.std().item(),
                f"acc/part.{name_s}->{name_r}.mean": acc_part.mean().item(),
                f"acc/part.{name_s}->{name_r}.std": acc_part.std().item(),
            }

    return metrics


def accuracy_dataframe(
    metrics: Dict[str, float], senders: Sequence[str], receivers: Sequence[str]
) -> pd.DataFrame:
    matrix: List[List[float]] = [[-1.0 for _ in receivers] for _ in senders]
    for s in range(len(senders)):
        for r in range(len(receivers)):
            matrix[s][r] = metrics[f"acc/part.{senders[s]}->{receivers[r]}.mean"]

    return pd.DataFrame(data=matrix, columns=receivers, index=senders)


def message_metrics(batch: Batch, vocab_size: int, max_len: int) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for name_s in batch.messages:  # type: ignore
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


def loss_metrics(batch: Batch, loss: Loss) -> Dict[str, float]:
    loss_com = loss.receiver_communication_losses(batch)
    loss_imi = loss.sender_imitation_losses(batch)

    loss_sen = loss.sender_losses(batch, loss_com, loss_imi)
    loss_rec = loss.receiver_losses(batch, loss_com, loss_imi)

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

    metrics["topsim/mean"] = np.mean(list(metrics.values()))
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

    metrics["langsim/mean"] = np.mean(list(metrics.values()))
    return metrics


def langsim_dataframe(
    metrics: Dict[str, float], senders: Sequence[str]
) -> pd.DataFrame:
    matrix: List[List[float]] = [[-1.0 for _ in senders] for _ in senders]
    for i in range(len(senders)):
        matrix[i][i] = 1.0
        for j in range(i + 1, len(senders)):
            langsim = metrics[f"langsim/{senders[i]}-{senders[j]}"]
            matrix[i][j] = langsim
            matrix[j][i] = langsim

    return pd.DataFrame(data=matrix, columns=senders, index=senders)


def drop_padding(x: NDArray[np.int32]) -> NDArray[np.int32]:
    i = np.argwhere(x == 0)
    return x if len(i) == 0 else x[: i[0, 0]]


def dataframe_to_image(
    df: pd.DataFrame,
    epoch: int,
    save_dir: Path,
    heatmap_kwargs: Dict[str, Any] = {},
) -> None:
    sns.heatmap(df, **heatmap_kwargs)
    plt.title(f"Epoch {epoch}")
    path = save_dir.joinpath(f"{epoch:0>8}.png")
    plt.savefig(path)
    plt.clf()
