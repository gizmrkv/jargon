import itertools
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, List, Sequence

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
from jargon.net.functional import drop_padding
from jargon.utils.analysis import language_similarity, topographic_similarity
from jargon.zoo.signet.loss import Loss

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
        eos: int = 0,
    ) -> None:
        self.num_elems = num_elems
        self.num_attrs = num_attrs
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.senders = list(senders)
        self.receivers = list(receivers)
        self.log_dir = log_dir
        self.eos = eos
        self.y_processor = lambda x: drop_padding(x, eos=self.eos)

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
        heavy_metrics = topsim_metrics(batch, self.y_processor) | langsim_metrics(
            batch, self.y_processor
        )

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
    outputs = batch.get_batch("outputs")
    metrics: Dict[str, float] = {}
    for name_s, outs in outputs.items():
        for name_r, output in outs.items():
            # [batch, num_attrs; bool]
            acc_flag = output == batch.get_tensor("target")
            # [batch; float]
            acc_comp = acc_flag.all(-1).float()
            # [batch; float]
            acc_part = acc_flag.float()
            metrics |= {
                f"acc/comp.{name_s}->{name_r}.mean": acc_comp.mean().item(),
                f"acc/comp.{name_s}->{name_r}.std": acc_comp.std().item(),
                f"acc/part.{name_s}->{name_r}.mean": acc_part.mean().item(),
                f"acc/part.{name_s}->{name_r}.std": acc_part.std().item(),
            }

    for name in ["comp", "part"]:
        metrics[f"acc/{name}.mean.mean"] = np.mean(
            [v for k, v in metrics.items() if re.match(rf"acc/{name}.*mean", k)]
        )

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
    messages = batch.get_batch("messages")
    msgs_logits = batch.get_batch("messages_logits")
    msgs_length = batch.get_batch("messages_length")
    msgs_mask = batch.get_batch("messages_mask")
    metrics: Dict[str, float] = {}
    for name_s in messages:
        # [batch, max_len; int]
        message = messages.get_tensor(name_s)
        # [batch, max_len, vocab_size; float]
        msg_logits = msgs_logits.get_tensor(name_s)
        # [batch; int]
        msg_length = msgs_length.get_tensor(name_s)
        # [batch, max_len; int]
        msg_mask = msgs_mask.get_tensor(name_s)

        distr = Categorical(logits=msg_logits.reshape(-1, max_len, vocab_size))
        # [batch, max_len; float]
        entropy = distr.entropy() * msg_mask

        # [batch; float]
        msg_length = msg_length.float()

        unique: float = message.unique(dim=0).shape[0] / message.shape[0]

        metrics |= {
            f"msg/entropy.{name_s}.mean": entropy.mean().item(),
            f"msg/entropy.{name_s}.std": entropy.std().item(),
            f"msg/length.{name_s}.mean": msg_length.mean().item(),
            f"msg/length.{name_s}.std": msg_length.std().item(),
            f"msg/unique.{name_s}": unique,
        }

    for name in ["entropy", "length"]:
        metrics[f"msg/{name}.mean.mean"] = np.mean(
            [v for k, v in metrics.items() if re.match(rf"msg/{name}.*mean", k)]
        )
    metrics["msg/unique.mean"] = np.mean(
        [v for k, v in metrics.items() if re.match(rf"msg/unique.*", k)]
    )

    return metrics


def topsim_metrics(
    batch: Batch, y_processor: Callable[[NDArray[np.int32]], Sequence[Hashable]]
) -> Dict[str, float]:
    # [batch, num_attrs; int]
    input = batch.get_tensor("input")
    messages = batch.get_batch("messages")
    input = input.cpu().numpy()

    metrics: Dict[str, float] = {}
    for name_s, message in messages.items():
        # [batch, max_len; int]
        message = message.cpu().numpy()
        metrics |= {
            f"topsim/{name_s}": topographic_similarity(
                input, message, y_processor=y_processor, y_dist="Levenshtein"
            )
        }

    topsim_mean = np.mean(list(metrics.values()))
    topsim_min = np.min(list(metrics.values()))
    topsim_max = np.max(list(metrics.values()))
    metrics |= {
        f"topsim/mean": topsim_mean,
        f"topsim/min": topsim_min,
        f"topsim/max": topsim_max,
    }

    return metrics


def langsim_metrics(
    batch: Batch, y_processor: Callable[[NDArray[np.int32]], Sequence[Hashable]]
) -> Dict[str, float]:
    messages = batch.get_batch("messages")
    names_s = list(messages)
    metrics = {}
    for i in range(len(names_s)):
        for j in range(i + 1, len(names_s)):
            m1 = messages.get_tensor(names_s[i]).cpu().numpy()
            m2 = messages.get_tensor(names_s[j]).cpu().numpy()
            ls = language_similarity(m1, m2, processor=y_processor)
            metrics[f"langsim/{names_s[i]}-{names_s[j]}"] = ls

    langsim_mean = np.mean(list(metrics.values()))
    langsim_min = np.min(list(metrics.values()))
    langsim_max = np.max(list(metrics.values()))
    metrics |= {
        f"langsim/mean": langsim_mean,
        f"langsim/min": langsim_min,
        f"langsim/max": langsim_max,
    }

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


class LanguageMetrics:
    def __init__(self, senders: Sequence[str], log_dir: Path, eos: int = 0) -> None:
        self.senders = list(senders)
        self.log_dir = log_dir
        self.eos = eos
        self.langs_dir = {s: log_dir / "langs" / s for s in senders}

        for d in self.langs_dir.values():
            os.makedirs(d, exist_ok=True)

    def __call__(self, batch: Batch, epoch: int) -> Dict[str, float]:
        input = batch.get_tensor("input")
        messages = batch.get_batch("messages")

        inp_lines = [",".join([str(y) for y in x]) for x in input.tolist()]
        langs = {}
        for s, message in messages.items():
            msg_list = message.tolist()
            msg_list = [
                x[: x.index(self.eos) if self.eos in x else len(x)] for x in msg_list
            ]
            msg_list = ["-".join([str(y) for y in x]) for x in msg_list]
            lines = [",".join([x, y]) for x, y in zip(inp_lines, msg_list)]
            langs[s] = "\n".join(lines)

        for s, lang in langs.items():
            path = self.langs_dir[s] / f"{epoch:0>8}.csv"
            with open(path, "a") as f:
                f.write(lang + "\n")
