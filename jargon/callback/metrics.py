from pathlib import Path
from typing import Callable, List

from jargon.logger import Logger
from jargon.utils import torchdict
from jargon.utils.torchdict import TensorDict

from .callback import Callback


class MetricsCallback(Callback):
    def __init__(
        self, metrics: Callable[[TensorDict], TensorDict], loggers: List[Logger]
    ):
        self.metrics = metrics
        self.loggers = loggers

    def on_epoch_begin(self, epoch: int, log_dir: Path):
        self.metrics_list = []
        self.losses_list = []

    def on_batch_end(self, epoch: int, batches: TensorDict, losses: TensorDict):
        self.metrics_list.append(self.metrics(batches))
        self.losses_list.append(losses)

    def on_epoch_end(self, epoch: int, log_dir: Path):
        metrics = torchdict.stack(self.metrics_list)
        metrics = metrics.apply(lambda x: x.mean())
        losses = torchdict.stack(self.losses_list)
        losses = losses.apply(lambda x: x.mean())

        metrics_dict = {f"metric/{k}": v.item() for k, v in metrics.flatten().items()}  # type: ignore
        losses_dict = {f"loss/{k}": v.item() for k, v in losses.flatten().items()}  # type: ignore

        for logger in self.loggers:
            logger.log(epoch, metrics_dict | losses_dict)
