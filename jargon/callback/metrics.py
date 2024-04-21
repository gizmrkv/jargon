from pathlib import Path
from typing import Callable, List

from jargon.envs import Environment
from jargon.logger import Logger
from jargon.utils import torchdict
from jargon.utils.torchdict import TensorDict

from .callback import Callback


class MetricsCallback(Callback):
    def __init__(
        self,
        env: Environment,
        metrics: Callable[[TensorDict], TensorDict],
        loggers: List[Logger],
        interval: int = 1,
    ):
        self.env = env
        self.metrics = metrics
        self.loggers = loggers
        self.interval = interval

    def on_epoch_end(self, epoch: int, log_dir: Path):
        if epoch % self.interval != 0:
            return

        metrics = {}
        for mode in self.env.modes:
            self.env.mode = mode
            met = [self.metrics(batches) for batches in self.env.rollout()]
            met = torchdict.stack(met)
            met = met.apply(lambda m: m.mean())
            met = {f"{mode}/{k}": v.item() for k, v in met.flatten("/").items()}  # type: ignore
            metrics |= met

        for logger in self.loggers:
            logger.log(epoch, metrics)
