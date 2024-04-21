from abc import ABC
from pathlib import Path

from jargon.utils.torchdict import TensorDict


class Callback(ABC):
    def on_train_begin(self, log_dir: Path):
        pass

    def on_epoch_begin(self, epoch: int, log_dir: Path):
        pass

    def on_batch_begin(self, epoch: int):
        pass

    def on_batch_end(self, epoch: int, batches: TensorDict, losses: TensorDict):
        pass

    def on_epoch_end(self, epoch: int, log_dir: Path):
        pass

    def on_early_end(self, epoch: int, log_dir: Path):
        pass

    def on_train_end(self, log_dir: Path):
        pass
