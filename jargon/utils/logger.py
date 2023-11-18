import datetime
from pathlib import Path
from typing import Any, Mapping

import wandb


class BaseLogger:
    """The base class for loggers."""

    def log(self, epoch: int, data: Mapping[str, Any]) -> None:
        pass

    def log_movies(self, epoch: int, movies: Mapping[str, Path]) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class DummyLogger(BaseLogger):
    """The dummy logger."""

    pass


class WandbLogger(BaseLogger):
    """The logger for wandb."""

    def __init__(self, prefix: str = "", **wandb_config: Any) -> None:
        self.prefix = prefix
        if "name" not in wandb_config:
            dt = datetime.datetime.now()
            name = dt.strftime("%Y/%m/%d %H:%M:%S.%f")
            wandb_config["name"] = name
        wandb.init(**wandb_config)

    def log(self, epoch: int, data: Mapping[str, Any]) -> None:
        metrics = {f"{self.prefix}{k}": v for k, v in data.items()}
        wandb.log(metrics, step=epoch, commit=False)

    def log_movies(self, epoch: int, movies: Mapping[str, Path]) -> None:
        metrics = {
            f"{self.prefix}{k}": wandb.Video(v.as_posix()) for k, v in movies.items()
        }
        wandb.log(metrics, step=epoch, commit=False)

    def flush(self) -> None:
        wandb.log({}, commit=True)

    def close(self) -> None:
        wandb.finish()
