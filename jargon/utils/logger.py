from typing import Any, Mapping

import wandb


class BaseLogger:
    """The base class for loggers."""

    def log(self, epoch: int, data: Mapping[str, Any]) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class WandbLogger(BaseLogger):
    """The logger for wandb."""

    def __init__(self, **wandb_config: Any) -> None:
        wandb.init(**wandb_config)

    def log(self, epoch: int, data: Mapping[str, Any]) -> None:
        wandb.log(data, step=epoch, commit=False)

    def flush(self) -> None:
        wandb.log({}, commit=True)

    def close(self) -> None:
        wandb.finish()
