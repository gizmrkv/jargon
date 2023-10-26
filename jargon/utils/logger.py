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

    def __init__(self, prefix: str = "", **wandb_config: Any) -> None:
        self.prefix = prefix
        wandb.init(**wandb_config)

    def log(self, epoch: int, data: Mapping[str, Any]) -> None:
        metrics = {f"{self.prefix}{k}": v for k, v in data.items()}
        wandb.log(metrics, step=epoch, commit=False)

    def flush(self) -> None:
        wandb.log({}, commit=True)

    def close(self) -> None:
        wandb.finish()
