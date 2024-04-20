from typing import Any, Dict

import wandb

from .logger import Logger


class WandBLogger(Logger):
    def __init__(self, **wandb_config: Any) -> None:
        self.wandb_config = wandb_config

    def begin(self, run_name: str):
        if "name" not in self.wandb_config:
            self.wandb_config["name"] = run_name

        wandb.init(**self.wandb_config)

    def log(self, epoch: int, metrics: Dict[str, Any], flush: bool = False):
        wandb.log(metrics, step=epoch, commit=flush)

    def flush(self):
        wandb.log({}, commit=True)

    def close(self):
        wandb.finish()
