import argparse
from typing import Any, Callable, Dict

import wandb
from jargon.utils import date_to_str, read_config

from .logger import Logger


class WandBLogger(Logger):
    def __init__(self, **wandb_config: Any) -> None:
        wandb.init(**wandb_config)

    def log(self, epoch: int, metrics: Dict[str, Any], flush: bool = False):
        wandb.log(metrics, step=epoch, commit=flush)

    def flush(self):
        wandb.log({}, commit=True)

    def close(self):
        wandb.finish()


def wandb_sweep(main: Callable[..., Any]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path", "-c", type=str, default=None)
    parser.add_argument("--sweep_id", "-s", type=str, default=None)
    args = parser.parse_args()

    config = read_config(args.conf_path) if args.conf_path else None
    sweep_id = args.sweep_id

    if sweep_id is None:
        sweep_id = wandb.sweep(sweep=config)

    def func() -> None:
        name = date_to_str()
        wandb_logger = WandBLogger(name=name, project="jargon")
        main(wandb_logger=wandb_logger, **wandb.config)

    wandb.agent(sweep_id, func)
