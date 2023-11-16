import datetime
from typing import Any, Callable, Dict

import wandb
from jargon.utils import WandbLogger


def wandb_sweep(
    config: Dict[str, Any],
    main: Callable[..., Any],
    sweep_id: str | None = None,
    project: str = "jargon",
    prefix: str = "",
) -> None:
    if sweep_id is None:
        sweep_id = wandb.sweep(sweep=config, project=project)

    def func() -> None:
        logger = WandbLogger(prefix=prefix)
        main(logger=logger, **wandb.config)

    wandb.agent(sweep_id, func, project=project)
