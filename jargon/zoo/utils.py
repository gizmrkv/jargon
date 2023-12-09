import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict

import toml
import yaml

import wandb
from jargon.utils import WandbLogger


def wandb_sweep(main: Callable[..., Any]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path", "-c", type=str, default=None)
    parser.add_argument("--sweep_id", "-s", type=str, default=None)
    parser.add_argument("--project", "-p", type=str, default="jargon")
    parser.add_argument("--prefix", "-x", type=str, default="")
    args = parser.parse_args()

    config = read_config(args.conf_path) if args.conf_path else None
    sweep_id = args.sweep_id

    if sweep_id is None:
        sweep_id = wandb.sweep(sweep=config, project=args.project)

    def func() -> None:
        logger = WandbLogger(prefix=args.prefix)
        main(logger=logger, **wandb.config)

    wandb.agent(sweep_id, func, project=args.project)


def read_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    if config_path.suffix == ".json":
        with open(config_path) as f:
            config = json.load(f)
    elif config_path.suffix in (".yaml", ".yml"):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
    elif config_path.suffix == ".toml":
        with open(config_path) as f:
            config = toml.load(f)
    else:
        raise ValueError(f"Unknown config file type: {config_path.suffix}")
    return config
