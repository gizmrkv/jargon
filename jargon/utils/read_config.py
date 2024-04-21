import json
from pathlib import Path
from typing import Any, Dict

import toml
import yaml


def read_config(config_path: Path) -> Dict[str, Any]:
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
