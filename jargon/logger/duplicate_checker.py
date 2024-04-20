from typing import Any, Dict

from .logger import Logger


class DuplicateChecker(Logger):
    def __init__(self):
        self.metrics = {}

    def log(self, epoch: int, metrics: Dict[str, Any], flush: bool = False):
        old_keys = set(self.metrics.keys())
        new_keys = set(metrics.keys())
        keys_intersection = old_keys & new_keys
        assert (
            len(keys_intersection) == 0
        ), f"Duplicate metric keys detected: {keys_intersection}"
        self.metrics |= metrics

    def flush(self):
        self.metrics.clear()
