from abc import ABC, abstractmethod
from typing import Any, Dict


class Logger(ABC):
    def begin(self, run_name: str):
        pass

    @abstractmethod
    def log(self, epoch: int, metrics: Dict[str, Any], flush: bool = False):
        pass

    def flush(self):
        pass

    def close(self):
        pass
