from abc import ABC, abstractmethod


class EarlyStopper(ABC):
    @abstractmethod
    def step(self, loss: float) -> bool:
        pass
