from .stopper import EarlyStopper


class DummyEarlyStopper(EarlyStopper):
    def step(self, loss: float) -> bool:
        return False
