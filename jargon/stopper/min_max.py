from collections import deque

from .stopper import EarlyStopper


class MinMaxEarlyStopper(EarlyStopper):
    def __init__(self, threshold: float, window_size: int):
        self.threshold = threshold
        self.window_size = window_size
        self.losses = deque(maxlen=window_size)

    def step(self, loss: float) -> bool:
        self.losses.append(loss)

        if len(self.losses) == self.window_size:
            min_loss = min(self.losses)
            max_loss = max(self.losses)
            return max_loss - min_loss <= self.threshold
        else:
            return False
