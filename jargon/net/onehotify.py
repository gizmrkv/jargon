from torch import Tensor, nn


class Onehotify(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.one_hot(x, self.num_classes).float()
