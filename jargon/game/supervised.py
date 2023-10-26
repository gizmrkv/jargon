from torch import Tensor, nn

from ..core import Batch


class SupervisedGame(nn.Module):
    """Supervised game

    Parameters
    ----------
    model : nn.Module
        The model that receives the input and produces an output.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input: Tensor, target: Tensor) -> Batch:
        output, output_aux = self.model(input)
        batch = Batch(
            input=input,
            output=output,
            target=target,
        )
        if output_aux is not None:
            batch.output_aux = output_aux

        return batch
