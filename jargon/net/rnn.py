from typing import Any, Dict, Tuple, Type

from torch import Tensor, nn


class RNN(nn.Module):
    """A wrapper class for RNN, LSTM, and GRU.

    Parameters
    ----------
    input_dim : int
        The size of the input.
    hidden_size : int
        The size of the hidden state.
    num_layers : int, optional
        The number of layers, by default 1
    cell_type : Type[nn.Module], optional
        The type of the cell, by default nn.LSTM
    cell_args : Dict[str, Any], optional
        The arguments for the cell, by default None

    Examples
    --------
    >>> rnn = RNN(32, 64)
    >>> x = torch.randn(100, 10, 32)
    >>> y, _ = rnn(x)
    >>> y.shape
    torch.Size([100, 10, 64])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        cell_type: Type[nn.Module] | str = nn.LSTM,
        cell_args: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        if isinstance(cell_type, str):
            cell_type = getattr(nn, cell_type)

        self.cell_type = cell_type
        self.cell_args = cell_args or {}

        self.cells = cell_type(
            input_dim,
            hidden_size,
            num_layers,
            bidirectional=bool(bidirectional),
            dropout=dropout,
            batch_first=True,
            **self.cell_args
        )

    def forward(self, x: Tensor, state: Any = None) -> Tuple[Tensor, Any]:
        x, state = self.cells(x, state)
        return x, state
