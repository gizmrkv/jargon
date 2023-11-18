from typing import Any, Dict, Tuple, Type

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


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
        cell_type: Type[nn.Module] | str = nn.LSTM,
        cell_args: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if isinstance(cell_type, str):
            cell_type = getattr(nn, cell_type)

        self.cell_type = cell_type
        self.cell_args = cell_args or {}

        self.cells = cell_type(  # type: ignore
            input_dim, hidden_size, num_layers, batch_first=True, **self.cell_args
        )

    def forward(self, x: Tensor, state: Any = None) -> Tuple[Tensor, Any]:
        x, state = self.cells(x, state)
        return x, state

    def generate_discrete_sequence(
        self,
        length: int,
        embedding: nn.Embedding,
        start_embedding: Tensor,
        state: Any = None,
        output_layer: nn.Module | None = None,
    ) -> Tuple[Tensor, Tensor, Any]:
        """Generate a discrete sequence.

        Parameters
        ----------
        length : int
            The length of the sequence.
        embedding : nn.Embedding
            The embedding layer.
        start_embedding : Tensor
            The start embedding.
            The size is (batch_size, embedding_dim).
        state : Any, optional
            The initial state, by default None
        output_layer : nn.Module | None, optional
            The output layer, by default None

        Returns
        -------
        Tuple[Tensor, Tensor, Any]
            The sequence and the final state.

        Examples
        --------
        >>> rnn = RNN(8, 32)
        >>> sos = torch.randn(8).unsqueeze(0).repeat(100, 1)
        >>> linear = nn.Linear(32, 10)
        >>> emb = nn.Embedding(10, 8)
        >>> y, _, _ = rnn.generate_discrete_sequence(10, emb, sos, output_layer=linear)
        >>> y.shape
        torch.Size([100, 10])
        """
        emb = start_embedding.unsqueeze(1)
        symbol_list = []
        logits_list = []
        for _ in range(length):
            logits_step, state = self(emb, state)
            if output_layer is not None:
                logits_step = output_layer(logits_step)

            if self.training:
                distr = Categorical(logits=logits_step)
                symbol = distr.sample()
            else:
                symbol = logits_step.argmax(dim=-1)

            emb = embedding(symbol)
            symbol_list.append(symbol)
            logits_list.append(logits_step)

        sequence = torch.cat(symbol_list, dim=1)
        logits = torch.cat(logits_list, dim=1)

        return sequence, logits, state
