from typing import Any, Dict, Tuple, Type

import torch
from torch import Tensor, nn

from .rnn import RNN


class Receiver(nn.Module):
    """The agent that receives the message and produces an output.

    The model that converts the message into a feature vector and inputs it to the decoder using RNN.
    The size of the message is represented as (batch_size, max_len) and the type is int.
    The output of the encoder is represented as (batch_size, max_len, hidden_size).
    The output size of the encoder and the input size of the decoder must be the same.

    Parameters
    ----------
    decoder : nn.Module
        The decoder that converts the feature vector into an output.
    vocab_size : int
        The size of the vocabulary.
    output_dim : int
        The dimension of the output.
    embedding_dim : int
        The dimension of the embedding.
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
    >>> decoder = nn.Linear(32, 5)
    >>> receiver = Receiver(
    ...     decoder=decoder,
    ...     vocab_size=50,
    ...     output_dim=32,
    ...     embedding_dim=8,
    ...     hidden_size=64,
    ...     num_layers=2,
    ...     cell_type=nn.GRU,
    ... )
    >>> x = torch.randint(0, 50, (64, 10))
    >>> output = receiver(x)
    >>> output.shape
    torch.Size([64, 10, 5])
    """

    def __init__(
        self,
        decoder: nn.Module,
        vocab_size: int,
        output_dim: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        cell_type: Type[nn.Module] | str = nn.LSTM,
        cell_args: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.cell_args = cell_args

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = RNN(
            input_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            cell_type=cell_type,
            cell_args=cell_args,
        )
        self.output_layer = nn.Linear(hidden_size, output_dim)
        self.start_hidden = nn.Parameter(torch.randn(num_layers, hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        emb = self.embedding(x)
        hidden = self.start_hidden.unsqueeze(1).repeat(1, x.shape[0], 1)

        output, _ = self.encoder(emb, hidden)
        output = self.output_layer(output)
        output = self.decoder(output)
        return output
