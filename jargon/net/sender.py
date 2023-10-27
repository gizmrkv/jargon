from typing import Any, Dict, Tuple, Type

import torch
from torch import Tensor, nn

from .rnn import RNN


class Sender(nn.Module):
    """The agent that sends the message.

    The model that converts the feature vector into an integer sequence using RNN.
    The output size of the encoder is (batch_size, hidden_size).
    The output size of the decoder is (batch_size, max_len) and the type is int.
    The output size of the encoder and the input size of the decoder must be the same.

    Parameters
    ----------
    encoder : nn.Module
        The encoder that converts an input into a feature vector.
    vocab_size : int
        The size of the vocabulary.
    length : int
        The length of the sequence.
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
    >>> encoder = nn.Linear(32, 64)
    >>> sender = Sender(encoder, 50, 10, 32, 64)
    >>> x = torch.randn(100, 32)
    >>> y, _ = sender(x)
    >>> y.shape
    torch.Size([100, 10])
    """

    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        vocab_size: int,
        length: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        cell_type: Type[nn.Module] | str = nn.LSTM,
        cell_args: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.length = length
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.cell_args = cell_args

        self.input_layer = nn.Linear(input_dim, hidden_size)
        self.decoder = RNN(
            input_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            cell_type=cell_type,
            cell_args=cell_args,
        )
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.start = nn.Parameter(torch.randn(embedding_dim))

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        x = self.encoder(x)
        if isinstance(x, tuple):
            x, _ = x
        x = self.input_layer(x)
        hidden = x.unsqueeze(0)
        hidden = x.repeat(self.num_layers, 1, 1)
        start = self.start.repeat(x.shape[0], 1)

        sequence, aux = self.decoder.generate_discrete_sequence(
            self.length,
            self.embedding,
            start,
            hidden,
            self.output_layer,
        )

        return sequence, aux
