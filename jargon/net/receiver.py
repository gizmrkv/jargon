from typing import Any, Dict, Tuple, Type

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

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
    num_elems : int
        The number of elements.
    num_attrs : int
        The number of attributes.
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
    >>> decoder = nn.Linear(32, 20)
    >>> receiver = Receiver(
    ...     decoder=decoder,
    ...     vocab_size=50,
    ...     output_dim=32,
    ...     num_elems=10,
    ...     num_attrs=2,
    ...     embedding_dim=8,
    ...     hidden_size=64,
    ...     num_layers=2,
    ...     cell_type=nn.GRU,
    ... )
    >>> x = torch.randint(0, 50, (64, 10))
    >>> output, logits = receiver(x)
    >>> output.shape
    torch.Size([64, 2])
    """

    def __init__(
        self,
        decoder: nn.Module,
        vocab_size: int,
        output_dim: int,
        num_elems: int,
        num_attrs: int,
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
        self.num_elems = num_elems
        self.num_attrs = num_attrs
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

    def forward(self, x: Tensor) -> Tensor:
        emb = self.embedding(x)
        _, hidden = self.encoder(emb)
        if isinstance(hidden, tuple):
            hidden, _ = hidden
        hidden = hidden[-1]
        logits = self.output_layer(hidden)
        logits = self.decoder(logits)
        logits = logits.reshape(-1, self.num_attrs, self.num_elems)
        if self.training:
            distr = Categorical(logits=logits)
            output = distr.sample()
        else:
            output = logits.argmax(dim=-1)

        return output, logits
