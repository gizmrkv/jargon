from typing import Any, Dict, Tuple, Type

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

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
    input_dim : int
        The dimension of the input.
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
    >>> encoder = nn.Linear(10, 32)
    >>> sender = Sender(
    ...     encoder=encoder,
    ...     input_dim=32,
    ...     vocab_size=50,
    ...     length=10,
    ...     embedding_dim=8,
    ...     hidden_size=64,
    ...     num_layers=2,
    ...     cell_type=nn.GRU,
    ... )
    >>> x = torch.randn(64, 10)
    >>> sequence, aux = sender(x)
    >>> sequence.shape
    torch.Size([64, 10])
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
        self.msg_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.sos_embedding = nn.Parameter(torch.randn(embedding_dim))

    def forward(
        self, x: Tensor, message: Tensor | None = None
    ) -> Tuple[Tensor, Tensor]:
        x = self.encoder(x)
        x = self.input_layer(x)
        hidden = x.repeat(self.num_layers, 1, 1)
        if isinstance(self.decoder.cells, nn.LSTM):
            hidden = (hidden, torch.zeros_like(hidden))

        emb = self.sos_embedding.repeat(x.shape[0], 1).unsqueeze(1)
        if message is None:
            symbol_list = []
            logits_list = []
            for _ in range(self.length):
                logits_step, hidden = self.decoder(emb, hidden)
                logits_step = self.output_layer(logits_step)

                if self.training:
                    distr = Categorical(logits=logits_step)
                    symbol = distr.sample()
                else:
                    symbol = logits_step.argmax(dim=-1)

                emb = self.msg_embedding(symbol)
                symbol_list.append(symbol)
                logits_list.append(logits_step)

            sequence = torch.cat(symbol_list, dim=1)
            logits = torch.cat(logits_list, dim=1)
        else:
            emb = torch.cat([emb, self.msg_embedding(message[:, :-1])], dim=1)
            logits, _ = self.decoder(emb, hidden)
            logits = self.output_layer(logits)
            if self.training:
                distr = Categorical(logits=logits)
                sequence = distr.sample()
            else:
                sequence = logits.argmax(-1)

        return sequence, logits
