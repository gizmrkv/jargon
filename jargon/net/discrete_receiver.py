from typing import Any, Dict, Type

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from .mlp import MLP
from .onehotify import Onehotify
from .rnn import RNN


class DiscreteReceiver(nn.Module):
    def __init__(
        self,
        num_elems: int,
        num_attrs: int,
        vocab_size: int,
        hidden_size: int,
        embedding_dim: int | None = None,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        cell_type: Type[nn.Module] | str = nn.LSTM,
        cell_args: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.num_elems = num_elems
        self.num_attrs = num_attrs
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim or vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.cell_type = cell_type
        self.cell_args = cell_args

        if embedding_dim is None:
            self.embedding = Onehotify(vocab_size)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = RNN(
            input_dim=self.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            cell_type=cell_type,
            cell_args=cell_args,
        )
        self.output_layer = MLP(
            input_dim=hidden_size,
            output_dim=num_attrs * num_elems,
            hidden_sizes=[hidden_size],
            activation_type=nn.GELU,
            normalization_type=nn.LayerNorm,
        )

    def forward(self, x: Tensor) -> Tensor:
        # [batch, max_len, embedding_dim; float]
        emb = self.embedding(x)
        # [batch, max_len, hidden_size * (bidirectional + 1); float]
        outputs, _ = self.rnn(emb)

        if self.bidirectional:
            # [batch, max_len, hidden_size; float]
            outputs = (
                outputs[:, :, : self.hidden_size] + outputs[:, :, self.hidden_size :]
            )

        # [batch, hidden_size; float]
        output = outputs.sum(dim=1)
        # [batch, num_attrs * num_elems; float]
        logits = self.output_layer(output)
        # [batch, num_attrs, num_elems; float]
        logits = logits.reshape(-1, self.num_attrs, self.num_elems)

        if self.training:
            distr = Categorical(logits=logits)
            # [batch, num_attrs; int]
            output = distr.sample()
        else:
            # [batch, num_attrs; int]
            output = logits.argmax(dim=-1)

        return output, logits
