from typing import Any, Dict, Type

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from .attention import ScaledDotProductAttention
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
        attention: bool = False,
        attention_weight: bool = False,
        attention_dropout: float = 0.0,
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
        self.attention = attention
        self.attention_weight = attention_weight
        self.attention_dropout = attention_dropout

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
        self.output_layers = nn.ModuleList(
            [
                MLP(
                    input_dim=hidden_size + (hidden_size if attention else 0),
                    output_dim=num_elems,
                    hidden_sizes=[hidden_size],
                    activation_type=nn.GELU,
                    normalization_type=nn.LayerNorm,
                )
                for _ in range(num_attrs)
            ]
        )

        if attention:
            self.attention_layer = ScaledDotProductAttention(
                hidden_size, attention_dropout, attention_weight
            )
            self.queries = nn.ModuleList(
                [nn.Linear(hidden_size, hidden_size) for _ in range(num_attrs)]
            )

    def forward(self, x: Tensor) -> Tensor:
        emb = self.embedding(x)
        outputs, _ = self.rnn(emb)

        if self.bidirectional:
            outputs = (
                outputs[:, :, : self.hidden_size] + outputs[:, :, self.hidden_size :]
            )

        output = outputs.sum(dim=1)
        output = output.unsqueeze(1).repeat(1, self.num_attrs, 1)

        if self.attention:
            tgt = torch.stack(
                [layer(output[:, i, :]) for i, layer in enumerate(self.queries)], dim=1
            )
            attn, _ = self.attention_layer(tgt, outputs, outputs)
            output = torch.cat([output, attn], dim=-1)

        logits_list = [
            layer(output[:, i, :]) for i, layer in enumerate(self.output_layers)
        ]
        logits = torch.stack(logits_list, dim=1)

        if self.training:
            distr = Categorical(logits=logits)
            output = distr.sample()
        else:
            output = logits.argmax(dim=-1)

        return output, logits
