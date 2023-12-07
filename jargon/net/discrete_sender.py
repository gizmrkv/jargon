from typing import Any, Dict, Tuple, Type

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from .attention import ScaledDotProductAttention
from .rnn import RNN


class DiscreteSender(nn.Module):
    def __init__(
        self,
        num_elems: int,
        num_attrs: int,
        vocab_size: int,
        max_len: int,
        input_embedding_dim: int,
        output_embedding_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        cell_type: Type[nn.Module] | str = nn.LSTM,
        cell_args: Dict[str, Any] | None = None,
        peeky: bool = False,
        attention: bool = False,
        attention_dropout: float = 0.0,
        attention_weight: bool = False,
    ) -> None:
        super().__init__()
        self.num_elems = num_elems
        self.num_attrs = num_attrs
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.input_embedding_dim = input_embedding_dim
        self.output_embedding_dim = output_embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.cell_type = cell_type
        self.cell_args = cell_args
        self.peeky = peeky
        self.attention = attention
        self.attention_dropout = attention_dropout
        self.attention_weight = attention_weight

        self.input_embedding = nn.Embedding(num_elems, input_embedding_dim)
        self.output_embedding = nn.Embedding(vocab_size, output_embedding_dim)
        self.rnn = RNN(
            input_dim=output_embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            cell_type=cell_type,
            cell_args=cell_args,
        )
        self.input_linear = nn.Linear(input_embedding_dim * num_attrs, hidden_size)
        self.output_linear = nn.Linear(hidden_size * (1 + bidirectional), vocab_size)
        self.sos_embedding = nn.Parameter(torch.randn(output_embedding_dim))

        if attention:
            self.embed_to_hidden = nn.Linear(input_embedding_dim, hidden_size)
            self.attention_layer = ScaledDotProductAttention(
                embed_dim=hidden_size,
                dropout=attention_dropout,
                weight=attention_weight,
            )

    def forward(
        self, x: Tensor, message: Tensor | None = None
    ) -> Tuple[Tensor, Tensor]:
        x = self.input_embedding(x)

        if self.attention:
            value = self.embed_to_hidden(x)

        x = x.reshape(x.shape[0], -1)
        x = self.input_linear(x)
        hidden = x.repeat(self.num_layers * (1 + self.bidirectional), 1, 1)
        hidden0 = hidden
        if isinstance(self.rnn.cells, nn.LSTM):
            hidden = (hidden, torch.zeros_like(hidden))

        emb = self.sos_embedding.repeat(x.shape[0], 1).unsqueeze(1)
        if message is None:
            symbol_list = []
            logits_list = []
            for _ in range(self.max_len):
                logits_step, hidden = self.rnn(emb, hidden)
                logits_step = self.output_linear(logits_step)

                if self.training:
                    distr = Categorical(logits=logits_step)
                    symbol = distr.sample()
                else:
                    symbol = logits_step.argmax(dim=-1)

                emb = self.output_embedding(symbol)
                symbol_list.append(symbol)
                logits_list.append(logits_step)

                if self.peeky:
                    if isinstance(self.rnn.cells, nn.LSTM):
                        hidden[0] = hidden[0] + hidden0
                    else:
                        hidden = hidden + hidden0

                if self.attention:
                    if isinstance(self.rnn.cells, nn.LSTM):
                        attn, _ = self.attention_layer(
                            hidden[0].transpose(0, 1), value, value
                        )
                        hidden[0] = hidden[0] + attn.transpose(0, 1)
                    else:
                        attn, _ = self.attention_layer(
                            hidden.transpose(0, 1), value, value
                        )
                        hidden = hidden + attn.transpose(0, 1)

            sequence = torch.cat(symbol_list, dim=1)
            logits = torch.cat(logits_list, dim=1)
        else:
            emb = torch.cat([emb, self.output_embedding(message[:, :-1])], dim=1)
            logits, _ = self.rnn(emb, hidden)
            logits = self.output_linear(logits)
            if self.training:
                distr = Categorical(logits=logits)
                sequence = distr.sample()
            else:
                sequence = logits.argmax(-1)

        return sequence, logits
