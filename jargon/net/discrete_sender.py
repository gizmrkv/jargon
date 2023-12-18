from typing import Any, Dict, Tuple, Type

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from .attention import ScaledDotProductAttention
from .onehotify import Onehotify
from .rnn import RNN


class DiscreteSender(nn.Module):
    def __init__(
        self,
        num_elems: int,
        num_attrs: int,
        vocab_size: int,
        max_len: int,
        hidden_size: int,
        input_embedding_dim: int | None = None,
        output_embedding_dim: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        cell_type: Type[nn.Module] | str = nn.LSTM,
        cell_args: Dict[str, Any] | None = None,
        peeky: bool = False,
        attention: bool = False,
        attention_weight: bool = False,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_elems = num_elems
        self.num_attrs = num_attrs
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.input_embedding_dim = input_embedding_dim or num_elems
        self.output_embedding_dim = output_embedding_dim or vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.cell_type = cell_type
        self.cell_args = cell_args
        self.peeky = peeky
        self.attention = attention
        self.attention_weight = attention_weight
        self.attention_dropout = attention_dropout

        if input_embedding_dim is None:
            self.input_embedding = Onehotify(num_elems)
        else:
            self.input_embedding = nn.Embedding(num_elems, input_embedding_dim)

        if output_embedding_dim is None:
            self.output_embedding = Onehotify(vocab_size)
        else:
            self.output_embedding = nn.Embedding(vocab_size, output_embedding_dim)

        self.rnn_input_dim = self.output_embedding_dim + (
            hidden_size if attention else 0
        )
        self.rnn = RNN(
            input_dim=self.rnn_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            cell_type=cell_type,
            cell_args=cell_args,
        )
        self.input_linear = nn.Linear(self.input_embedding_dim * num_attrs, hidden_size)
        self.output_linear = nn.Linear(hidden_size, vocab_size)
        self.sos_embedding = nn.Parameter(torch.randn(self.output_embedding_dim))

        if attention:
            self.attention_layer = ScaledDotProductAttention(
                hidden_size, attention_dropout, attention_weight
            )
            self.input_to_hidden = nn.Linear(self.input_embedding_dim, hidden_size)

    def forward(
        self, x: Tensor, message: Tensor | None = None
    ) -> Tuple[Tensor, Tensor]:
        batch_size = x.shape[0]
        # [batch, num_attrs, input_embedding_dim; float]
        input_emb = self.input_embedding(x)
        # [batch, num_attrs * input_embedding_dim; float]
        input_emb_cat = input_emb.reshape(batch_size, -1)
        # [batch, hidden_size; float]
        hidden = self.input_linear(input_emb_cat).unsqueeze(0)

        if self.num_layers >= 2:
            # [num_layers, batch, hidden_size; float]
            hidden = torch.cat(
                [hidden, torch.zeros_like(hidden).repeat(self.num_layers - 1, 1, 1)],
                dim=0,
            )

        if self.peeky:
            # [num_layers, batch, hidden_size; float]
            hidden0 = hidden

        if isinstance(self.rnn.cells, nn.LSTM):
            hidden = (hidden, torch.zeros_like(hidden))

        if self.attention:
            # [batch, num_attrs, hidden_size; float]
            input_src = self.input_to_hidden(input_emb)

        length = self.max_len if message is None else message.shape[1]
        # [batch, 1, output_embedding_dim; float]
        emb = self.sos_embedding.repeat(batch_size, 1).unsqueeze(1)
        symbol_list = []
        logits_list = []
        for i in range(length):
            if self.attention:
                # [num_layers, batch, hidden_size; float]
                tgt = hidden[0] if isinstance(self.rnn.cells, nn.LSTM) else hidden
                # [batch, 1, hidden_size; float]
                tgt = tgt[-1].unsqueeze(1)
                # [batch, 1, hidden_size; float]
                attn, _ = self.attention_layer(tgt, input_src, input_src)
                # [batch, 1, output_embedding_dim + hidden_size; float]
                emb = torch.cat([emb, attn], dim=-1)

            # [batch, 1, hidden_size; float], [num_layers, batch, hidden_size; float]
            logits_step, hidden = self.rnn(emb, hidden)
            # [batch, 1, vocab_size; float]
            logits_step = self.output_linear(logits_step)

            if message is not None:
                # [batch, 1; int]
                symbol = message[:, i].unsqueeze(1)
            elif self.training:
                distr = Categorical(logits=logits_step)
                # [batch, 1; int]
                symbol = distr.sample()
            else:
                # [batch, 1; int]
                symbol = logits_step.argmax(dim=-1)

            # [batch, 1, output_embedding_dim; float]
            emb = self.output_embedding(symbol)
            symbol_list.append(symbol)
            logits_list.append(logits_step)

            if self.peeky:
                if isinstance(self.rnn.cells, nn.LSTM):
                    # [num_layers, batch, hidden_size; float]
                    hidden[0] = hidden[0] + hidden0
                else:
                    # [num_layers, batch, hidden_size; float]
                    hidden = hidden + hidden0

        # [batch, length; int]
        sequence = torch.cat(symbol_list, dim=1)
        # [batch, length, vocab_size; float]
        logits = torch.cat(logits_list, dim=1)

        return sequence, logits
