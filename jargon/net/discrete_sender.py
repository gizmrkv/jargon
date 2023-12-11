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
        self.input_embedding_dim = input_embedding_dim
        self.output_embedding_dim = output_embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.cell_type = cell_type
        self.cell_args = cell_args
        self.peeky = peeky
        self.attention = attention
        self.attention_weight = attention_weight
        self.attention_dropout = attention_dropout

        self.input_embedding = nn.Embedding(num_elems, input_embedding_dim)
        self.attr_embedding = nn.Parameter(torch.randn(num_attrs, input_embedding_dim))
        self.output_embedding = nn.Embedding(vocab_size, output_embedding_dim)
        self.rnn = RNN(
            input_dim=output_embedding_dim + (hidden_size if attention else 0),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            cell_type=cell_type,
            cell_args=cell_args,
        )
        self.input_linear = nn.Linear(input_embedding_dim * num_attrs, hidden_size)
        self.output_linear = nn.Linear(hidden_size, vocab_size)
        self.sos_embedding = nn.Parameter(torch.randn(output_embedding_dim))

        if attention:
            self.attention_layer = ScaledDotProductAttention(
                hidden_size, attention_dropout, attention_weight
            )
            self.input_to_hidden = nn.Linear(input_embedding_dim, hidden_size)

    def forward(
        self, x: Tensor, message: Tensor | None = None
    ) -> Tuple[Tensor, Tensor]:
        batch_size = x.shape[0]
        input_emb = self.input_embedding(x)
        attr_emb = self.attr_embedding.unsqueeze(0).repeat(batch_size, 1, 1)
        input_emb = input_emb + attr_emb
        input_emb_cat = input_emb.reshape(batch_size, -1)
        hidden = self.input_linear(input_emb_cat).unsqueeze(0)

        if self.num_layers >= 2:
            hidden = torch.cat(
                [hidden, torch.zeros_like(hidden).repeat(self.num_layers - 1, 1, 1)],
                dim=0,
            )

        if self.peeky:
            hidden0 = hidden

        if isinstance(self.rnn.cells, nn.LSTM):
            hidden = (hidden, torch.zeros_like(hidden))

        if self.attention:
            input_src = self.input_to_hidden(input_emb)

        length = self.max_len if message is None else message.shape[1]
        emb = self.sos_embedding.repeat(batch_size, 1).unsqueeze(1)
        symbol_list = []
        logits_list = []
        for i in range(length):
            if self.attention:
                tgt = hidden[0] if isinstance(self.rnn.cells, nn.LSTM) else hidden
                tgt = tgt[-1].unsqueeze(1)
                attn, _ = self.attention_layer(tgt, input_src, input_src)
                emb = torch.cat([emb, attn], dim=-1)

            logits_step, hidden = self.rnn(emb, hidden)
            logits_step = self.output_linear(logits_step)

            if message is not None:
                symbol = message[:, i].unsqueeze(1)
            elif self.training:
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

        sequence = torch.cat(symbol_list, dim=1)
        logits = torch.cat(logits_list, dim=1)

        return sequence, logits
