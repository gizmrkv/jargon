from typing import Any, Dict, Optional, Tuple, Type

import torch
from torch import Tensor, nn

from jargon.utils.torchdict import TensorDict

from .net import Net


class RecurrentNet(Net):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        rnn_type: Type[nn.Module] = nn.LSTM,
        rnn_args: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn = rnn_type(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
            **(rnn_args or {})
        )
        self.init_hidden = nn.Parameter(
            torch.randn(num_layers * (bidirectional + 1), 1, hidden_size)
        )
        if isinstance(self.rnn, nn.LSTM):
            self.init_cell = nn.Parameter(torch.randn(*self.init_hidden.shape))

    def forward(
        self, input: Tensor, info: Optional[TensorDict] = None
    ) -> Tuple[Tensor, TensorDict]:
        batch_size = input.size(0)
        if info is None:
            info = TensorDict(hidden=self.init_hidden.repeat(1, batch_size, 1))

        if isinstance(self.rnn, nn.LSTM) and "cell" not in info:
            info.cell = self.init_cell.repeat(1, batch_size, 1)

        hidden = info.get_tensor("hidden")
        if isinstance(self.rnn, nn.LSTM):
            cell = info.get_tensor("cell")
            hidden = (hidden, cell)

        output, hidden = self.rnn(input, hidden)

        if isinstance(self.rnn, nn.LSTM):
            hidden, cell = hidden
            info = TensorDict(hidden=hidden, cell=cell)
        else:
            info = TensorDict(hidden=hidden)

        return output, info
