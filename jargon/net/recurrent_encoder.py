from typing import Optional, Tuple

from torch import Tensor, nn

from jargon.utils.torchdict import TensorDict

from .net import Net
from .recurrent import RecurrentNet


class RecurrentEncoder(Net):
    def __init__(self, net: RecurrentNet, vocab_size: int, embed_size: int):
        super().__init__()
        assert net.input_size == embed_size
        self.net = net
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.input_size = self.net.input_size
        self.hidden_size = self.net.hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(
        self, input: Tensor, info: Optional[TensorDict] = None
    ) -> Tuple[Tensor, TensorDict]:
        input = self.embedding(input)
        output, hidden_dict = self.net(input)
        hidden_dict.output = output
        hidden = hidden_dict.get_tensor("hidden")
        if self.net.rnn.bidirectional:
            n = self.net.num_layers
            hidden = hidden[:n, :, :] + hidden[n:, :, :]
        return hidden[-1, :, :], hidden_dict
