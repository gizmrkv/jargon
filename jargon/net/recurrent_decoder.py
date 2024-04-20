from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from jargon.utils.torchdict import TensorDict

from .net import Net
from .recurrent import RecurrentNet


class RecurrentDecoder(Net):
    def __init__(
        self,
        net: RecurrentNet,
        vocab_size: int,
        embed_size: int,
        seq_length: int,
        train_temp: float = 1.0,
        eval_temp: float = 1e-5,
        peeky: bool = False,
    ):
        super().__init__()
        assert not net.bidirectional
        assert net.input_size == embed_size
        self.net = net
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.seq_length = seq_length
        self.train_temp = train_temp
        self.eval_temp = eval_temp
        self.peeky = peeky

        self.input_size = self.net.input_size
        self.hidden_size = self.net.hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(net.hidden_size, vocab_size)
        self.sos_emb = nn.Parameter(torch.randn(embed_size))

    def forward(
        self, input: Tensor, info: Optional[TensorDict] = None
    ) -> Tuple[Tensor, TensorDict]:
        target = None if info is None else info.get_tensor("target")
        batch_size = input.size(0)
        hidden = torch.cat(
            [
                input.unsqueeze(0),
                self.net.init_hidden[1:, :, :].repeat(1, batch_size, 1),
            ],
            dim=0,
        )

        hidden_dict = TensorDict(hidden=hidden)
        h0 = hidden

        length = self.seq_length if target is None else target.size(1)
        emb = self.sos_emb.repeat(batch_size, 1).unsqueeze(1)
        symbol_list = []
        logits_list = []
        for i in range(length):
            logits_step, hidden_dict = self.net(emb, hidden_dict)
            logits_step = self.linear(logits_step)
            logits_step /= self.train_temp if self.training else self.eval_temp

            distr = Categorical(logits=logits_step)
            symbol = distr.sample()
            emb = self.embedding(
                symbol if target is None else target[:, i].unsqueeze(1)
            )
            symbol_list.append(symbol)
            logits_list.append(logits_step)

            if self.peeky:
                hidden_dict.hidden += h0

        symbols = torch.cat(symbol_list, dim=1)
        logits = torch.cat(logits_list, dim=1)
        info = TensorDict(logits=logits)
        return symbols, info
