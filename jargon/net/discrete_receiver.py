from typing import Any, Dict, Type

from torch import Tensor, nn
from torch.distributions import Categorical

from .rnn import RNN


class DiscreteReceiver(nn.Module):
    def __init__(
        self,
        num_elems: int,
        num_attrs: int,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        cell_type: Type[nn.Module] | str = nn.LSTM,
        cell_args: Dict[str, Any] | None = None,
        instantly: bool = False,
    ) -> None:
        super().__init__()
        self.num_elems = num_elems
        self.num_attrs = num_attrs
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.cell_type = cell_type
        self.cell_args = cell_args
        self.instantly = instantly

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = RNN(
            input_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            cell_type=cell_type,
            cell_args=cell_args,
        )
        self.output_linear = nn.Linear(
            hidden_size * (1 + bidirectional), num_attrs * num_elems
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        logits, _ = self.rnn(x)
        if not self.instantly:
            logits = logits[:, -1, :]

        logits = self.output_linear(logits)
        if self.instantly:
            logits = logits.reshape(-1, x.shape[1], self.num_attrs, self.num_elems)
        else:
            logits = logits.reshape(-1, self.num_attrs, self.num_elems)

        if self.training:
            distr = Categorical(logits=logits)
            output = distr.sample()
        else:
            output = logits.argmax(dim=-1)

        return output, logits
