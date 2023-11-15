from typing import Any, Dict, List, Type

from torch import nn

from jargon.game import SupervisedGame
from jargon.net import MultiDiscreteMLP

from .loss import Loss
from .train import train


def train_mlp(
    num_elems: int,
    num_attrs: int,
    embedding_dim: int,
    hidden_sizes: List[int],
    activation_type: Type[nn.Module] | str = nn.ReLU,
    activation_args: Dict[str, Any] | None = None,
    normalization_type: Type[nn.Module] | str | None = None,
    normalization_args: Dict[str, Any] | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> None:
    model = MultiDiscreteMLP(
        high=num_elems,
        n=num_attrs,
        output_dim=num_elems * num_attrs,
        embedding_dim=embedding_dim,
        hidden_sizes=hidden_sizes,
        activation_type=activation_type,
        activation_args=activation_args,
        normalization_type=normalization_type,
        normalization_args=normalization_args,
        dropout=dropout,
    )
    game = SupervisedGame(model)
    loss = Loss(num_elems, num_attrs)
    train(game=game, loss_fn=loss, **kwargs)
