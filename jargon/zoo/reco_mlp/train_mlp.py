from typing import Any, Dict, List, Type

from torch import nn

from jargon.game import SupervisedGame
from jargon.net import MultiDiscreteMLP
from jargon.zoo.reco.loss import Loss
from jargon.zoo.reco.train import train


def train_mlp(
    num_elems: int = 50,
    num_attrs: int = 2,
    embedding_dim: int = 8,
    hidden_sizes: List[int] = [64],
    activation_type: Type[nn.Module] | str = "GELU",
    activation_args: Dict[str, Any] | None = None,
    normalization_type: Type[nn.Module] | str | None = None,
    normalization_args: Dict[str, Any] | None = None,
    dropout: float = 0.0,
    **train_args: Any,
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
    train(
        num_elems=num_elems,
        num_attrs=num_attrs,
        game=game,
        loss_fn=loss,
        additional_metrics_fn=loss.metrics,
        **train_args,
    )


if __name__ == "__main__":
    import argparse

    from jargon.zoo.utils import read_config, wandb_sweep

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path", "-c", type=str, default=None)
    parser.add_argument("--sweep_id", "-s", type=str, default=None)
    args = parser.parse_args()

    conf = read_config(args.conf_path) if args.conf_path else None
    wandb_sweep(train_mlp, conf, args.sweep_id)
