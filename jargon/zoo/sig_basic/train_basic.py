from typing import Any, Dict, List, Type

from torch import nn

from jargon.game import SignalingGame
from jargon.net import DiscreteReceiver, DiscreteSender
from jargon.zoo.sig.loss import Loss
from jargon.zoo.sig.train import train


def train_basic(
    num_elems: int = 100,
    num_attrs: int = 2,
    vocab_size: int = 100,
    max_len: int = 3,
    entropy_loss_weight: float = 0.5,
    length_loss_weight: float = 0.0,
    discount_factor: float = 0.1,
    instantly: bool = False,
    sender_input_embedding_dim: int = 16,
    sender_output_embedding_dim: int = 16,
    sender_hidden_size: int = 500,
    sender_num_layers: int = 1,
    sender_bidirectional: bool = False,
    sender_cell_type: Type[nn.Module] | str = nn.GRU,
    sender_cell_args: Dict[str, Any] | None = None,
    sender_peeky: bool = False,
    sender_attention: bool = False,
    sender_attention_dropout: float = 0.0,
    sender_attention_weight: bool = False,
    receiver_embedding_dim: int = 16,
    receiver_hidden_size: int = 500,
    receiver_num_layers: int = 1,
    receiver_bidirectional: bool = False,
    receiver_cell_type: Type[nn.Module] | str = nn.GRU,
    receiver_cell_args: Dict[str, Any] | None = None,
    **train_args: Any,
) -> None:
    sender = DiscreteSender(
        num_elems=num_elems,
        num_attrs=num_attrs,
        vocab_size=vocab_size,
        max_len=max_len,
        input_embedding_dim=sender_input_embedding_dim,
        output_embedding_dim=sender_output_embedding_dim,
        hidden_size=sender_hidden_size,
        num_layers=sender_num_layers,
        bidirectional=sender_bidirectional,
        cell_type=sender_cell_type,
        cell_args=sender_cell_args,
        peeky=sender_peeky,
        attention=sender_attention,
        attention_dropout=sender_attention_dropout,
        attention_weight=sender_attention_weight,
    )
    receiver = DiscreteReceiver(
        num_elems=num_elems,
        num_attrs=num_attrs,
        vocab_size=vocab_size,
        embedding_dim=receiver_embedding_dim,
        hidden_size=receiver_hidden_size,
        num_layers=receiver_num_layers,
        bidirectional=receiver_bidirectional,
        cell_type=receiver_cell_type,
        cell_args=receiver_cell_args,
        instantly=instantly,
    )
    game = SignalingGame(sender, receiver)
    loss = Loss(
        num_elems,
        num_attrs,
        vocab_size,
        max_len,
        entropy_loss_weight,
        length_loss_weight,
        instantly,
        discount_factor,
    )

    train(
        num_elems=num_elems,
        num_attrs=num_attrs,
        vocab_size=vocab_size,
        max_len=max_len,
        instantly=instantly,
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
    wandb_sweep(train_basic, conf, args.sweep_id)
