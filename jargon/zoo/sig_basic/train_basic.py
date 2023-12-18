from typing import Any, Dict, List, Type

from torch import nn

from jargon.game import SignalingGame
from jargon.net import DiscreteReceiver, DiscreteSender
from jargon.zoo.sig.loss import Loss
from jargon.zoo.sig.train import train


def train_basic(
    num_elems: int = 10,
    num_attrs: int = 3,
    vocab_size: int = 10,
    max_len: int = 10,
    entropy_loss_weight: float = 0.5,
    length_loss_weight: float = 0.0,
    sender_input_embedding_dim: int | None = None,
    sender_output_embedding_dim: int | None = None,
    sender_hidden_size: int = 500,
    sender_num_layers: int = 1,
    sender_cell_type: Type[nn.Module] | str = nn.GRU,
    sender_cell_args: Dict[str, Any] | None = None,
    sender_peeky: bool = False,
    sender_attention: bool = False,
    sender_attention_weight: bool = False,
    sender_attention_dropout: float = 0.0,
    receiver_embedding_dim: int | None = None,
    receiver_hidden_size: int = 500,
    receiver_num_layers: int = 1,
    receiver_bidirectional: bool = False,
    receiver_cell_type: Type[nn.Module] | str = nn.GRU,
    receiver_cell_args: Dict[str, Any] | None = None,
    receiver_attention: bool = False,
    receiver_attention_weight: bool = False,
    receiver_attention_dropout: float = 0.0,
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
        cell_type=sender_cell_type,
        cell_args=sender_cell_args,
        peeky=sender_peeky,
        attention=sender_attention,
        attention_weight=sender_attention_weight,
        attention_dropout=sender_attention_dropout,
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
        attention=receiver_attention,
        attention_weight=receiver_attention_weight,
        attention_dropout=receiver_attention_dropout,
    )
    game = SignalingGame(sender, receiver)
    loss = Loss(
        num_elems,
        num_attrs,
        vocab_size,
        max_len,
        entropy_loss_weight,
        length_loss_weight,
    )

    train(
        num_elems=num_elems,
        num_attrs=num_attrs,
        vocab_size=vocab_size,
        max_len=max_len,
        game=game,
        loss_fn=loss,
        additional_metrics_fn=loss.metrics,
        **train_args,
    )


if __name__ == "__main__":
    from jargon.zoo.utils import wandb_sweep

    wandb_sweep(train_basic)
