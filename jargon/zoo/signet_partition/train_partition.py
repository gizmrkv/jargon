from copy import deepcopy
from typing import Any, Dict, Literal, Type

from torch import nn

from jargon.game import SignalingNetworkGame
from jargon.net import DiscreteReceiver, DiscreteSender
from jargon.zoo.signet.loss import Loss
from jargon.zoo.signet_reset.train_reset import train_reset


def train_partition(
    num_elems: int = 100,
    num_attrs: int = 2,
    vocab_size: int = 100,
    max_len: int = 3,
    entropy_loss_weight: float = 0.5,
    length_loss_weight: float = 0.0,
    num_agents: int = 2,
    network_mode: Literal["fully"] = "fully",
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

    senders = {f"S{i}": deepcopy(sender) for i in range(num_agents)}
    receivers = {f"R{i}": deepcopy(receiver) for i in range(num_agents)}

    if network_mode == "fully":
        network = {s: {r for r in receivers} for s in senders}
    else:
        raise ValueError(f"Unknown graph mode: {network_mode}")

    adaptation_targets = {s: {r for r in receivers} for s in senders}
    adaptation_targets |= {f"R{i}": {f"S{i}"} for i in range(num_agents)}

    game = SignalingNetworkGame(senders, receivers, network)
    loss = Loss(
        num_elems=num_elems,
        num_attrs=num_attrs,
        vocab_size=vocab_size,
        max_len=max_len,
        game=game,
        adaptation_targets=adaptation_targets,
        entropy_loss_weight=entropy_loss_weight,
        length_loss_weight=length_loss_weight,
        discount_factor=discount_factor,
        instantly=instantly,
    )

    train_reset(
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
    from jargon.zoo.utils import wandb_sweep

    wandb_sweep(train_partition)
