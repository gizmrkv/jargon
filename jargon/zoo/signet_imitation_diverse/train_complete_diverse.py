from copy import deepcopy
from typing import Any, Dict, List, Type

from torch import nn

from jargon.game import SignalingNetworkGame
from jargon.net import DiscreteReceiver, DiscreteSender
from jargon.zoo.signet.loss import Loss
from jargon.zoo.signet.train import train
from jargon.zoo.signet_imitation.loss import ImitationLoss


def train_complete_diverse(
    num_elems: int = 50,
    num_attrs: int = 2,
    vocab_size: int = 50,
    max_len: int = 8,
    num_good_senders: int = 1,
    num_poor_senders: int = 1,
    num_receivers: int = 2,
    entropy_loss_weight: float = 0.0,
    length_loss_weight: float = 0.0,
    discount_factor: float = 0.1,
    instantly: bool = False,
    imitation: bool = False,
    imitation_threshold: float = 0.99,
    poor_sender_input_embedding_dim: int = 16,
    poor_sender_output_embedding_dim: int = 16,
    poor_sender_hidden_size: int = 500,
    poor_sender_num_layers: int = 1,
    poor_sender_bidirectional: bool = False,
    poor_sender_cell_type: Type[nn.Module] | str = nn.GRU,
    poor_sender_cell_args: Dict[str, Any] | None = None,
    good_sender_input_embedding_dim: int = 16,
    good_sender_output_embedding_dim: int = 16,
    good_sender_hidden_size: int = 500,
    good_sender_num_layers: int = 1,
    good_sender_bidirectional: bool = False,
    good_sender_cell_type: Type[nn.Module] | str = nn.GRU,
    good_sender_cell_args: Dict[str, Any] | None = None,
    receiver_embedding_dim: int = 16,
    receiver_hidden_size: int = 500,
    receiver_num_layers: int = 1,
    receiver_bidirectional: bool = False,
    receiver_cell_type: Type[nn.Module] | str = nn.GRU,
    receiver_cell_args: Dict[str, Any] | None = None,
    **train_args: Any,
) -> None:
    poor_sender = DiscreteSender(
        num_elems=num_elems,
        num_attrs=num_attrs,
        vocab_size=vocab_size,
        max_len=max_len,
        input_embedding_dim=poor_sender_input_embedding_dim,
        output_embedding_dim=poor_sender_output_embedding_dim,
        hidden_size=poor_sender_hidden_size,
        num_layers=poor_sender_num_layers,
        bidirectional=poor_sender_bidirectional,
        cell_type=poor_sender_cell_type,
        cell_args=poor_sender_cell_args,
    )
    good_sender = DiscreteSender(
        num_elems=num_elems,
        num_attrs=num_attrs,
        vocab_size=vocab_size,
        max_len=max_len,
        input_embedding_dim=good_sender_input_embedding_dim,
        output_embedding_dim=good_sender_output_embedding_dim,
        hidden_size=good_sender_hidden_size,
        num_layers=good_sender_num_layers,
        bidirectional=good_sender_bidirectional,
        cell_type=good_sender_cell_type,
        cell_args=good_sender_cell_args,
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
    senders = {}
    senders |= {f"gS{i}": deepcopy(good_sender) for i in range(num_good_senders)}
    senders |= {f"pS{i}": deepcopy(poor_sender) for i in range(num_poor_senders)}
    receivers = {f"R{i}": deepcopy(receiver) for i in range(num_receivers)}

    network = {s: {r for r in receivers} for s in senders}
    adaptation_targets = {s: {r for r in receivers} for s in senders}
    adaptation_targets |= {r: {s for s in senders} for r in receivers}
    if imitation:
        imitation_targets = {s1: {s2 for s2 in senders if s1 != s2} for s1 in senders}
        imitation_triggers = {s: {r for r in receivers} for s in senders}
    else:
        imitation_targets = None
        imitation_triggers = None

    game = SignalingNetworkGame(senders, receivers, network)
    loss = Loss(
        num_elems=num_elems,
        num_attrs=num_attrs,
        vocab_size=vocab_size,
        max_len=max_len,
        game=game,
        entropy_loss_weight=entropy_loss_weight,
        length_loss_weight=length_loss_weight,
        adaptation_targets=adaptation_targets,
        instantly=instantly,
        discount_factor=discount_factor,
    )
    loss = ImitationLoss(
        loss=loss,
        imitation_targets=imitation_targets,
        imitation_triggers=imitation_triggers,
        imitation_threshold=imitation_threshold,
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
    from jargon.zoo.utils import wandb_sweep

    wandb_sweep(train_complete_diverse)
