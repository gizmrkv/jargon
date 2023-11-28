from copy import deepcopy
from typing import Any, Dict, List, Type

from torch import nn

from jargon.game import SignalingNetworkGame
from jargon.net import MLP, MultiDiscreteMLP, Receiver, Sender
from jargon.zoo.signet.loss import Loss
from jargon.zoo.signet.train import train
from jargon.zoo.signet_imitation.loss import ImitationLoss


def train_complete(
    num_elems: int = 50,
    num_attrs: int = 2,
    vocab_size: int = 50,
    max_len: int = 8,
    num_good_senders: int = 1,
    num_poor_senders: int = 1,
    num_receivers: int = 2,
    entropy_loss_weight: float = 0.0,
    length_loss_weight: float = 0.0,
    imitation: bool = False,
    imitation_threshold: float = 0.99,
    good_encoder_embedding_dim: int = 8,
    good_encoder_hidden_sizes: List[int] = [64],
    good_encoder_activation_type: Type[nn.Module] | str = nn.GELU,
    good_encoder_activation_args: Dict[str, Any] | None = None,
    good_encoder_normalization_type: Type[nn.Module] | str | None = nn.LayerNorm,
    good_encoder_normalization_args: Dict[str, Any] | None = None,
    good_encoder_dropout: float = 0.0,
    good_sender_input_dim: int = 64,
    good_sender_embedding_dim: int = 8,
    good_sender_hidden_size: int = 128,
    good_sender_num_layers: int = 2,
    good_sender_cell_type: Type[nn.Module] | str = nn.GRU,
    good_sender_cell_args: Dict[str, Any] | None = None,
    poor_encoder_embedding_dim: int = 8,
    poor_encoder_hidden_sizes: List[int] = [64],
    poor_encoder_activation_type: Type[nn.Module] | str = nn.GELU,
    poor_encoder_activation_args: Dict[str, Any] | None = None,
    poor_encoder_normalization_type: Type[nn.Module] | str | None = nn.LayerNorm,
    poor_encoder_normalization_args: Dict[str, Any] | None = None,
    poor_encoder_dropout: float = 0.0,
    poor_sender_input_dim: int = 64,
    poor_sender_embedding_dim: int = 8,
    poor_sender_hidden_size: int = 128,
    poor_sender_num_layers: int = 2,
    poor_sender_cell_type: Type[nn.Module] | str = nn.GRU,
    poor_sender_cell_args: Dict[str, Any] | None = None,
    decoder_hidden_sizes: List[int] = [64],
    decoder_activation_type: Type[nn.Module] | str = nn.GELU,
    decoder_activation_args: Dict[str, Any] | None = None,
    decoder_normalization_type: Type[nn.Module] | str | None = nn.LayerNorm,
    decoder_normalization_args: Dict[str, Any] | None = None,
    decoder_dropout: float = 0.0,
    receiver_output_dim: int = 64,
    receiver_embedding_dim: int = 8,
    receiver_hidden_size: int = 128,
    receiver_num_layers: int = 2,
    receiver_cell_type: Type[nn.Module] | str = nn.GRU,
    receiver_cell_args: Dict[str, Any] | None = None,
    receiver_instantly: bool = False,
    **train_args: Any,
) -> None:
    good_encoder = MultiDiscreteMLP(
        high=num_elems,
        n=num_attrs,
        output_dim=good_sender_input_dim,
        embedding_dim=good_encoder_embedding_dim,
        hidden_sizes=good_encoder_hidden_sizes,
        activation_type=good_encoder_activation_type,
        activation_args=good_encoder_activation_args,
        normalization_type=good_encoder_normalization_type,
        normalization_args=good_encoder_normalization_args,
        dropout=good_encoder_dropout,
    )
    good_sender = Sender(
        encoder=good_encoder,
        input_dim=good_sender_input_dim,
        vocab_size=vocab_size,
        length=max_len,
        embedding_dim=good_sender_embedding_dim,
        hidden_size=good_sender_hidden_size,
        num_layers=good_sender_num_layers,
        cell_type=good_sender_cell_type,
        cell_args=good_sender_cell_args,
    )
    poor_encoder = MultiDiscreteMLP(
        high=num_elems,
        n=num_attrs,
        output_dim=poor_sender_input_dim,
        embedding_dim=poor_encoder_embedding_dim,
        hidden_sizes=poor_encoder_hidden_sizes,
        activation_type=poor_encoder_activation_type,
        activation_args=poor_encoder_activation_args,
        normalization_type=poor_encoder_normalization_type,
        normalization_args=poor_encoder_normalization_args,
        dropout=poor_encoder_dropout,
    )
    poor_sender = Sender(
        encoder=poor_encoder,
        input_dim=poor_sender_input_dim,
        vocab_size=vocab_size,
        length=max_len,
        embedding_dim=poor_sender_embedding_dim,
        hidden_size=poor_sender_hidden_size,
        num_layers=poor_sender_num_layers,
        cell_type=poor_sender_cell_type,
        cell_args=poor_sender_cell_args,
    )
    decoder = MLP(
        input_dim=receiver_output_dim,
        output_dim=num_elems * num_attrs,
        hidden_sizes=decoder_hidden_sizes,
        activation_type=decoder_activation_type,
        activation_args=decoder_activation_args,
        normalization_type=decoder_normalization_type,
        normalization_args=decoder_normalization_args,
        dropout=decoder_dropout,
    )
    receiver = Receiver(
        decoder=decoder,
        vocab_size=vocab_size,
        output_dim=receiver_output_dim,
        num_elems=num_elems,
        num_attrs=num_attrs,
        embedding_dim=receiver_embedding_dim,
        hidden_size=receiver_hidden_size,
        num_layers=receiver_num_layers,
        cell_type=receiver_cell_type,
        cell_args=receiver_cell_args,
        instantly=receiver_instantly,
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
    wandb_sweep(train_complete, conf, args.sweep_id, prefix="signet/")
