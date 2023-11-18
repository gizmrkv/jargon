from copy import deepcopy
from typing import Any, Dict, List, Type

from torch import nn

from jargon.game import SignalingNetworkGame
from jargon.net import MLP, MultiDiscreteMLP, Receiver, Sender
from jargon.zoo.signaling_network.loss import Loss
from jargon.zoo.signaling_network.train import train


def train_one_way(
    num_elems: int = 50,
    num_attrs: int = 2,
    vocab_size: int = 50,
    max_len: int = 8,
    num_agents: int = 3,
    ring: bool = False,
    entropy_loss_weight: float = 0.0,
    length_loss_weight: float = 0.0,
    imitation: bool = False,
    imitation_threshold: float = 0.99,
    encoder_embedding_dim: int = 8,
    encoder_hidden_sizes: List[int] = [64],
    encoder_activation_type: Type[nn.Module] | str = nn.GELU,
    encoder_activation_args: Dict[str, Any] | None = None,
    encoder_normalization_type: Type[nn.Module] | str | None = nn.LayerNorm,
    encoder_normalization_args: Dict[str, Any] | None = None,
    encoder_dropout: float = 0.0,
    sender_input_dim: int = 64,
    sender_embedding_dim: int = 8,
    sender_hidden_size: int = 128,
    sender_num_layers: int = 2,
    sender_cell_type: Type[nn.Module] | str = nn.GRU,
    sender_cell_args: Dict[str, Any] | None = None,
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
    **train_args: Any,
) -> None:
    encoder = MultiDiscreteMLP(
        high=num_elems,
        n=num_attrs,
        output_dim=sender_input_dim,
        embedding_dim=encoder_embedding_dim,
        hidden_sizes=encoder_hidden_sizes,
        activation_type=encoder_activation_type,
        activation_args=encoder_activation_args,
        normalization_type=encoder_normalization_type,
        normalization_args=encoder_normalization_args,
        dropout=encoder_dropout,
    )
    sender = Sender(
        encoder=encoder,
        input_dim=sender_input_dim,
        vocab_size=vocab_size,
        length=max_len,
        embedding_dim=sender_embedding_dim,
        hidden_size=sender_hidden_size,
        num_layers=sender_num_layers,
        cell_type=sender_cell_type,
        cell_args=sender_cell_args,
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
    )

    senders = {f"S{i}": deepcopy(sender) for i in range(num_agents)}
    receivers = {f"R{i}": deepcopy(receiver) for i in range(num_agents)}

    # network = {s: {r for r in receivers} for s in senders}
    # adaptation_targets = {s: {r for r in receivers} for s in senders}
    # adaptation_targets |= {r: {s for s in senders} for r in receivers}
    # if imitation:
    #     imitation_targets = {s1: {s2 for s2 in senders if s1 != s2} for s1 in senders}
    #     imitation_triggers = {s: {r for r in receivers} for s in senders}
    # else:
    #     imitation_targets = None
    #     imitation_triggers = None

    network = {f"S{i}": {} for i in range(num_agents)}
    for i in range(num_agents - 1):
        network[f"S{i}"] = {f"R{i}", f"R{i+1}"}
    network[f"S{num_agents-1}"] = {f"R{num_agents-1}"}

    adaptation_targets = {f"S{i}": {} for i in range(num_agents)}
    for i in range(num_agents - 1):
        adaptation_targets[f"S{i}"] = {f"R{i}", f"R{i+1}"}
    adaptation_targets[f"S{num_agents-1}"] = {f"R{num_agents-1}"}
    for i in range(num_agents - 1):
        adaptation_targets[f"R{i+1}"] = {f"S{i}", f"S{i+1}"}
    adaptation_targets["R0"] = {"S0"}

    if ring:
        network[f"S{num_agents-1}"] |= {"R0"}
        adaptation_targets[f"S{num_agents-1}"] |= {"R0"}
        adaptation_targets["R0"] |= {f"S{num_agents-1}"}

    if imitation:
        imitation_targets = {f"S{i}": {f"S{i+1}"} for i in range(num_agents - 1)}
        imitation_triggers = {f"S{i}": {f"R{i+1}"} for i in range(num_agents - 1)}

        if ring:
            imitation_targets[f"S{num_agents-1}"] = {"S0"}
            imitation_triggers[f"S{num_agents-1}"] = {"R0"}
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
    wandb_sweep(train_one_way, conf, args.sweep_id, prefix="signet/")
