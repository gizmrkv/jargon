from copy import deepcopy
from typing import Any, Dict, List, Type

from torch import nn

from jargon.game import SignalingNetworkGame
from jargon.net import MLP, MultiDiscreteMLP, Receiver, Sender
from jargon.zoo.signet.loss import Loss
from jargon.zoo.signet_imitation.train_imitation import train_imitation


def train_imitation_oneway(
    num_elems: int = 50,
    num_attrs: int = 2,
    vocab_size: int = 50,
    max_len: int = 8,
    num_senders: int = 6,
    num_receivers: int = 1,
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

    senders = {f"S{i}": deepcopy(sender) for i in range(num_senders)}
    receivers = {f"R{i}": deepcopy(receiver) for i in range(num_receivers)}

    assert num_receivers == 1, "Only implemented when there is one receiver."

    network = {s: {r for r in receivers} for s in senders}
    adaptation_targets = {s: {r for r in receivers} for s in senders}
    adaptation_targets |= {r: {s for s in senders} for r in receivers}

    imitation_triggers = {s: {r for r in receivers} for s in senders}
    imitation_targets = {}
    for i in range(num_senders - 1):
        imitation_targets[f"S{i}"] = {f"S{i+1}"}

    game = SignalingNetworkGame(senders, receivers, network)
    train_imitation(
        game=game,
        adaptation_targets=adaptation_targets,
        imitation_targets=imitation_targets,
        imitation_triggers=imitation_triggers,
        num_elems=num_elems,
        num_attrs=num_attrs,
        vocab_size=vocab_size,
        max_len=max_len,
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
    wandb_sweep(train_imitation_oneway, conf, args.sweep_id)
