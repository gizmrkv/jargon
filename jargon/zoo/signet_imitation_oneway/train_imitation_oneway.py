from copy import deepcopy
from typing import Any, Dict, Type

from torch import nn

from jargon.game import SignalingNetworkGame
from jargon.net import DiscreteReceiver, DiscreteSender
from jargon.zoo.signet_imitation.train_imitation import train_imitation


def train_imitation_oneway(
    num_elems: int = 50,
    num_attrs: int = 2,
    vocab_size: int = 50,
    max_len: int = 8,
    num_senders: int = 3,
    num_receivers: int = 1,
    instantly: bool = False,
    additional_imitation_edges: str = "",
    sender_input_embedding_dim: int = 16,
    sender_output_embedding_dim: int = 16,
    sender_hidden_size: int = 200,
    sender_num_layers: int = 1,
    sender_bidirectional: bool = False,
    sender_cell_type: Type[nn.Module] | str = nn.GRU,
    sender_cell_args: Dict[str, Any] | None = None,
    sender_peeky: bool = False,
    sender_attention: bool = False,
    sender_attention_dropout: float = 0.0,
    sender_attention_weight: bool = False,
    receiver_embedding_dim: int = 16,
    receiver_hidden_size: int = 200,
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

    senders = {f"S{i}": deepcopy(sender) for i in range(num_senders)}
    receivers = {f"R{i}": deepcopy(receiver) for i in range(num_receivers)}

    network = {s: {r for r in receivers} for s in senders}
    adaptation_targets = {s: {r for r in receivers} for s in senders}
    adaptation_targets |= {r: {s for s in senders} for r in receivers}

    imitation_triggers = {s: {r for r in receivers} for s in senders}
    imitation_targets = {s: set() for s in senders}
    for i in range(num_senders - 1):
        imitation_targets[f"S{i}"] = {f"S{i+1}"}

    if additional_imitation_edges:
        additional_imitation_edges = additional_imitation_edges.split(",")
        additional_imitation_edges = [e.split("->") for e in additional_imitation_edges]
        for s1, s2 in additional_imitation_edges:
            imitation_targets[s1].add(s2)

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
        instantly=instantly,
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
