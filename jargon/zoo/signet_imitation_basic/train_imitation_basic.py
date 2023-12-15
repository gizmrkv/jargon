from copy import deepcopy
from typing import Any, Dict, Type

from torch import nn

from jargon.game import SignalingNetworkGame
from jargon.net import DiscreteReceiver, DiscreteSender
from jargon.zoo.signet_imitation.train_imitation import train_imitation


def train_imitation_basic(
    num_elems: int = 50,
    num_attrs: int = 2,
    vocab_size: int = 50,
    max_len: int = 8,
    num_senders: int = 3,
    num_receivers: int = 1,
    network_type: str = "fully",
    imitation_graph_type: str = "fully",
    sender_input_embedding_dim: int = 16,
    sender_output_embedding_dim: int = 16,
    sender_hidden_size: int = 200,
    sender_num_layers: int = 1,
    sender_cell_type: Type[nn.Module] | str = nn.GRU,
    sender_cell_args: Dict[str, Any] | None = None,
    sender_peeky: bool = False,
    sender_attention: bool = False,
    sender_attention_weight: bool = False,
    sender_attention_dropout: float = 0.0,
    receiver_embedding_dim: int = 16,
    receiver_hidden_size: int = 200,
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

    senders = {f"S{i}": deepcopy(sender) for i in range(num_senders)}
    receivers = {f"R{i}": deepcopy(receiver) for i in range(num_receivers)}

    if network_type == "fully":
        network = {s: {r for r in receivers} for s in senders}
        adaptation_targets = {s: {r for r in receivers} for s in senders}
        adaptation_targets |= {r: {s for s in senders} for r in receivers}
    elif network_type == "individual":
        assert (
            num_senders == num_receivers
        ), "Individual adaptation requires equal numbers of senders and receivers"
        network = {f"S{i}": {f"R{i}"} for i in range(num_senders)}
        adaptation_targets = {f"S{i}": {f"R{i}"} for i in range(num_senders)}
        adaptation_targets |= {f"R{i}": {f"S{i}"} for i in range(num_receivers)}
    else:
        raise ValueError(f"Unknown adaptation graph type {network_type}")

    if imitation_graph_type == "fully":
        imitation_triggers = {s: {r for r in receivers} for s in senders}
        imitation_targets = {s: {s2 for s2 in senders if s2 != s} for s in senders}
    elif imitation_graph_type == "oneway":
        imitation_triggers = {s: {r for r in receivers} for s in senders}
        imitation_targets = {s: set() for s in senders}
        for i in range(num_senders - 1):
            imitation_targets[f"S{i}"].add(f"S{i+1}")
    elif imitation_graph_type == "oneway2":
        imitation_triggers = {s: {r for r in receivers} for s in senders}
        imitation_targets = {s: set() for s in senders}
        for i in range(num_senders - 1):
            imitation_targets[f"S{i}"].add(f"S{i+1}")
            imitation_targets[f"S{i+1}"].add(f"S{i}")
    elif imitation_graph_type == "ring":
        imitation_triggers = {s: {r for r in receivers} for s in senders}
        imitation_targets = {s: set() for s in senders}
        for i in range(num_senders):
            imitation_targets[f"S{i}"].add(f"S{(i+1)%num_senders}")
    elif imitation_graph_type == "ring2":
        imitation_triggers = {s: {r for r in receivers} for s in senders}
        imitation_targets = {s: set() for s in senders}
        for i in range(num_senders):
            imitation_targets[f"S{i}"].add(f"S{(i+1)%num_senders}")
            imitation_targets[f"S{(i+1)%num_senders}"].add(f"S{i}")
    elif imitation_graph_type == "none":
        imitation_triggers = {s: set() for s in senders}
        imitation_targets = {s: set() for s in senders}
    else:
        raise ValueError(f"Unknown imitation graph type {imitation_graph_type}")

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
    from jargon.zoo.utils import wandb_sweep

    wandb_sweep(train_imitation_basic)
