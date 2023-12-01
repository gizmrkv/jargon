from typing import Any, Dict, List, Type

from torch import nn

from jargon.game import SignalingGame
from jargon.net import MLP, MultiDiscreteMLP, Receiver, Sender
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
    encoder_embedding_dim: int = 8,
    encoder_hidden_sizes: List[int] = [64],
    encoder_activation_type: Type[nn.Module] | str = nn.GELU,
    encoder_activation_args: Dict[str, Any] | None = None,
    encoder_normalization_type: Type[nn.Module] | str | None = nn.LayerNorm,
    encoder_normalization_args: Dict[str, Any] | None = None,
    encoder_dropout: float = 0.0,
    sender_input_dim: int = 64,
    sender_embedding_dim: int = 8,
    sender_hidden_size: int = 500,
    sender_num_layers: int = 1,
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
    receiver_hidden_size: int = 500,
    receiver_num_layers: int = 1,
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
