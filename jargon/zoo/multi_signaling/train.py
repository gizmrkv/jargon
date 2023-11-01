import itertools
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping

import torch
from numpy.typing import NDArray
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, TensorDataset

from jargon.core import Batch, Trainer
from jargon.game import MultiSignalingGame
from jargon.net import MLP, MultiDiscreteMLP, Receiver, Sender
from jargon.utils import BaseLogger, fix_seed, init_weights, random_split
from jargon.zoo.multi_signaling.loss import ExplorationLoss
from jargon.zoo.multi_signaling.metrics import (
    accuracy_metrics,
    langsim_metrics,
    loss_metrics,
    message_metrics,
    topsim_metrics,
)


@dataclass
class Result:
    epoch: int
    elapsed_time: float
    train_dataset: Tensor
    test_dataset: Tensor
    game: MultiSignalingGame
    loss_fn: Callable[[Batch], Tensor]
    metrics_fn: Callable[[Batch], Dict[str, Any]]


def main(
    seed: int = 42,
    device: str = "cuda",
    num_elems: int = 50,
    num_attrs: int = 2,
    train_proportion: float = 0.8,
    test_proportion: float = 0.2,
    batch_size: int = 4096,
    vocab_size: int = 50,
    max_len: int = 8,
    sender_input_dim: int = 128,
    receiver_output_dim: int = 128,
    encoder_args: Mapping[str, Any]
    | None = {
        "embedding_dim": 8,
        "hidden_sizes": [64],
        "activation_type": "GELU",
        "activation_args": None,
        "normalization_type": "LayerNorm",
        "normalization_args": None,
        "dropout": 0.1,
    },
    sender_args: Mapping[str, Any]
    | None = {
        "embedding_dim": 8,
        "hidden_size": 200,
        "num_layers": 2,
        "cell_type": "GRU",
        "cell_args": None,
    },
    receiver_args: Mapping[str, Any]
    | None = {
        "embedding_dim": 8,
        "hidden_size": 200,
        "num_layers": 2,
        "cell_type": "GRU",
        "cell_args": None,
    },
    decoder_args: Mapping[str, Any]
    | None = {
        "hidden_sizes": [64],
        "activation_type": "GELU",
        "activation_args": None,
        "normalization_type": "LayerNorm",
        "normalization_args": None,
        "dropout": 0.1,
    },
    lr: float = 1e-3,
    discount_factor: float = 0.99,
    entropy_loss_weight: float = 0.0,
    length_loss_weight: float = 0.0,
    logger: BaseLogger | None = None,
    max_epochs: int = 3000,
    test_per_epoch: int = 10,
    show_progress: bool = True,
    use_amp: bool = True,
) -> Result:
    fix_seed(seed)

    if "cuda" in device and not torch.cuda.is_available():
        print("CUDA is not available. Use CPU instead.")
        device = "cpu"

    dataset = (
        torch.Tensor(list(itertools.product(torch.arange(num_elems), repeat=num_attrs)))
        .long()
        .to(device)
    )
    train_dataset, test_dataset = random_split(
        dataset, [train_proportion, test_proportion]
    )

    train_dataloader = DataLoader(
        TensorDataset(train_dataset, train_dataset),
        batch_size=batch_size,
        shuffle=True,
    )

    encoder_args = encoder_args or {}
    encoder = MultiDiscreteMLP(
        high=num_elems,
        n=num_attrs,
        output_dim=sender_input_dim,
        **encoder_args,
    )
    sender_args = sender_args or {}
    sender = Sender(
        encoder=encoder,
        input_dim=sender_input_dim,
        vocab_size=vocab_size,
        length=max_len,
        **sender_args,
    )

    decoder_args = decoder_args or {}
    decoder = MLP(
        input_dim=receiver_output_dim,
        output_dim=num_attrs * num_elems,
        **decoder_args,
    )
    receiver_args = receiver_args or {}
    receiver = Receiver(
        decoder=decoder,
        vocab_size=vocab_size,
        output_dim=receiver_output_dim,
        **receiver_args,
    )

    num_senders = 2
    num_receivers = 2
    senders = {f"S{i}": deepcopy(sender) for i in range(num_senders)}
    receivers = {f"R{i}": deepcopy(receiver) for i in range(num_receivers)}
    channels = {
        "S0": {"R0"},
        "S1": {"R1"},
    }
    # channels_fully = {s: {r for r in receivers} for s in senders}
    # channels_co = {s: {r for r in receivers if r not in channels[s]} for s in senders}
    # channels_empty = {s: set[str]() for s in senders}

    game = MultiSignalingGame(senders, receivers, channels).to(device)
    game.apply(init_weights)
    optimizer = optim.Adam(game.parameters(), lr=lr)

    exploration_targets = {
        "S0": {"R0"},
        "S1": {"R1"},
        "R0": {"S0"},
        "R1": {"S1"},
    }

    loss_fn = ExplorationLoss(
        num_elems=num_elems,
        num_attrs=num_attrs,
        vocab_size=vocab_size,
        max_len=max_len,
        game=game,
        discount_factor=discount_factor,
        exploration_targets=exploration_targets,
    )

    def metrics_fn(batch: Batch) -> Dict[str, Any]:
        metrics = accuracy_metrics(batch, num_elems, num_attrs)
        metrics |= message_metrics(batch, vocab_size, max_len)
        metrics |= loss_metrics(batch, loss_fn)
        metrics |= topsim_metrics(batch)
        metrics |= langsim_metrics(batch)
        return metrics

    def test_fn(epoch: int) -> None:
        metrics = {
            f"train/{k}": v
            for k, v in metrics_fn(game(train_dataset, train_dataset)).items()
        }
        metrics |= {
            f"test/{k}": v
            for k, v in metrics_fn(game(test_dataset, test_dataset)).items()
        }
        if logger is not None:
            logger.log(epoch, metrics)

    trainer = Trainer(
        model=game,
        loss_fn=loss_fn,
        optim=optimizer,
        max_epochs=max_epochs,
        dataloader=train_dataloader,
        test_per_epoch=test_per_epoch,
        test_fn=test_fn if logger else None,
        show_progress=show_progress,
        use_amp=use_amp,
    )
    epoch, elapsed_time = trainer.run()

    if logger is not None:
        logger.close()

    return Result(
        epoch=epoch,
        elapsed_time=elapsed_time,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        game=game,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
    )


if __name__ == "__main__":
    from jargon.utils.logger import WandbLogger

    torch.backends.cudnn.deterministic = True

    logger = WandbLogger(project="jargon")
    result = main(logger=logger, max_epochs=2000, num_elems=50)
    print(result)
