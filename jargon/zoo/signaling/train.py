import itertools
from typing import Any, Callable, Dict

import torch
from torch import Tensor, optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset

from jargon.core import Batch, Trainer
from jargon.game import SignalingGame
from jargon.utils import BaseLogger, DummyLogger, fix_seed, init_weights, random_split

from .metrics import Metrics


def train(
    num_elems: int,
    num_attrs: int,
    train_proportion: float,
    test_proportion: float,
    vocab_size: int,
    max_len: int,
    game: SignalingGame,
    loss_fn: Callable[[Batch], Tensor],
    max_epochs: int,
    batch_size: int,
    lr: float,
    test_per_epoch: int,
    seed: int | None = None,
    show_progress: bool = True,
    use_amp: bool = False,
    logger: BaseLogger = DummyLogger(),
    additional_metrics_fn: Callable[[Batch], Dict[str, Any]] = lambda _: {},
) -> Dict[str, Any]:
    if seed is not None:
        fix_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    game = game.to(device)
    game.apply(init_weights)
    optimizer = optim.Adam(game.parameters(), lr=lr)

    metrics_fn = Metrics(num_elems, num_attrs, vocab_size, max_len, loss_fn)

    def test_fn(epoch: int) -> None:
        train_batch = game(train_dataset, train_dataset)
        test_batch = game(test_dataset, test_dataset)

        metrics = {}
        metrics |= {f"train/{k}": v for k, v in metrics_fn(train_batch).items()}
        metrics |= {f"test/{k}": v for k, v in metrics_fn(test_batch).items()}

        if additional_metrics_fn is not None:
            metrics |= {
                f"train/{k}": v for k, v in additional_metrics_fn(train_batch).items()
            }
            metrics |= {
                f"test/{k}": v for k, v in additional_metrics_fn(test_batch).items()
            }

        logger.log(epoch, metrics)

    trainer = Trainer(
        model=game,
        loss_fn=loss_fn,
        optim=optimizer,
        max_epochs=max_epochs,
        dataloader=train_dataloader,
        test_per_epoch=test_per_epoch,
        test_fn=test_fn,
        show_progress=show_progress,
        use_amp=use_amp,
    )
    epoch, elapsed_time = trainer.run()

    logger.close()

    return {
        "epoch": epoch,
        "elapsed_time": elapsed_time,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "game": game,
    }
