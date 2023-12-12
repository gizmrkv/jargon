import itertools
from typing import Any, Callable, Dict

import torch
from torch import Tensor, optim
from torch.utils.data import DataLoader, TensorDataset

from jargon.core import Batch, Trainer
from jargon.game import SignalingNetworkGame
from jargon.utils import (
    BaseLogger,
    DummyLogger,
    fix_seed,
    init_weights,
    make_log_dir,
    random_split,
)

from .metrics import LanguageMetrics, Metrics


def train(
    game: SignalingNetworkGame,
    loss_fn: Callable[[Batch], Tensor],
    num_elems: int = 50,
    num_attrs: int = 2,
    train_proportion: float = 0.8,
    test_proportion: float = 0.2,
    vocab_size: int = 50,
    max_len: int = 8,
    fix_len: bool = False,
    max_epochs: int = 5001,
    batch_size: int = 65536,
    lr: float = 1e-3,
    test_per_epoch: int = 25,
    heavy_test_per_test: int = 8,
    seed: int | None = None,
    show_progress: bool = True,
    use_amp: bool = False,
    logger: BaseLogger = DummyLogger(),
    additional_metrics_fn: Callable[[Batch], Dict[str, Any]] = lambda _: {},
    epoch_begin_fn: Callable[[int], None] | None = None,
    epoch_end_fn: Callable[[int], None] | None = None,
    device: torch.device | str | None = None,
) -> Dict[str, Any]:
    if seed is not None:
        fix_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = make_log_dir()

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

    torch.save(game, log_dir / "initial.pth")

    game = game.to(device)
    game.apply(init_weights)
    optimizer = optim.Adam(game.parameters(), lr=lr)

    game_comp = SignalingNetworkGame(
        senders=game.senders,
        receivers=game.receivers,
        network={s: {r for r in game.receivers} for s in game.senders},
    )

    eos = -1 if fix_len else 0
    game.eos = eos
    game_comp.eos = eos
    metrics_train_fn = Metrics(
        num_elems=num_elems,
        num_attrs=num_attrs,
        vocab_size=vocab_size,
        max_len=max_len,
        senders=game.senders,
        receivers=game.receivers,
        log_dir=log_dir / "train",
        eos=eos,
    )
    metrics_test_fn = Metrics(
        num_elems=num_elems,
        num_attrs=num_attrs,
        vocab_size=vocab_size,
        max_len=max_len,
        senders=game.senders,
        receivers=game.receivers,
        log_dir=log_dir / "test",
        eos=eos,
    )
    lang_metrics_fn = LanguageMetrics(game.senders, log_dir, eos)

    def test_fn(epoch: int) -> None:
        train_batch = game_comp(train_dataset, train_dataset)
        test_batch = game_comp(test_dataset, test_dataset)

        train_metrics = metrics_train_fn(train_batch)
        test_metrics = metrics_test_fn(test_batch)

        if (epoch // test_per_epoch) % heavy_test_per_test == 0:
            train_metrics |= metrics_train_fn.heavy_test(
                train_batch, train_metrics, epoch
            )
            test_metrics |= metrics_test_fn.heavy_test(test_batch, test_metrics, epoch)

            batch = game_comp(dataset, dataset)
            lang_metrics_fn(batch, epoch)

        if additional_metrics_fn is not None:
            train_metrics |= additional_metrics_fn(train_batch)
            test_metrics |= additional_metrics_fn(test_batch)

        metrics = {}
        metrics |= {f"train/{k}": v for k, v in train_metrics.items()}
        metrics |= {f"test/{k}": v for k, v in test_metrics.items()}

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
        epoch_begin_fn=epoch_begin_fn,
        epoch_end_fn=epoch_end_fn,
    )
    epoch, elapsed_time = trainer.run()

    torch.save(game, log_dir / "final.pth")

    movies = {
        f"train/movie/{k}": v for k, v in metrics_train_fn.frames_to_movies().items()
    }
    movies |= {
        f"test/movie/{k}": v for k, v in metrics_test_fn.frames_to_movies().items()
    }

    logger.log_movies(epoch, movies)
    logger.flush()
    logger.close()

    return {
        "epoch": epoch,
        "elapsed_time": elapsed_time,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
    }
