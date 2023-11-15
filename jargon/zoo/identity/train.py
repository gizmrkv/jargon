import itertools
from typing import Any, Callable, Dict, List, Literal, Mapping, Set, Type

import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset

from jargon.core import Batch, Trainer
from jargon.game import SupervisedGame
from jargon.net import MultiDiscreteMLP
from jargon.net.loss import pg_loss
from jargon.utils import BaseLogger, DummyLogger, fix_seed, init_weights, random_split


def train(
    num_elems: int,
    num_attrs: int,
    train_proportion: float,
    test_proportion: float,
    game: SupervisedGame,
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

    def test_fn(epoch: int) -> None:
        if metrics_fn is None:
            return

        metrics = {
            f"train/{k}": v
            for k, v in metrics_fn(game(train_dataset, train_dataset)).items()
        }
        metrics |= {
            f"test/{k}": v
            for k, v in metrics_fn(game(test_dataset, test_dataset)).items()
        }
        logger.log(epoch, metrics)

    def metrics_fn(batch: Batch) -> Dict[str, Any]:
        output: Tensor = batch.output  # type: ignore
        target: Tensor = batch.target  # type: ignore
        loss = loss_fn(batch)

        acc_flag = output.reshape(-1, num_attrs, num_elems).argmax(-1) == target
        acc_comp = acc_flag.all(-1).float()
        acc_part = acc_flag.float()

        distr = Categorical(logits=output)
        entropy: Tensor = distr.entropy()

        metrics = {
            "loss.mean": loss.mean().item(),
            "acc_comp.mean": acc_comp.mean().item(),
            "acc_part.mean": acc_part.mean().item(),
            "entropy.mean": entropy.mean().item(),
            "loss.std": loss.std().item(),
            "acc_comp.std": acc_comp.std().item(),
            "acc_part.std": acc_part.std().item(),
            "entropy.std": entropy.std().item(),
        }
        metrics |= additional_metrics_fn(batch)
        return metrics

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
