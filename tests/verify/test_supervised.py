import datetime
import itertools
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import Tensor, optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset

from jargon.core import Batch, Trainer
from jargon.game import SupervisedGame
from jargon.net import MultiDiscreteMLP
from jargon.net.loss import pg_loss
from jargon.utils import BaseLogger, WandbLogger, fix_seed, init_weights, random_split


def _run_supervised(
    seed: int | None = None, logger: BaseLogger | None = None
) -> Dict[str, Any]:
    if seed is not None:
        fix_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_elems = 50
    num_attrs = 2
    dataset = (
        torch.Tensor(list(itertools.product(torch.arange(num_elems), repeat=num_attrs)))
        .long()
        .to(device)
    )
    train_dataset, test_dataset = random_split(dataset, [8, 2])
    train_target = train_dataset

    train_dataloader = DataLoader(
        TensorDataset(train_dataset, train_target),
        batch_size=1024,
        shuffle=True,
    )

    embedding_dim = 16
    hidden_sizes = [64, 64]
    model = MultiDiscreteMLP(
        num_elems, num_attrs, num_elems * num_attrs, embedding_dim, hidden_sizes
    )

    game = SupervisedGame(model).to(device)
    game.apply(init_weights)
    optimizer = optim.Adam(game.parameters(), lr=1e-3)

    def loss_fn(batch: Batch) -> Tensor:
        output: Tensor = batch.output  # type: ignore
        target: Tensor = batch.target  # type: ignore
        output = output.reshape(-1, num_elems)
        target = target.reshape(-1)
        loss = F.cross_entropy(output, target, reduction="none")
        loss = loss.reshape(-1, num_attrs).mean(-1)
        return loss

    def metrics_fn(batch: Batch) -> Dict[str, Any]:
        output: Tensor = batch.output  # type: ignore
        target: Tensor = batch.target  # type: ignore
        loss = loss_fn(batch)

        acc_flag = output.reshape(-1, num_attrs, num_elems).argmax(-1) == target
        acc_comp = acc_flag.all(-1).float()
        acc_part = acc_flag.float()

        distr = Categorical(logits=output)
        entropy: Tensor = distr.entropy()

        return {
            "loss.mean": loss.mean().item(),
            "acc_comp.mean": acc_comp.mean().item(),
            "acc_part.mean": acc_part.mean().item(),
            "entropy.mean": entropy.mean().item(),
            "loss.std": loss.std().item(),
            "acc_comp.std": acc_comp.std().item(),
            "acc_part.std": acc_part.std().item(),
            "entropy.std": entropy.std().item(),
        }

    def test_fn(epoch: int) -> None:
        metrics = {"epoch": epoch}
        metrics |= {
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
        max_epochs=200,
        dataloader=train_dataloader,
        test_per_epoch=10,
        test_fn=test_fn,
    )
    epoch, elapsed_time = trainer.run()

    metrics = {"epoch": epoch}
    metrics |= {
        f"train/{k}": v
        for k, v in metrics_fn(game(train_dataset, train_dataset)).items()
    }
    metrics |= {
        f"test/{k}": v for k, v in metrics_fn(game(test_dataset, test_dataset)).items()
    }

    return metrics


def _run_supervised_pg(
    seed: int | None = None, logger: BaseLogger | None = None
) -> Dict[str, Any]:
    if seed is not None:
        fix_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_elems = 10
    num_attrs = 2
    dataset = (
        torch.Tensor(list(itertools.product(torch.arange(num_elems), repeat=num_attrs)))
        .long()
        .to(device)
    )
    train_dataset, test_dataset = random_split(dataset, [8, 2])
    train_target = train_dataset

    train_dataloader = DataLoader(
        TensorDataset(train_dataset, train_target),
        batch_size=1024,
        shuffle=True,
    )

    embedding_dim = 16
    hidden_sizes = [64, 64]
    model = MultiDiscreteMLP(
        num_elems, num_attrs, num_elems * num_attrs, embedding_dim, hidden_sizes
    )

    game = SupervisedGame(model).to(device)
    game.apply(init_weights)
    optimizer = optim.Adam(game.parameters(), lr=1e-3)

    def loss_fn(batch: Batch, training: bool = True) -> Tensor:
        output: Tensor = batch.output  # type: ignore
        target: Tensor = batch.target  # type: ignore
        output = output.reshape(-1, num_attrs, num_elems)
        distr = Categorical(logits=output)
        if training:
            action = distr.sample()
        else:
            action = output.argmax(-1)
        log_prob = distr.log_prob(action)
        reward = (action == target).float()
        return pg_loss(log_prob, reward).mean(-1)

    def metrics_fn(batch: Batch) -> Dict[str, Any]:
        output: Tensor = batch.output  # type: ignore
        target: Tensor = batch.target  # type: ignore
        loss = loss_fn(batch)

        acc_flag = output.reshape(-1, num_attrs, num_elems).argmax(-1) == target
        acc_comp = acc_flag.all(-1).float()
        acc_part = acc_flag.float()

        distr = Categorical(logits=output)
        entropy: Tensor = distr.entropy()

        return {
            "loss.mean": loss.mean().item(),
            "acc_comp.mean": acc_comp.mean().item(),
            "acc_part.mean": acc_part.mean().item(),
            "entropy.mean": entropy.mean().item(),
            "loss.std": loss.std().item(),
            "acc_comp.std": acc_comp.std().item(),
            "acc_part.std": acc_part.std().item(),
            "entropy.std": entropy.std().item(),
        }

    def test_fn(epoch: int) -> None:
        metrics = {"epoch": epoch}
        metrics |= {
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
        max_epochs=400,
        dataloader=train_dataloader,
        test_per_epoch=10,
        test_fn=test_fn,
    )
    epoch, elapsed_time = trainer.run()

    metrics = {
        f"train/{k}": v
        for k, v in metrics_fn(game(train_dataset, train_dataset)).items()
    }
    metrics |= {
        f"test/{k}": v for k, v in metrics_fn(game(test_dataset, test_dataset)).items()
    }

    return metrics


def test_supervised() -> None:
    for _ in range(5):
        dt = datetime.datetime.now()
        name = dt.strftime("%Y/%m/%d %H:%M:%S.%f")
        name += " supervised"
        logger = WandbLogger("supervised_pg/", project="tests", name=name)
        metrics = _run_supervised(logger=logger)
        logger.close()
        assert metrics["train/acc_part.mean"] > 0.9
        assert metrics["test/acc_part.mean"] > 0.8


def test_supervised_pg() -> None:
    for _ in range(5):
        dt = datetime.datetime.now()
        name = dt.strftime("%Y/%m/%d %H:%M:%S.%f")
        name += " supervised_pg"
        logger = WandbLogger("supervised/", project="tests", name=name)
        metrics = _run_supervised_pg(logger=logger)
        logger.close()
        assert metrics["train/acc_part.mean"] > 0.9
        assert metrics["test/acc_part.mean"] > 0.8
