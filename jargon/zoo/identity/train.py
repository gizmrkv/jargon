import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Mapping

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


@dataclass
class Result:
    epoch: int
    elapsed_time: float
    train_dataset: Tensor
    test_dataset: Tensor
    game: SupervisedGame
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
    model_args: Mapping[str, Any]
    | None = {
        "embedding_dim": 8,
        "hidden_sizes": [64, 64],
        "activation_type": "ReLU",
        "activation_args": None,
        "normalization_type": "LayerNorm",
        "normalization_args": None,
        "dropout": 0.0,
    },
    lr: float = 1e-3,
    loss_type: Literal["sv", "pg"] = "sv",
    logger: BaseLogger | None = None,
    max_epochs: int = 1000,
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

    model_args = model_args or {}
    model = MultiDiscreteMLP(
        high=num_elems,
        n=num_attrs,
        output_dim=num_elems * num_attrs,
        **model_args,
    )

    game = SupervisedGame(model).to(device)
    game.apply(init_weights)
    optimizer = optim.Adam(game.parameters(), lr=lr)

    def loss_fn(batch: Batch) -> Tensor:
        output: Tensor = batch.output  # type: ignore
        target: Tensor = batch.target  # type: ignore

        if loss_type == "sv":
            output = output.reshape(-1, num_elems)
            target = target.reshape(-1)
            loss = F.cross_entropy(output, target, reduction="none")
            loss = loss.reshape(-1, num_attrs).mean(-1)
            return loss
        elif loss_type == "pg":
            distr = Categorical(logits=output)
            if game.training:
                action = distr.sample()
            else:
                action = output.argmax(-1)
            log_prob = distr.log_prob(action)
            reward = (action == target).float()
            return pg_loss(log_prob, reward).mean(-1)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Use 'sv' or 'pg'")

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

    if logger:
        logger = logger
    else:
        logger = DummyLogger()

    def test_fn(epoch: int) -> None:
        metrics = {
            f"train/{k}": v
            for k, v in metrics_fn(game(train_dataset, train_dataset)).items()
        }
        metrics |= {
            f"test/{k}": v
            for k, v in metrics_fn(game(test_dataset, test_dataset)).items()
        }
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
