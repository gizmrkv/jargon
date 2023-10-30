import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor, nn, optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset

from jargon.core import Batch, Trainer
from jargon.game import SignalingGame
from jargon.net import MLP, MultiDiscreteMLP, Receiver, Sender
from jargon.net.loss import pg_loss
from jargon.utils import BaseLogger, fix_seed, init_weights, random_split
from jargon.utils.analysis import topographic_similarity


@dataclass
class Result:
    epoch: int
    elapsed_time: float
    train_dataset: Tensor
    test_dataset: Tensor
    game: SignalingGame
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
    max_len: int = 10,
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
        "dropout": 0.0,
    },
    sender_args: Mapping[str, Any]
    | None = {
        "embedding_dim": 8,
        "hidden_size": 64,
        "num_layers": 2,
        "cell_type": "GRU",
        "cell_args": None,
    },
    receiver_args: Mapping[str, Any]
    | None = {
        "embedding_dim": 8,
        "hidden_size": 64,
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
        "dropout": 0.0,
    },
    lr: float = 1e-3,
    entropy_loss_weight: float = 0.0,
    length_loss_weight: float = 0.0,
    logger: BaseLogger | None = None,
    max_epochs: int = 3000,
    test_per_epoch: int = 20,
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

    game = SignalingGame(sender, receiver).to(device)
    game.apply(init_weights)
    optimizer = optim.Adam(game.parameters(), lr=lr)

    def loss_fn(batch: Batch) -> Tensor:
        output: Tensor = batch.output  # type: ignore
        target: Tensor = batch.target  # type: ignore
        message: Tensor = batch.message  # type: ignore
        msg_logits: Tensor = batch.message_logits  # type: ignore
        msg_length: Tensor = batch.message_length  # type: ignore
        msg_mask: Tensor = batch.message_mask  # type: ignore

        output = output.reshape(-1, num_elems)
        target = target.unsqueeze(1).expand(-1, max_len, -1).reshape(-1)
        loss_r = F.cross_entropy(output, target, reduction="none")
        loss_r = loss_r.reshape(-1, max_len, num_attrs).mean(-1)

        distr = Categorical(logits=msg_logits)
        log_prob = distr.log_prob(message)
        log_prob = log_prob * msg_mask

        loss_s = pg_loss(log_prob, -loss_r.detach(), 1.0)

        loss_ent = entropy_loss_weight * distr.entropy() * msg_mask

        length = msg_length.float() / max_len
        loss_len = length_loss_weight * pg_loss(
            log_prob.sum(-1).unsqueeze(1), -length.unsqueeze(1)
        )

        loss = loss_s + loss_r
        loss += loss_len
        loss += loss_ent
        loss = loss.mean(-1)
        return loss

    def metrics_fn(batch: Batch) -> Dict[str, Any]:
        input: Tensor = batch.input  # type: ignore
        output: Tensor = batch.output  # type: ignore
        target: Tensor = batch.target  # type: ignore
        message: Tensor = batch.message  # type: ignore
        msg_logits: Tensor = batch.message_logits  # type: ignore
        msg_mask: Tensor = batch.message_mask  # type: ignore
        msg_length: Tensor = batch.message_length  # type: ignore
        loss = loss_fn(batch)

        output = output[:, -1, :]
        acc_flag = output.reshape(-1, num_attrs, num_elems).argmax(-1) == target
        acc_comp = acc_flag.all(-1).float()
        acc_part = acc_flag.float()

        distr_r = Categorical(logits=output)
        entropy_r: Tensor = distr_r.entropy()

        distr_s = Categorical(logits=msg_logits.reshape(-1, max_len, vocab_size))
        entropy_s = distr_s.entropy() * msg_mask

        msg_length = msg_length.float()

        unique = message.unique(dim=0).shape[0] / message.shape[0]

        def drop_padding(x: NDArray[np.int32]) -> NDArray[np.int32]:
            i = np.argwhere(x == 0)
            return x if len(i) == 0 else x[: i[0, 0]]

        topsim = topographic_similarity(
            input.cpu().numpy(),
            message.cpu().numpy(),
            y_processor=drop_padding,  # type: ignore
        )

        return {
            "loss.mean": loss.mean().item(),
            "acc_comp.mean": acc_comp.mean().item(),
            "acc_part.mean": acc_part.mean().item(),
            "entropy_r.mean": entropy_r.mean().item(),
            "entropy_s.mean": entropy_s.mean().item(),
            "length.mean": msg_length.mean().item(),
            "loss.std": loss.std().item(),
            "acc_comp.std": acc_comp.std().item(),
            "acc_part.std": acc_part.std().item(),
            "entropy_r.std": entropy_r.std().item(),
            "entropy_s.std": entropy_s.std().item(),
            "length.std": msg_length.std().item(),
            "unique": unique,
            "topsim": topsim,
        }

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
