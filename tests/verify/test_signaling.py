import datetime
import itertools
from typing import Any, Dict, Literal

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
from jargon.utils import BaseLogger, WandbLogger, fix_seed, init_weights, random_split
from jargon.utils.analysis import topographic_similarity


def _run_signaling(
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
        batch_size=2**12,
        shuffle=True,
    )

    vocab_size = 50
    max_len = 8
    encoder_embedding_dim = 8
    encoder_hidden_sizes = [64]
    sender_embedding_dim = 8
    sender_hidden_size = 128
    sender_num_layers = 2
    sender_cell_type: Literal["rnn", "lstm", "gru"] = "gru"
    encoder = MultiDiscreteMLP(
        num_elems,
        num_attrs,
        sender_hidden_size,
        encoder_embedding_dim,
        encoder_hidden_sizes,
        nn.GELU(),
        dropout=0.1,
    )
    sender = Sender(
        encoder,
        vocab_size,
        max_len,
        sender_embedding_dim,
        sender_hidden_size,
        sender_num_layers,
        sender_cell_type,
    )

    decoder_hidden_sizes = [64]
    receiver_embedding_dim = 8
    receiver_hidden_size = 128
    receiver_num_layers = 2
    receiver_cell_type: Literal["rnn", "lstm", "gru"] = "gru"
    decoder = MLP(
        receiver_hidden_size,
        num_attrs * num_elems,
        decoder_hidden_sizes,
        nn.GELU(),
        dropout=0.1,
    )
    receiver = Receiver(
        decoder,
        vocab_size,
        receiver_embedding_dim,
        receiver_hidden_size,
        receiver_num_layers,
        receiver_cell_type,
    )

    game = SignalingGame(sender, receiver).to(device)
    game.apply(init_weights)
    optimizer = optim.Adam(game.parameters(), lr=1e-3)

    def loss_fn(batch: Batch) -> Tensor:
        output: Tensor = batch.output  # type: ignore
        target: Tensor = batch.target  # type: ignore
        message: Tensor = batch.message  # type: ignore
        msg_logits: Tensor = batch.message_aux.logits  # type: ignore
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

        loss_ent = 0.0 * distr.entropy() * msg_mask

        length = msg_length.float() / max_len
        loss_len = 0.0 * pg_loss(log_prob.sum(-1).unsqueeze(1), -length.unsqueeze(1))

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
        msg_logits: Tensor = batch.message_aux.logits  # type: ignore
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
        max_epochs=3000,
        dataloader=train_dataloader,
        test_per_epoch=50,
        test_fn=test_fn,
    )
    epoch, elapsed_time = trainer.run()
    print(f"Epoch: {epoch}, Elapsed time: {elapsed_time:0.2f} seconds")

    metrics = {"epoch": epoch}
    metrics |= {
        f"train/{k}": v
        for k, v in metrics_fn(game(train_dataset, train_dataset)).items()
    }
    metrics |= {
        f"test/{k}": v for k, v in metrics_fn(game(test_dataset, test_dataset)).items()
    }

    return metrics


def test_signaling() -> None:
    for _ in range(5):
        dt = datetime.datetime.now()
        name = dt.strftime("%Y/%m/%d %H:%M:%S.%f")
        name += " signaling"
        logger = WandbLogger("signaling/", project="tests", name=name)
        metrics = _run_signaling(logger=logger)
        logger.close()
        assert metrics["train/acc_part.mean"] > 0.7
        assert metrics["test/acc_part.mean"] > 0.6
