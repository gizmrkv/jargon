from typing import Literal, Optional

import torch
from torch import nn, optim

from jargon.callback import MetricsCallback
from jargon.core import Trainer
from jargon.envs import AutoencoderEnv
from jargon.logger import DuplicateChecker, WandBLogger, wandb_sweep
from jargon.loss import LossSelector, TextCrossEntropyLoss
from jargon.net import RecurrentDecoder, RecurrentEncoder, RecurrentNet
from jargon.stopper import MinMaxEarlyStopper
from jargon.utils import fix_seed, init_weights
from jargon.utils.torchdict import TensorDict


def train_autoencoder(
    epochs: int,
    vocab_size: int,
    seq_length: int,
    batch_size: int,
    encoder_embed_size: int,
    encoder_hidden_size: int,
    decoder_embed_size: int,
    decoder_hidden_size: int,
    encoder_num_layers: int = 1,
    encoder_bias: bool = True,
    encoder_dropout: float = 0.0,
    encoder_bidirectional: bool = True,
    encoder_rnn_type: Literal["rnn", "lstm", "gru"] = "lstm",
    decoder_num_layers: int = 1,
    decoder_bias: bool = True,
    decoder_dropout: float = 0.0,
    decoder_rnn_type: Literal["rnn", "lstm", "gru"] = "lstm",
    decoder_train_temp: float = 1.0,
    decoder_eval_temp: float = 1e-5,
    decoder_peeky: bool = False,
    stopper_threshold: float = 1e-6,
    stopper_window_size: int = 10,
    lr: float = 1e-3,
    seed: Optional[int] = None,
    wandb_logger: Optional[WandBLogger] = None,
):
    if seed is not None:
        fix_seed(seed)

    rnn_types = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
    encoder = RecurrentEncoder(
        RecurrentNet(
            encoder_embed_size,
            encoder_hidden_size,
            num_layers=encoder_num_layers,
            bias=bool(encoder_bias),
            dropout=encoder_dropout,
            bidirectional=bool(encoder_bidirectional),
            rnn_type=rnn_types[encoder_rnn_type],
        ),
        vocab_size=vocab_size,
        embed_size=encoder_embed_size,
    )
    decoder = RecurrentDecoder(
        RecurrentNet(
            decoder_embed_size,
            decoder_hidden_size,
            num_layers=decoder_num_layers,
            bias=bool(decoder_bias),
            dropout=decoder_dropout,
            rnn_type=rnn_types[decoder_rnn_type],
        ),
        vocab_size=vocab_size,
        embed_size=decoder_embed_size,
        seq_length=seq_length,
        train_temp=decoder_train_temp,
        eval_temp=decoder_eval_temp,
        peeky=bool(decoder_peeky),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = AutoencoderEnv(
        encoder, decoder, vocab_size, seq_length, batch_size, device=device
    ).to(device)
    env.apply(init_weights)
    loss_selector = LossSelector(decoder=TextCrossEntropyLoss(vocab_size, seq_length))
    optimizer = optim.AdamW(env.parameters(), lr=lr)

    def metrics(batches: TensorDict) -> TensorDict:
        input = batches.get_tensor_dict("encoder")
        input = input.get_tensor("input")
        output = batches.get_tensor_dict("decoder")
        output = output.get_tensor("output")
        acc = (input == output).float().mean()
        return TensorDict(acc=acc)

    loggers = [DuplicateChecker()]
    if wandb_logger is not None:
        loggers.append(wandb_logger)

    callbacks = {"metrics": MetricsCallback(env, metrics, loggers, interval=5)}
    stopper = MinMaxEarlyStopper(stopper_threshold, stopper_window_size)
    trainer = Trainer(
        env,
        loss_selector,
        optimizer,
        loggers=loggers,
        callbacks=callbacks,
        stopper=stopper,
    )
    trainer.train(epochs)


if __name__ == "__main__":
    wandb_sweep(train_autoencoder)
