from itertools import product
from typing import Iterator, Literal, Optional

import torch
from torch.utils.data import DataLoader

from jargon.net import Net
from jargon.utils import random_split
from jargon.utils.torchdict import TensorDict

from .environment import Environment


class AutoencoderEnv(Environment):
    def __init__(
        self,
        encoder: Net,
        decoder: Net,
        vocab_size: int,
        seq_length: int,
        batch_size: int,
        train_proportion: float = 0.8,
        test_proportion: float = 0.2,
        device: Optional[torch.device] = None,
        mode: Literal["train", "test"] = "train",
    ):
        super().__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.train_proportion = train_proportion
        self.test_proportion = test_proportion
        self.device = device
        self.mode = mode

        self.dataset = (
            torch.Tensor(list(product(torch.arange(vocab_size), repeat=seq_length)))
            .long()
            .to(device)
        )
        self.train_dataset, self.test_dataset = random_split(
            self.dataset, [train_proportion, test_proportion]
        )
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=True
        )

    def rollout(self) -> Iterator[TensorDict]:
        dataloader = (
            self.train_dataloader if self.mode == "train" else self.test_dataloader
        )
        for input in dataloader:
            encoded, enc_info = self.encoder.forward(input)
            decoded, dec_info = self.decoder.forward(encoded, TensorDict(target=input))
            yield TensorDict(
                encoder=TensorDict(input=input, output=encoded, info=enc_info),
                decoder=TensorDict(
                    input=encoded, output=decoded, target=input, info=dec_info
                ),
            )
