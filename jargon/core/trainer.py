import datetime
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from jargon.callback import Callback
from jargon.envs import Environment
from jargon.logger import Logger
from jargon.loss import LossSelector
from jargon.stopper import DummyEarlyStopper, EarlyStopper


class Trainer:
    def __init__(
        self,
        env: Environment,
        loss_selector: LossSelector,
        optimizer: optim.Optimizer,
        loggers: Optional[List[Logger]] = None,
        stopper: Optional[EarlyStopper] = None,
        callbacks: Optional[Dict[str, Callback]] = None,
        checkpoint_interval: Optional[int] = None,
        checkpoint_path: Optional[Path] = None,
        loss_max_norm: float = 0.5,
        use_tqdm: bool = True,
    ):
        self.env = env
        self.loss_selector = loss_selector
        self.optimizer = optimizer
        self.loggers = loggers or []
        self.stopper = stopper or DummyEarlyStopper()
        self.callbacks = callbacks or {}
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_path = checkpoint_path
        self.loss_max_norm = loss_max_norm
        self.use_tqdm = use_tqdm

        assert checkpoint_interval is None or checkpoint_interval >= 1

        dt = datetime.datetime.now()
        self.run_name = dt.strftime("%Y-%m-%d %H-%M-%S-%f")
        self.log_dir = Path(f"logs/{self.run_name}")
        os.makedirs(self.log_dir, exist_ok=True)

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def train(self, epochs: int):
        if epochs % 10 == 0:
            epochs += 1

        for logger in self.loggers:
            logger.begin(self.run_name)

        train_begin_log_dir = self.log_dir / "begin"
        for name, callback in self.callbacks.items():
            callback.on_train_begin(train_begin_log_dir / name)

        zfill_width = len(str(epochs))
        with tqdm(range(epochs), disable=not self.use_tqdm) as pbar:
            for epoch in pbar:
                epoch_log_dir = self.log_dir / str(epoch).zfill(zfill_width)
                for name, callback in self.callbacks.items():
                    callback.on_epoch_begin(epoch, epoch_log_dir / name)

                batches_iter = self.env.rollout()
                loss_list = []
                while True:
                    for callback in self.callbacks.values():
                        callback.on_batch_begin(epoch)

                    try:
                        batches = next(batches_iter)
                    except StopIteration:
                        break

                    self.env.train()
                    losses = self.loss_selector.forward(batches)
                    losses = losses.apply(lambda l: l.mean())
                    loss = torch.stack([l for l in losses.values()]).sum()
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.env.parameters(), self.loss_max_norm)
                    self.optimizer.step()

                    loss_list.append(loss.item())

                    for callback in self.callbacks.values():
                        callback.on_batch_end(epoch, batches, losses)

                for name, callback in self.callbacks.items():
                    callback.on_epoch_end(epoch, epoch_log_dir / name)

                for logger in self.loggers:
                    logger.flush()

                if (
                    self.checkpoint_interval is not None
                    and epoch % self.checkpoint_interval == 0
                ):
                    self.save_checkpoint(epoch_log_dir / "checkpoint.bin")

                if self.stopper.step(sum(loss_list) / len(loss_list)):
                    for callback in self.callbacks.values():
                        callback.on_early_end(epoch, epoch_log_dir)

                    break

        train_end_log_dir = self.log_dir / "end"
        for name, callback in self.callbacks.items():
            callback.on_train_end(train_end_log_dir / name)

        for logger in self.loggers:
            logger.close()

    def save_checkpoint(self, checkpoint_path: Path):
        checkpoint = {
            "env": self.env.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            # "scheduler": scheduler.state_dict(),
            # "amp": amp.state_dict(), # apex混合精度を使用する場合は必要
            "random": random.getstate(),
            "np_random": np.random.get_state(),  # numpy.randomを使用する場合は必要
            "torch": torch.get_rng_state(),
            "torch_random": torch.random.get_rng_state(),
            "cuda_random": torch.cuda.get_rng_state(),  # gpuを使用する場合は必要
            "cuda_random_all": torch.cuda.get_rng_state_all(),  # 複数gpuを使用する場合は必要
        }
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path)
        self.env.load_state_dict(checkpoint["env"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        # scheduler.load_state_dict(checkpoint["scheduler"])
        # amp.load_state_dict(checkpoint["amp"])
        random.setstate(checkpoint["random"])
        np.random.set_state(checkpoint["np_random"])
        torch.set_rng_state(checkpoint["torch"])
        torch.random.set_rng_state(checkpoint["torch_random"])
        torch.cuda.set_rng_state(checkpoint["cuda_random"])
        torch.cuda.torch.cuda.set_rng_state_all(checkpoint["cuda_random_all"])
