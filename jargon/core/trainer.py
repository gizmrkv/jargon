import time
from typing import Any, Callable, Iterable, Tuple

import torch
import tqdm
from torch import Tensor, nn, optim

from .batch import Batch


class Trainer:
    """A simple trainer for PyTorch models.

    en: This class is a simple trainer for PyTorch models. It is designed to
    be as simple as possible, while still being flexible enough to be used
    in a wide variety of situations.

    Example
    -------
    >>> from torch.utils.data import DataLoader
    >>>
    >>> class Model(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.linear = nn.Linear(10, 1)
    ...
    ...     def forward(self, x: Tensor) -> Batch:
    ...         return Batch(y=self.linear(x))
    >>>
    >>> def loss_fn(batch: Batch) -> Tensor:
    ...     return batch.y
    >>>
    >>> model = Model()
    >>> optim = optim.Adam(model.parameters())
    >>> dataloader = DataLoader(torch.randn(100, 10), batch_size=10)
    >>> trainer = Trainer(
    ...     model=model,
    ...     loss_fn=loss_fn,
    ...     optim=optim,
    ...     max_epochs=10,
    ...     dataloader=dataloader,
    ...     use_amp=False,
    ... )
    >>> epoch, elapsed_time = trainer.run()
    >>> epoch
    10
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[[Batch], Tensor],
        optim: optim.Optimizer,
        max_epochs: int,
        dataloader: Iterable[Any],
        test_per_epoch: int = 1,
        test_fn: Callable[[int], None] | None = None,
        early_stop: Callable[[int], bool] | None = None,
        show_progress: bool = True,
        use_amp: bool = True,
        epoch_begin_fn: Callable[[int], None] | None = None,
        epoch_end_fn: Callable[[int], None] | None = None,
    ) -> None:
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn
        self.max_epochs = max_epochs
        self.dataloader = dataloader
        self.test_per_epoch = test_per_epoch
        self.test_fn = test_fn
        self.early_stop = early_stop
        self.show_progress = show_progress
        self.epoch_begin_fn = epoch_begin_fn
        self.epoch_end_fn = epoch_end_fn

        if use_amp and not torch.cuda.is_available():
            print("CUDA is not available. Use CPU instead.")
            use_amp = False

        self.use_amp = use_amp
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else DummyGradScaler()

    def run(self) -> Tuple[int, float]:
        """Run the training loop.

        Returns
        -------
        Tuple[int, float]
            The number of epochs run and the elapsed time in seconds.
        """
        progress = tqdm.tqdm if self.show_progress else DummyTqdm

        prog = progress(total=self.max_epochs)
        losses: list[Tensor] = []

        elapsed_time = time.time()
        loss_min = 1e10
        loss_min_epoch = 0
        for epoch in range(self.max_epochs):
            self.model.train()
            losses.clear()

            if self.epoch_begin_fn:
                self.epoch_begin_fn(epoch)

            for data in self.dataloader:
                self.optim.zero_grad()
                with torch.autocast(self.device_type, enabled=bool(self.use_amp)):
                    batch: Batch = self._call_model(data)
                    loss = self.loss_fn(batch)
                    losses.append(loss)
                    loss = loss.mean()

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optim)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optim)
                self.scaler.update()

            loss = torch.cat(losses)
            loss_mean = loss.mean().item()
            if loss_mean < loss_min:
                loss_min = loss_mean
                loss_min_epoch = epoch
            prog.set_postfix(
                mean=f"{loss_mean:0.3f}",
                min=f"{loss_min:0.3f}",
                min_epoch=loss_min_epoch,
            )
            prog.set_description(f"Epoch #{epoch}")
            prog.update()

            if self.test_fn and epoch % self.test_per_epoch == 0:
                with torch.no_grad():
                    self.model.eval()
                    self.test_fn(epoch)

            if self.early_stop and self.early_stop(epoch):
                break

            if self.epoch_end_fn:
                self.epoch_end_fn(epoch)

        prog.close()
        elapsed_time = time.time() - elapsed_time

        return epoch + 1, elapsed_time

    def _call_model(self, data: Any) -> Batch:
        if isinstance(data, (list, tuple)):
            return self.model(*data)
        elif isinstance(data, dict):
            return self.model(**data)
        else:
            return self.model(data)


class DummyTqdm:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_postfix(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_description(self, desc: str) -> None:
        pass

    def update(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass


class DummyGradScaler:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def scale(self, loss: Tensor) -> Tensor:
        return loss

    def unscale_(self, optim: optim.Optimizer) -> None:
        pass

    def step(self, optim: optim.Optimizer) -> None:
        optim.step()

    def update(self) -> None:
        pass
