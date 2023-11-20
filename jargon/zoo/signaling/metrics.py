import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray
from torch import Tensor
from torch.distributions import Categorical

from jargon.core import Batch
from jargon.utils.analysis import topographic_similarity


class Metrics:
    def __init__(
        self,
        num_elems: int,
        num_attrs: int,
        vocab_size: int,
        max_len: int,
    ) -> None:
        self.num_elems = num_elems
        self.num_attrs = num_attrs
        self.vocab_size = vocab_size
        self.max_len = max_len

    def __call__(self, batch: Batch) -> Dict[str, Any]:
        output: Tensor = batch.output  # type: ignore
        target: Tensor = batch.target  # type: ignore
        message: Tensor = batch.message  # type: ignore
        msg_logits: Tensor = batch.message_logits  # type: ignore
        msg_mask: Tensor = batch.message_mask  # type: ignore
        msg_length: Tensor = batch.message_length  # type: ignore

        acc_flag = output == target
        acc_comp = acc_flag.all(-1).float()
        acc_part = acc_flag.float()

        distr_s = Categorical(
            logits=msg_logits.reshape(-1, self.max_len, self.vocab_size)
        )
        entropy_s = distr_s.entropy() * msg_mask

        msg_length = msg_length.float()

        unique = message.unique(dim=0).shape[0] / message.shape[0]

        return {
            "acc/comp.mean": acc_comp.mean().item(),
            "acc/comp.std": acc_comp.std().item(),
            "acc/part.mean": acc_part.mean().item(),
            "acc/part.std": acc_part.std().item(),
            "msg/entropy.mean": entropy_s.mean().item(),
            "msg/entropy.std": entropy_s.std().item(),
            "msg/length.mean": msg_length.mean().item(),
            "msg/length.std": msg_length.std().item(),
            "msg/unique": unique,
        }

    def heavy_test(self, batch: Batch) -> Dict[str, Any]:
        input: Tensor = batch.input
        message: Tensor = batch.message

        def drop_padding(x: NDArray[np.int32]) -> NDArray[np.int32]:
            i = np.argwhere(x == 0)
            return x if len(i) == 0 else x[: i[0, 0]]

        topsim = topographic_similarity(
            input.cpu().numpy(),
            message.cpu().numpy(),
            y_processor=drop_padding,  # type: ignore
        )

        return {
            "topsim/mean": topsim,
        }


class LanguageMetrics:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir / "langs"
        os.makedirs(self.log_dir, exist_ok=True)

    def __call__(self, batch: Batch, epoch: int) -> Dict[str, float]:
        input: Tensor = batch.input
        message: Tensor = batch.message

        inp_lines = [",".join([str(y) for y in x]) for x in input.tolist()]
        msg_list = message.tolist()
        msg_list = [x[: x.index(0) if 0 in x else len(x)] for x in msg_list]
        msg_list = ["-".join([str(y) for y in x]) for x in msg_list]
        lines = [",".join([x, y]) for x, y in zip(inp_lines, msg_list)]
        lang = "\n".join(lines)

        path = self.log_dir / f"{epoch:0>8}.csv"
        with open(path, "a") as f:
            f.write(lang + "\n")
