import os
from pathlib import Path
from typing import Any, Dict

from torch import Tensor
from torch.distributions import Categorical

from jargon.core import Batch
from jargon.net.functional import drop_padding
from jargon.utils.analysis import topographic_similarity


class Metrics:
    def __init__(
        self,
        num_elems: int,
        num_attrs: int,
        vocab_size: int,
        max_len: int,
        eos: int = 0,
    ) -> None:
        self.num_elems = num_elems
        self.num_attrs = num_attrs
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.eos = eos
        self.y_processor = lambda x: drop_padding(x, eos=self.eos)

    def __call__(self, batch: Batch) -> Dict[str, Any]:
        # [batch, num_attrs; int]
        output = batch.get_tensor("output")
        # [batch, num_attrs; int]
        target = batch.get_tensor("target")
        # [batch, max_len; int]
        message = batch.get_tensor("message")
        # [batch, max_len, vocab_size; float]
        msg_logits = batch.get_tensor("message_logits")
        # [batch, max_len; int]
        msg_mask = batch.get_tensor("message_mask")
        # [batch; int]
        msg_length = batch.get_tensor("message_length")

        # [batch, num_attrs; bool]
        acc_flag = output == target
        # [batch; float]
        acc_comp = acc_flag.all(-1).float()
        # [batch; float]
        acc_part = acc_flag.mean(-1).float()

        distr_s = Categorical(
            logits=msg_logits.reshape(-1, self.max_len, self.vocab_size)
        )
        # [batch, max_len; float]
        entropy_s = distr_s.entropy() * msg_mask

        # [batch; float]
        msg_length = msg_length.float()

        unique: float = message.unique(dim=0).shape[0] / message.shape[0]

        return {
            "acc/comp.mean": acc_comp.mean().item(),
            "acc/part.mean": acc_part.mean().item(),
            "msg/entropy.mean": entropy_s.mean().item(),
            "msg/length.mean": msg_length.mean().item(),
            "msg/unique": unique,
            # "acc/comp.std": acc_comp.std().item(),
            # "acc/part.std": acc_part.std().item(),
            # "msg/entropy.std": entropy_s.std().item(),
            # "msg/length.std": msg_length.std().item(),
        }

    def heavy_test(self, batch: Batch) -> Dict[str, Any]:
        # [batch, num_attrs; int]
        input = batch.get_tensor("input")
        # [batch, max_len; int]
        message = batch.get_tensor("message")

        input = input.cpu().numpy()
        message = message.cpu().numpy()
        topsims = {
            f"topsim/topsim": topographic_similarity(
                input, message, y_processor=self.y_processor
            )
        }

        return topsims


class LanguageMetrics:
    def __init__(self, log_dir: Path, eos: int = 0) -> None:
        self.log_dir = log_dir / "langs"
        self.eos = eos
        os.makedirs(self.log_dir, exist_ok=True)

    def __call__(self, batch: Batch, epoch: int) -> Dict[str, float]:
        # [batch, num_attrs; int]
        input = batch.get_tensor("input")
        # [batch, max_len; int]
        message = batch.get_tensor("message")

        inp_lines = [",".join([str(y) for y in x]) for x in input.tolist()]
        msg_list = message.tolist()
        msg_list = [
            x[: x.index(self.eos) if self.eos in x else len(x)] for x in msg_list
        ]
        msg_list = ["-".join([str(y) for y in x]) for x in msg_list]
        lines = [",".join([x, y]) for x, y in zip(inp_lines, msg_list)]
        lang = "\n".join(lines)

        path = self.log_dir / f"{epoch:0>8}.csv"
        with open(path, "a") as f:
            f.write(lang + "\n")
