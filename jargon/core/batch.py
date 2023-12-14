import pprint
from copy import deepcopy
from typing import Any, Dict, Iterable, Union

import torch
from torch import Tensor


class Batch:
    """The data structure used to store a batch of data.

    This is a simple wrapper around a dictionary of tensors. It is used to
    store a batch of data, and is used throughout the library.
    It supports addition and subtraction of batches of the same shape,
    as well as concatenation of batches of the same shape.

    Example:
    >>> batch = Batch(x=torch.zeros(2, 3), b=Batch(y=torch.ones(2, 2)))
    >>> batch
    Batch(
        x: tensor([[0., 0., 0.],
                   [0., 0., 0.]]),
        b: Batch(
               y: tensor([[1., 1.],
                          [1., 1.]]),
           ),
    )
    """

    def __init__(self, **kwargs: Union[Tensor, "Batch", dict[str, Any]]) -> None:
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def __deepcopy__(self, memo: dict[int, Any]) -> "Batch":
        return Batch(**{k: deepcopy(v, memo) for k, v in self.__dict__.items()})

    def __getattr__(self, key: str) -> Union[Tensor, "Batch"]:
        assert key in self.__dict__, f"{key} is not in {self.__dict__.keys()}"
        assert isinstance(
            self.__dict__[key], (Tensor, Batch)
        ), f"{key} is not a tensor or batch"
        return self.__dict__[key]

    def __setattr__(
        self, key: str, value: Union[Tensor, "Batch", dict[str, Any]]
    ) -> None:
        assert isinstance(
            value, (Tensor, Batch, dict)
        ), f"{key} is not a tensor, batch or dict, but {key} is {type(value).__name__}"
        if isinstance(value, dict):
            self.__dict__[key] = Batch(**value)
        else:
            self.__dict__[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def __getstate__(self) -> dict[str, Union[Tensor, "Batch"]]:
        return self.__dict__

    def __setstate__(self, state: dict[str, Union[Tensor, "Batch"]]) -> None:
        self.__dict__.update(state)

    def __getitem__(self, key: str) -> Union[Tensor, "Batch"]:
        assert key in self.__dict__, f"{key} is not in {self.__dict__.keys()}"
        assert isinstance(
            self.__dict__[key], (Tensor, Batch)
        ), f"{key} is not a tensor or batch"
        return self.__dict__[key]

    def get_tensor(self, key: str) -> Tensor:
        value = self.__dict__[key]
        assert isinstance(value, Tensor), f"{key} is not a tensor"
        return value

    def get_batch(self, key: str) -> "Batch":
        value = self.__dict__[key]
        assert isinstance(value, Batch), f"{key} is not a batch"
        return value

    def __iter__(self) -> Iterable[str]:
        return iter(self.__dict__)

    def __iadd__(self, rhs: "Batch") -> "Batch":
        self.__dict__.update({k: v + rhs[k] for k, v in self.__dict__.items()})
        return self

    def __add__(self, rhs: "Batch") -> "Batch":
        return Batch(**{k: v + rhs[k] for k, v in self.__dict__.items()})

    def __imul__(self, rhs: "Batch") -> "Batch":
        self.__dict__.update({k: v * rhs[k] for k, v in self.__dict__.items()})
        return self

    def __mul__(self, rhs: "Batch") -> "Batch":
        return Batch(**{k: v * rhs[k] for k, v in self.__dict__.items()})

    def __isub__(self, rhs: "Batch") -> "Batch":
        self.__dict__.update({k: v - rhs[k] for k, v in self.__dict__.items()})
        return self

    def __sub__(self, rhs: "Batch") -> "Batch":
        return Batch(**{k: v - rhs[k] for k, v in self.__dict__.items()})

    def __itruediv__(self, rhs: "Batch") -> "Batch":
        self.__dict__.update({k: v / rhs[k] for k, v in self.__dict__.items()})
        return self

    def __truediv__(self, rhs: "Batch") -> "Batch":
        return Batch(**{k: v / rhs[k] for k, v in self.__dict__.items()})

    def __floordiv__(self, rhs: "Batch") -> "Batch":
        return Batch(**{k: v // rhs[k] for k, v in self.__dict__.items()})

    def __ifloordiv__(self, rhs: "Batch") -> "Batch":
        self.__dict__.update({k: v // rhs[k] for k, v in self.__dict__.items()})
        return self

    def __repr__(self) -> str:
        self_str = "Batch(\n"
        for k, v in self.__dict__.items():
            rpl = "\n" + " " * (6 + len(k))
            name = pprint.pformat(v).replace("\n", rpl)
            self_str += f"    {k}: {name},\n"
        self_str += ")"
        return self_str

    def to(self, device: torch.device) -> "Batch":
        return Batch(**{k: v.to(device) for k, v in self.__dict__.items()})

    def keys(self) -> Iterable[str]:
        return self.__dict__.keys()

    def values(self) -> Iterable[Union[Tensor, "Batch"]]:
        return self.__dict__.values()

    def items(self) -> Iterable[tuple[str, Union[Tensor, "Batch"]]]:
        return self.__dict__.items()

    def __len__(self) -> int:
        return len(self.__dict__)

    def flatten_dict(self) -> Dict[str, Tensor]:
        flatted: Dict[str, Tensor] = {}
        for k, v in self.items():
            if isinstance(v, Batch):
                sub = v.flatten_dict()
                flatted |= {f"{k}.{k2}": v2 for k2, v2 in sub.items()}
            else:
                flatted[k] = v
        return flatted

    def flatten(self) -> "Batch":
        return Batch(**self.flatten_dict())

    def empty(self) -> bool:
        return len(self.__dict__) == 0

    def has_batch(self) -> bool:
        for v in self.values():
            if isinstance(v, Batch):
                return True
        return False

    def tensors(self) -> Iterable[Tensor]:
        for v in self.values():
            if isinstance(v, Tensor):
                yield v
            elif isinstance(v, Batch):
                yield from v.tensors()

    def batch_size(self) -> int:
        # return batch size of tensors in this batch
        return next(iter(self.tensors())).shape[0]


def cat(batches: Iterable[Batch], dim: int = 0) -> Batch:
    """Concatenate batches of the same shape.

    Parameters
    ----------
    batches : Iterable[Batch]
        batches to concatenate, must be of the same shape.

    dim : int, optional
        dimension to concatenate on, by default 0

    Returns
    -------
    Batch
        concatenated batch

    Example
    -------
    >>> batch1 = Batch(x=torch.randn(10, 3), b=Batch(y=torch.randn(10, 3)))
    >>> batch2 = Batch(x=torch.randn(10, 3), b=Batch(y=torch.randn(10, 3)))
    >>> batch3 = cat([batch1, batch2])
    >>> batch3.x.shape
    torch.Size([20, 3])
    """
    catted = {}
    sample = next(iter(batches))
    for k, v in sample.__dict__.items():
        if isinstance(v, Batch):
            catted[k] = cat([b[k] for b in batches])  # type: ignore
        if isinstance(v, Tensor):
            catted[k] = torch.cat([b[k] for b in batches], dim=dim)  # type: ignore

    return Batch(**catted)


def stack(batches: Iterable[Batch], dim: int = 0) -> Batch:
    """Stack batches of the same shape.

    Parameters
    ----------
    batches : Iterable[Batch]
        batches to stack, must be of the same shape.
    dim : int, optional
        dimension to stack on, by default 0

    Returns
    -------
    Batch
        stacked batch

    Example
    -------
    >>> import torch
    >>> from jargon.core import Batch, stack
    >>> batch1 = Batch(x=torch.randn(10, 3), b=Batch(y=torch.randn(10, 3)))
    >>> batch2 = Batch(x=torch.randn(10, 3), b=Batch(y=torch.randn(10, 3)))
    >>> batch3 = stack([batch1, batch2])
    >>> batch3.x.shape
    torch.Size([2, 10, 3])
    """
    stacked = {}
    sample = next(iter(batches))
    for k, v in sample.__dict__.items():
        if isinstance(v, Batch):
            stacked[k] = stack([b[k] for b in batches])  # type: ignore
        if isinstance(v, Tensor):
            stacked[k] = torch.stack([b[k] for b in batches], dim=dim)  # type: ignore

    return Batch(**stacked)
