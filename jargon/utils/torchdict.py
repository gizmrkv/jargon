import pprint
from copy import deepcopy
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch import Tensor

SizeDict = Dict[str, Union[torch.Size, "SizeDict"]]
IntDict = Dict[str, Union[int, "IntDict"]]


class TensorDict:
    def __init__(self, **kwargs: Union[Tensor, "TensorDict"]):
        self.__dict__["_raw_data"] = kwargs

    def __getattr__(self, key: str) -> Union[Tensor, "TensorDict", Callable]:
        assert key != "_raw_data", "Key cannot be '_raw_data'"
        return self._raw_data[key]

    def __setattr__(self, key: str, value: Union[Tensor, "TensorDict"]) -> None:
        assert key != "_raw_data", "Key cannot be '_raw_data'"
        self._raw_data[key] = value

    def __delattr__(self, key: str) -> None:
        assert key != "_raw_data", "Key cannot be '_raw_data'"
        del self._raw_data[key]

    def __getitem__(self, key: str) -> Union[Tensor, "TensorDict"]:
        assert key != "_raw_data", "Key cannot be '_raw_data'"
        return self._raw_data[key]

    def __setitem__(self, key: str, value: Union[Tensor, "TensorDict"]) -> None:
        assert key != "_raw_data", "Key cannot be '_raw_data'"
        self._raw_data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._raw_data[key]

    def get(
        self, key: str, default: Optional[Union[Tensor, "TensorDict"]] = None
    ) -> Optional[Union[Tensor, "TensorDict"]]:
        return self._raw_data.get(key, default)

    def setdefault(
        self, key: str, default: Union[Tensor, "TensorDict"]
    ) -> Union[Tensor, "TensorDict"]:
        return self._raw_data.setdefault(key, default)

    def pop(
        self, key: str, default: Optional[Union[Tensor, "TensorDict"]] = None
    ) -> Optional[Union[Tensor, "TensorDict"]]:
        return self._raw_data.pop(key, default)

    def popitem(self) -> Tuple[str, Union[Tensor, "TensorDict"]]:
        return self._raw_data.popitem()

    def clear(self) -> None:
        self._raw_data.clear()

    def copy(self) -> "TensorDict":
        return TensorDict(**self._raw_data.copy())

    def update(self, other: "TensorDict") -> None:
        self._raw_data.update(other)

    def fromkeys(
        self, keys: Union[str, Tuple[str, ...]], value: Union[Tensor, "TensorDict"]
    ) -> "TensorDict":
        return TensorDict(**dict.fromkeys(keys, value))

    def keys(self):
        return self._raw_data.keys()

    def values(self):
        return self._raw_data.values()

    def items(self):
        return self._raw_data.items()

    def empty(self) -> bool:
        return not bool(self._raw_data)

    def __len__(self) -> int:
        return len(self._raw_data)

    def __iter__(self):
        return iter(self._raw_data)

    def __contains__(self, key: str) -> bool:
        return key in self._raw_data

    def __deepcopy__(self, memo: Dict[int, object]) -> "TensorDict":
        return TensorDict(**deepcopy(self._raw_data, memo))

    def __getstate__(self) -> Dict[str, Union[Tensor, "TensorDict"]]:
        return self._raw_data

    def __setstate__(self, state: Dict[str, Union[Tensor, "TensorDict"]]) -> None:
        self._raw_data = state

    def __repr__(self) -> str:
        self_str = "TensorDict(\n"
        for k, v in self.items():
            rpl = "\n" + " " * (6 + len(k))
            name = pprint.pformat(v).replace("\n", rpl)
            self_str += f"    {k}: {name},\n"
        self_str += ")"
        return self_str

    def __pos__(self) -> "TensorDict":
        return self.apply(lambda t: +t)

    def __neg__(self) -> "TensorDict":
        return self.apply(lambda t: -t)

    def __abs__(self) -> "TensorDict":
        return self.apply(lambda t: abs(t))

    def __round__(self) -> "TensorDict":
        return self.apply(lambda t: round(t))

    def __eq__(self, other: object) -> "TensorDict":  # type: ignore
        return merge(self, other, lambda t1, t2: t1 == t2)

    def __ne__(self, other: object) -> "TensorDict":  # type: ignore
        return merge(self, other, lambda t1, t2: t1 != t2)

    def __add__(self, other: "TensorDict") -> "TensorDict":
        return merge(self, other, lambda t1, t2: t1 + t2)

    def __sub__(self, other: "TensorDict") -> "TensorDict":
        return merge(self, other, lambda t1, t2: t1 - t2)

    def __mul__(self, other: "TensorDict") -> "TensorDict":
        return merge(self, other, lambda t1, t2: t1 * t2)

    def __truediv__(self, other: "TensorDict") -> "TensorDict":
        return merge(self, other, lambda t1, t2: t1 / t2)

    def __floordiv__(self, other: "TensorDict") -> "TensorDict":
        return merge(self, other, lambda t1, t2: t1 // t2)

    def __mod__(self, other: "TensorDict") -> "TensorDict":
        return merge(self, other, lambda t1, t2: t1 % t2)

    def __pow__(self, other: "TensorDict") -> "TensorDict":
        return merge(self, other, lambda t1, t2: t1**t2)

    def apply(self, func: Callable[[Tensor], Tensor]) -> "TensorDict":
        return apply(self, func)

    def detach(self) -> "TensorDict":
        return detach(self)

    def to(self, device: torch.device) -> "TensorDict":
        return to(self, device)

    def cpu(self) -> "TensorDict":
        return cpu(self)

    def size(self) -> SizeDict:
        return {k: v.size() for k, v in self.items()}

    def dim(self) -> IntDict:
        return {k: v.dim() for k, v in self.items()}

    def flatten(self, separator: str = "_") -> "TensorDict":
        return flatten(self, separator)

    def get_tensor(self, key: str) -> Tensor:
        return get_tensor(self, key)

    def get_tensor_dict(self, key: str) -> "TensorDict":
        return get_tensor_dict(self, key)

    def tensors(self) -> Iterable[Tensor]:
        return tensors(self)

    def is_isomorphic(
        self,
        other: "TensorDict",
        check_tensor_shape: bool = True,
        check_tensor_dtype: bool = True,
    ) -> bool:
        return is_isomorphic(self, other, check_tensor_shape, check_tensor_dtype)


def apply(td: TensorDict, op: Callable[[Tensor], TensorDict]) -> TensorDict:
    ret = TensorDict()
    for k, v in td.items():
        if isinstance(v, Tensor):
            ret[k] = op(v)
        else:
            ret[k] = apply(v, op)
    return ret


def detach(td: TensorDict) -> TensorDict:
    return apply(td, lambda t: t.detach())


def to(td: TensorDict, device: torch.device) -> TensorDict:
    return apply(td, lambda t: t.to(device))


def cpu(td: TensorDict) -> TensorDict:
    return apply(td, lambda t: t.cpu())


def flatten(td: TensorDict, separator: str = "_") -> TensorDict:
    flat = TensorDict()
    for k, v in td.items():
        if isinstance(v, TensorDict):
            for k2, v2 in flatten(v, separator).items():
                flat[f"{k}{separator}{k2}"] = v2
        else:
            flat[k] = v
    return flat


def get_tensor(td: TensorDict, key: str) -> Tensor:
    ret = td.get(key)
    assert ret is not None, f"Key '{key}' not found in TensorDict"
    assert isinstance(ret, Tensor), f"Value at key '{key}' is not a Tensor"
    return ret


def get_tensor_dict(td: TensorDict, key: str) -> TensorDict:
    ret = td.get(key)
    assert ret is not None, f"Key '{key}' not found in TensorDict"
    assert isinstance(ret, TensorDict), f"Value at key '{key}' is not a TensorDict"
    return ret


def tensors(td: TensorDict) -> Iterable[Tensor]:
    for v in td.values():
        if isinstance(v, Tensor):
            yield v
        else:
            yield from tensors(v)


def is_isomorphic(
    td1: TensorDict,
    td2: TensorDict,
    check_tensor_shape: bool = True,
    check_tensor_dtype: bool = True,
) -> bool:
    if td1.keys() != td2.keys():
        return False

    for k, v1 in td1.items():
        v2 = td2[k]
        if isinstance(v1, TensorDict) and isinstance(v2, TensorDict):
            if not is_isomorphic(v1, v2, check_tensor_shape, check_tensor_dtype):
                return False
        elif isinstance(v1, Tensor) and isinstance(v2, Tensor):
            if check_tensor_shape and v1.size() != v2.size():
                return False
            if check_tensor_dtype and v1.dtype != v2.dtype:
                return False
        else:
            return False

    return True


def merge(
    td1: TensorDict, td2: TensorDict, op: Callable[[Tensor, Tensor], Tensor]
) -> TensorDict:
    assert is_isomorphic(
        td1, td2, check_tensor_shape=False, check_tensor_dtype=False
    ), "TensorDicts must have the same structure to be merged"
    ret = TensorDict()
    for k, v1 in td1.items():
        v2 = td2[k]
        if isinstance(v1, TensorDict):
            ret[k] = merge(v1, v2, op)
        else:
            ret[k] = op(v1, v2)

    return ret


def add(td1: TensorDict, td2: TensorDict) -> TensorDict:
    return merge(td1, td2, lambda t1, t2: t1 + t2)


def sub(td1: TensorDict, td2: TensorDict) -> TensorDict:
    return merge(td1, td2, lambda t1, t2: t1 - t2)


def mul(td1: TensorDict, td2: TensorDict) -> TensorDict:
    return merge(td1, td2, lambda t1, t2: t1 * t2)


def div(td1: TensorDict, td2: TensorDict) -> TensorDict:
    return merge(td1, td2, lambda t1, t2: t1 / t2)


def eq(td1: TensorDict, td2: TensorDict) -> TensorDict:
    return merge(td1, td2, lambda t1, t2: t1 == t2)


def ne(td1: TensorDict, td2: TensorDict) -> TensorDict:
    return merge(td1, td2, lambda t1, t2: t1 != t2)


def equal(td1: TensorDict, td2: TensorDict):
    return is_isomorphic(td1, td2) and all(
        t.all().item() for t in (td1 == td2).tensors()
    )


def is_all_isomorphic(
    tds: Iterable[TensorDict],
    check_tensor_shape: bool = True,
    check_tensor_dtype: bool = True,
) -> bool:
    td_iter = iter(tds)
    td_first = next(td_iter)
    return all(
        is_isomorphic(td, td_first, check_tensor_shape, check_tensor_dtype)
        for td in td_iter
    )


def cat(tds: Iterable[TensorDict], dim: int = 0) -> TensorDict:
    assert is_all_isomorphic(
        tds, check_tensor_shape=False
    ), "All TensorDicts must have the same structure to be concatenated"

    ret = TensorDict()
    sample = next(iter(tds))
    for k, v in sample.items():
        if isinstance(v, Tensor):
            ret[k] = torch.cat([td.get_tensor(k) for td in tds], dim=dim)
        else:
            ret[k] = cat([td[k] for td in tds], dim=dim)
    return ret


def stack(tds: Iterable[TensorDict], dim: int = 0) -> TensorDict:
    assert is_all_isomorphic(
        tds
    ), "All TensorDicts must have the same structure to be stacked"

    ret = TensorDict()
    sample = next(iter(tds))
    for k, v in sample.items():
        if isinstance(v, Tensor):
            ret[k] = torch.stack([td.get_tensor(k) for td in tds], dim=dim)
        else:
            ret[k] = stack([td[k] for td in tds], dim=dim)
    return ret
