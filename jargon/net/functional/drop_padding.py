import numpy as np
from numpy.typing import NDArray


def drop_padding(x: NDArray[np.int32], eos: int = 0) -> NDArray[np.int32]:
    i = np.argwhere(x == eos)
    return x if len(i) == 0 else x[: i[0, 0]]
