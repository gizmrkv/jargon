import numpy as np
from numba import njit


def discounted_return(
    reward: np.ndarray,
    done: np.ndarray,
    discount_factor: float,
    centering: bool = True,
    scaling: bool = True,
    epsilon: float = 1e-8,
) -> np.ndarray:
    ret = discounted_return_jit(reward, done, discount_factor)

    if centering:
        ret -= ret.mean(axis=0, keepdims=True)
    if scaling:
        ret /= ret.std(axis=0, keepdims=True) + epsilon

    return ret


@njit(cache=True, fastmath=True)
def discounted_return_jit(
    reward: np.ndarray,
    done: np.ndarray,
    discount_factor: float,
) -> np.ndarray:
    assert reward.shape == done.shape
    done = ~done
    gamma = np.float32(discount_factor)
    returns = []
    ret = np.zeros_like(reward[:, 0])
    batch_size, num_steps = reward.shape
    for i in range(num_steps - 1, -1, -1):
        ret = reward[:, i] + gamma * ret * done[:, i]
        returns.insert(0, ret)

    stacked_returns = np.empty((num_steps, batch_size), dtype=np.float32)
    for i in range(len(returns)):
        stacked_returns[i] = returns[i]

    return stacked_returns.transpose()
