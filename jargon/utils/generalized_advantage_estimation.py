import numpy as np
from numba import njit


def generalized_advantage_estimation(
    reward: np.ndarray,
    value: np.ndarray,
    done: np.ndarray,
    discount_factor: float = 0.99,
    trace_decay: float = 0.97,
    centering: bool = True,
    scaling: bool = True,
) -> np.ndarray:
    advantage = generalized_advantage_estimation_jit(
        reward, value, done, discount_factor, trace_decay
    )
    if centering:
        advantage -= advantage.mean(axis=0, keepdims=True)
    if scaling:
        advantage /= advantage.std(axis=0, keepdims=True) + 1e-8

    return advantage


@njit(cache=True, fastmath=True)
def generalized_advantage_estimation_jit(
    reward: np.ndarray,
    value: np.ndarray,
    done: np.ndarray,
    discount_factor: float = 0.99,
    trace_decay: float = 0.97,
) -> np.ndarray:
    assert reward.shape == done.shape
    done = ~done
    gamma = np.float32(discount_factor)
    lam = np.float32(trace_decay)
    advantages = []
    next_value = np.zeros_like(reward[:, 0], dtype=np.float32)
    advantage = np.zeros_like(reward[:, 0], dtype=np.float32)
    batch_size, num_steps = reward.shape
    for i in range(num_steps - 1, -1, -1):
        td_error = reward[:, i] + gamma * next_value * done[:, i] - value[:, i]
        advantage = td_error + gamma * lam * advantage
        advantage = td_error
        next_value = value[:, i]
        advantages.insert(0, advantage)

    stacked_advantages = np.empty((num_steps, batch_size), dtype=np.float32)
    for i in range(len(advantages)):
        stacked_advantages[i] = advantages[i]

    return stacked_advantages.transpose()
