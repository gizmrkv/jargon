import torch
from torch import Tensor


def discounted_cumulative_reward(reward: Tensor, gamma: float) -> Tensor:
    """Calculate the discounted cumulative reward.

    Parameters
    ----------
    reward : Tensor
        The reward to calculate the discounted cumulative reward for.
        The size of the reward must be (batch_size, episode_length).
    gamma : float
        The discount factor.

    Returns
    -------
    Tensor
        The discounted cumulative reward.
        The size of the discounted cumulative reward is (batch_size, episode_length).

    Examples
    --------
    >>> reward = torch.tensor([[0, 0, 1, 1], [1, 0, 0, 1]], dtype=torch.float)
    >>> discounted_cumulative_reward(reward, 0.5)
    tensor([[0.3750, 0.7500, 1.5000, 1.0000],
            [1.1250, 0.2500, 0.5000, 1.0000]])
    """
    r = torch.zeros_like(reward)
    next = torch.zeros_like(reward[:, -1])
    for i in range(reward.shape[1] - 1, -1, -1):
        r[:, i] = reward[:, i] + gamma * next
        next = r[:, i]
    return r
