from torch import Tensor, nn

from ..functional import discounted_cumulative_reward


class PGLoss(nn.Module):
    """Policy gradient loss

    Calculate the loss from the log of the probability distribution of the action and the reward.
    The reward is converted to discounted cumulative reward.
    The mean of the reward between batches is used as the baseline.

    Parameters
    ----------
    gamma : float, optional
        The discount factor, by default 0.99

    Examples
    --------
    >>> import torch
    >>> log_prob = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.1]])
    >>> reward = torch.tensor([[0, 0, 1], [1, 0, 0]], dtype=torch.float)
    >>> PGLoss()(log_prob, reward)
    tensor([[-0.0621, -0.1281, -0.1980],
            [-0.1320,  0.3873,  0.1291]])
    """

    def __init__(self, gamma: float = 0.99) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, log_prob: Tensor, reward: Tensor) -> Tensor:
        assert (
            log_prob.dim() == reward.dim()
        ), "log_prob and reward must have the same number of dimensions."

        if log_prob.dim() == 2:
            reward = discounted_cumulative_reward(reward, self.gamma)

        reward = reward.detach()
        reward = reward - reward.mean().item()
        reward = reward / (reward.std().item() + 1e-8)

        loss = -log_prob * reward
        return loss


def pg_loss(log_prob: Tensor, reward: Tensor, gamma: float = 0.99) -> Tensor:
    return PGLoss(gamma)(log_prob, reward)
