import torch
from torch import Tensor


def padding_mask(message: Tensor, eos: int = 0) -> Tensor:
    """Create a mask for the padding in the message.

    Parameters
    ----------
    message : Tensor
        The message to create the mask for.

    Returns
    -------
    Tensor
        The mask for the padding in the message.
        0 is placed at the position to pad, and 1 is placed at the other positions.

    Examples
    --------
    >>> message = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
    >>> padding_mask(message)
    tensor([[1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0]])
    """
    mask = message == eos
    indices = torch.argmax(mask.int(), dim=1)
    no_mask = ~mask.any(dim=1)
    indices[no_mask] = message.shape[1]
    mask = torch.arange(message.shape[1]).expand(message.shape).to(message.device)
    mask = (mask <= indices.unsqueeze(-1)).long()
    return mask
