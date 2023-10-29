import torch
from torch import Tensor, nn

from ..core import Batch


class SignalingGame(nn.Module):
    """Signaling game

    A signaling game is a game with two agents: a sender and a receiver.
    The sender receives an input and sends a message.
    The receiver receives a message and produces an output.

    en: The goal of the receiver is to guess the target based on the message sent by the sender.
    The goal of the sender is to send a message that allows the receiver to guess the target.

    Messages are represented as Tensor[batch_size, max_len; int].
    The 0 in the message is an end-of-sequence symbol, and the rest is padded with 0.

    Parameters:
    -----------
    sender : nn.Module
        The agent that sends the message.
        Messages are represented as Tensor[batch_size, max_len; int].
        The 0 in the message is an end-of-sequence symbol, and the rest is padded with 0.

    receiver : nn.Module
        The agent that receives the message and produces an output.
    """

    def __init__(self, sender: nn.Module, receiver: nn.Module) -> None:
        super().__init__()
        self.sender = sender
        self.receiver = receiver

    def forward(self, input: Tensor, target: Tensor) -> Batch:
        message, msg_logits = self.sender(input)
        msg_mask = padding_mask(message)
        message = message * msg_mask
        msg_length = msg_mask.sum(dim=-1)
        output = self.receiver(message)
        batch = Batch(
            input=input,
            message=message,
            message_logits=msg_logits,
            message_mask=msg_mask,
            message_length=msg_length,
            output=output,
            target=target,
        )

        return batch


def padding_mask(message: Tensor) -> Tensor:
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
    mask = message == 0
    indices = torch.argmax(mask.int(), dim=1)
    no_mask = ~mask.any(dim=1)
    indices[no_mask] = message.shape[1]
    mask = torch.arange(message.shape[1]).expand(message.shape).to(message.device)
    mask = (mask <= indices.unsqueeze(-1)).long()
    return mask
