from torch import Tensor, nn

from ..core import Batch
from ..net.functional import padding_mask


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

    def __init__(self, sender: nn.Module, receiver: nn.Module, eos: int = 0) -> None:
        super().__init__()
        self.sender = sender
        self.receiver = receiver
        self.eos = eos

    def forward(self, input: Tensor, target: Tensor) -> Batch:
        message, msg_logits = self.sender(input)
        msg_mask = padding_mask(message, eos=self.eos)
        message = message * msg_mask
        msg_length = msg_mask.sum(dim=-1)
        output, logits = self.receiver(message)
        batch = Batch(
            input=input,
            message=message,
            message_logits=msg_logits,
            message_mask=msg_mask,
            message_length=msg_length,
            output=output,
            output_logits=logits,
            target=target,
        )

        return batch
