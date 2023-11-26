from typing import Dict, Mapping, Set

from torch import Tensor, nn

from ..core import Batch
from ..net.functional import padding_mask


class SignalingNetworkGame(nn.Module):
    def __init__(
        self,
        senders: Mapping[str, nn.Module],
        receivers: Mapping[str, nn.Module],
        network: Mapping[str, Set[str]],
        eos: int = 0,
    ) -> None:
        super().__init__()
        self.senders = nn.ModuleDict(senders)
        self.receivers = nn.ModuleDict(receivers)
        self.network = network
        self.eos = eos

    def forward(self, input: Tensor, target: Tensor) -> Batch:
        messages = {}
        messages_logits = {}
        messages_mask = {}
        messages_length = {}
        for sender_name, sender in self.senders.items():
            message, msg_logits = sender(input)
            msg_mask = padding_mask(message, eos=self.eos)
            message = message * msg_mask
            msg_length = msg_mask.sum(dim=-1)
            messages[sender_name] = message
            messages_logits[sender_name] = msg_logits
            messages_mask[sender_name] = msg_mask
            messages_length[sender_name] = msg_length

        outputs: Dict[str, Dict[str, Tensor]] = {
            sender_name: {} for sender_name in self.senders.keys()
        }
        outputs_logits: Dict[str, Dict[str, Tensor]] = {
            sender_name: {} for sender_name in self.senders.keys()
        }
        for name_s, names_r in self.network.items():
            for name_r in names_r:
                output, logits = self.receivers[name_r](messages[name_s])
                outputs[name_s][name_r] = output
                outputs_logits[name_s][name_r] = logits

        batch = Batch(
            input=input,
            messages=messages,
            messages_logits=messages_logits,
            messages_mask=messages_mask,
            messages_length=messages_length,
            outputs=outputs,
            outputs_logits=outputs_logits,
            target=target,
        )

        return batch
