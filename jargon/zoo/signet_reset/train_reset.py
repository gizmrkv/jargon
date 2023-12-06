import random
from typing import Any

from jargon.game import SignalingNetworkGame
from jargon.utils import init_weights
from jargon.zoo.signet.train import train


def train_reset(
    game: SignalingNetworkGame,
    reset_senders_interval: int | None = None,
    reset_receivers_interval: int | None = None,
    **train_args: Any,
) -> None:
    assert (
        reset_senders_interval is None or reset_senders_interval > 0
    ), "reset_senders_interval must be None or > 0"
    assert (
        reset_receivers_interval is None or reset_receivers_interval > 0
    ), "reset_receivers_interval must be None or > 0"

    senders = list(game.senders.values())
    receivers = list(game.receivers.values())

    def reset(epoch: int) -> None:
        if reset_senders_interval is not None:
            if epoch % reset_senders_interval == 0:
                index = (epoch // reset_senders_interval) % len(senders)
                senders[index].apply(init_weights)

        if reset_receivers_interval is not None:
            if epoch % reset_receivers_interval == 0:
                index = (epoch // reset_receivers_interval) % len(receivers)
                receivers[index].apply(init_weights)

    train(
        game=game,
        epoch_begin_fn=reset,
        **train_args,
    )
