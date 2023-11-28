import random
from typing import Any

from jargon.game import SignalingNetworkGame
from jargon.utils import init_weights
from jargon.zoo.signet.train import train


def train_reset(
    game: SignalingNetworkGame,
    reset_senders: bool = False,
    sender_life_epoch_min: int | None = None,
    sender_life_epoch_max: int | None = None,
    reset_receivers: bool = False,
    receiver_life_epoch_min: int | None = None,
    receiver_life_epoch_max: int | None = None,
    **train_args: Any,
) -> None:
    age_s = {
        k: random.randint(sender_life_epoch_min, sender_life_epoch_max)
        for k in game.senders.keys()
    }
    age_r = {
        k: random.randint(receiver_life_epoch_min, receiver_life_epoch_max)
        for k in game.receivers.keys()
    }

    def reset(epoch: int) -> None:
        if reset_senders:
            for name, model in game.senders.items():
                if age_s[name] == 0:
                    model.apply(init_weights)
                    age_s[name] = random.randint(
                        sender_life_epoch_min, sender_life_epoch_max
                    )
                else:
                    age_s[name] -= 1

        if reset_receivers:
            for name, model in game.receivers.items():
                if age_r[name] == 0:
                    model.apply(init_weights)
                    age_r[name] = random.randint(
                        receiver_life_epoch_min, receiver_life_epoch_max
                    )
                else:
                    age_r[name] -= 1

    train(
        game=game,
        epoch_begin_fn=reset,
        **train_args,
    )
