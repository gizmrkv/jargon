from typing import Any

from jargon.game import SignalingGame
from jargon.utils import init_weights
from jargon.zoo.signaling.train import train


def train_reset(
    game: SignalingGame,
    reset_sender_per_epoch: int | None = None,
    reset_receiver_per_epoch: int | None = None,
    **train_args: Any,
) -> None:
    def reset(epoch: int) -> None:
        if reset_sender_per_epoch is not None and epoch % reset_sender_per_epoch == 0:
            game.sender.apply(init_weights)

        if (
            reset_receiver_per_epoch is not None
            and epoch % reset_receiver_per_epoch == 0
        ):
            game.receiver.apply(init_weights)

    train(
        game=game,
        epoch_begin_fn=reset,
        **train_args,
    )
