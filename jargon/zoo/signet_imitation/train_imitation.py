from typing import Any, Dict, Set

from jargon.game import SignalingNetworkGame
from jargon.zoo.signet.loss import Loss
from jargon.zoo.signet.train import train
from jargon.zoo.signet_imitation.loss import ImitationLoss


def train_imitation(
    game: SignalingNetworkGame,
    adaptation_targets: Dict[str, Set[str]],
    imitation_targets: Dict[str, Set[str]],
    imitation_triggers: Dict[str, Set[str]],
    num_elems: int = 50,
    num_attrs: int = 2,
    vocab_size: int = 50,
    max_len: int = 8,
    entropy_loss_weight: float = 0.0,
    length_loss_weight: float = 0.0,
    discount_factor: float = 0.1,
    instantly: bool = False,
    imitation: bool = True,
    imitation_threshold: float = 0.9999,
    **train_args: Any,
) -> None:
    loss = Loss(
        num_elems=num_elems,
        num_attrs=num_attrs,
        vocab_size=vocab_size,
        max_len=max_len,
        game=game,
        entropy_loss_weight=entropy_loss_weight,
        length_loss_weight=length_loss_weight,
        adaptation_targets=adaptation_targets,
        discount_factor=discount_factor,
        instantly=instantly,
    )
    if imitation:
        loss = ImitationLoss(
            loss=loss,
            imitation_targets=imitation_targets,
            imitation_triggers=imitation_triggers,
            imitation_threshold=imitation_threshold,
        )
    train(
        game=game,
        loss_fn=loss,
        additional_metrics_fn=loss.metrics,
        **train_args,
    )
