from torch import Tensor, nn


def init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.RNN, nn.LSTM, nn.GRU)):
        if isinstance(m.weight_ih_l0, Tensor):
            nn.init.kaiming_uniform_(m.weight_ih_l0)
        if isinstance(m.weight_hh_l0, Tensor):
            nn.init.kaiming_uniform_(m.weight_hh_l0)
        if isinstance(m.bias_ih_l0, Tensor):
            nn.init.zeros_(m.bias_ih_l0)
        if isinstance(m.bias_hh_l0, Tensor):
            nn.init.zeros_(m.bias_hh_l0)
    elif isinstance(m, nn.Embedding):
        nn.init.kaiming_uniform_(m.weight)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
