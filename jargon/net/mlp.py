from typing import List, Tuple

from torch import Tensor, nn


class MLP(nn.Module):
    """Multi-layer perceptron

    Parameters
    ----------
    input_dim : int
        The dimension of the input.
    output_dim : int
        The dimension of the output.
    hidden_sizes : List[int]
        The sizes of the hidden layers.
    activation : nn.Module, optional
        The activation function, by default nn.ReLU()
    normalize : bool, optional
        Whether to apply layer normalization, by default False
    dropout : float, optional
        The dropout rate, by default 0.0

    Examples
    --------
    >>> import torch
    >>> mlp = MLP(10, 5, [20, 30, 40])
    >>> x = torch.randn(100, 10)
    >>> y, _ = mlp(x)
    >>> y.shape
    torch.Size([100, 5])
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: List[int],
        activation: nn.Module = nn.ReLU(),
        normalize: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.normalize = normalize
        self.dropout = dropout

        dims = [input_dim, *hidden_sizes]
        layers = []
        for i in range(len(dims) - 1):
            layers.extend(
                miniblock(dims[i], dims[i + 1], activation, normalize, dropout)
            )
        layers.append(nn.Linear(dims[-1], output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tuple[Tensor, None]:
        x = self.layers(x)
        return x, None


class MultiDiscreteMLP(nn.Module):
    """Multi-layer perceptron for multi-discrete space

    Apply MLP to a fixed-length tuple of discrete values.
    The input is passed through an Embedding layer and then concatenated.

    Parameters
    ----------
    high : int
        The upper bound of the discrete space.
    n : int
        The number of discrete values.
    output_dim : int
        The dimension of the output.
    embedding_dim : int
        The dimension of the embedding.
    hidden_sizes : List[int]
        The sizes of the hidden layers.
    activation : nn.Module, optional
        The activation function, by default nn.ReLU()
    normalize : bool, optional
        Whether to apply layer normalization, by default False
    dropout : float, optional
        The dropout rate, by default 0.0

    Examples
    --------
    >>> import torch
    >>> mlp = MultiDiscreteMLP(10, 3, 5, 20, [30, 40])
    >>> x = torch.randint(0, 10, (100, 3))
    >>> y, _ = mlp(x)
    >>> y.shape
    torch.Size([100, 5])
    """

    def __init__(
        self,
        high: int,
        n: int,
        output_dim: int,
        embedding_dim: int,
        hidden_sizes: List[int],
        activation: nn.Module = nn.ReLU(),
        normalize: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.high = high
        self.n = n
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.normalize = normalize
        self.dropout = dropout

        self.embedding = nn.Embedding(high, embedding_dim)
        self.mlp = MLP(
            embedding_dim * n, output_dim, hidden_sizes, activation, normalize, dropout
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, None]:
        x = self.embedding(x)
        x = x.reshape(x.shape[0], -1)
        x, _ = self.mlp(x)
        return x, None


def miniblock(
    input_dim: int,
    output_dim: int,
    activation: nn.Module = nn.ReLU(),
    normalize: bool = False,
    dropout: float = 0.0,
) -> List[nn.Module]:
    layers: List[nn.Module] = []
    layers.append(nn.Linear(input_dim, output_dim))

    if normalize:
        layers.append(nn.LayerNorm(output_dim))

    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))

    layers.append(activation)
    return layers
